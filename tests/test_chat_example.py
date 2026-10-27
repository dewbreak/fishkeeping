import importlib.util
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from fishkeeping.config import Settings


def example():
    spec = importlib.util.spec_from_file_location(
        "chat_example", Path(__file__).resolve().parents[1] / "examples" / "deepseek_chat.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_chat_example_uses_shared_settings_and_economical_payload(monkeypatch):
    module = example()
    configured = Settings(_env_file=None, deepseek_api_key="fake-test-key", deepseek_max_tokens=512)
    monkeypatch.setattr(module, "Settings", lambda: configured)
    monkeypatch.setattr("builtins.input", lambda _: "我的偏好是什么？")
    monkeypatch.setenv("MEMORY_EXTRACT", "false")
    seen = []
    events = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def context(self, *args, **kwargs):
            return SimpleNamespace(context="用户喜欢 Python")

        def enqueue(self, **kwargs):
            events.append(kwargs)

    def post(url, **kwargs):
        seen.append(kwargs)
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={"choices": [{"finish_reason": "stop", "message": {"content": "喜欢 Python"}}]},
        )

    monkeypatch.setattr(module, "MemoryClient", Client)
    monkeypatch.setattr(module.httpx, "post", post)
    module.main()
    assert seen[0]["json"]["thinking"] == {"type": "disabled"}
    assert seen[0]["json"]["max_tokens"] == 512
    assert seen[0]["json"]["model"] == "deepseek-flash"
    assert seen[0]["headers"]["Authorization"] == "Bearer fake-test-key"
    assert events[0]["extract"] is False
    assert events[0]["messages"][0]["role"] == "user"


def test_chat_example_missing_key_stops_before_prompt_or_network(monkeypatch):
    module = example()
    monkeypatch.setattr(module, "Settings", lambda: Settings(_env_file=None, deepseek_api_key=None))
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("input requested"))
    with pytest.raises(SystemExit, match="Configure"):
        module.main()
