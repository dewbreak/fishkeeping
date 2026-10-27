"""Verify paid-call bounds and redaction entirely with a mocked HTTP transport."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_spec = importlib.util.spec_from_file_location(
    "live_harness", Path(__file__).resolve().parents[1] / "scripts" / "live_deepseek.py"
)
_harness = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_harness)
MAX_CALLS = _harness.MAX_CALLS
MAX_OUTPUT_TOKENS = _harness.MAX_OUTPUT_TOKENS
MeteredTransport = _harness.MeteredTransport


def request_payload(**overrides):
    return {
        "model": "deepseek-flash",
        "max_tokens": MAX_OUTPUT_TOKENS,
        "thinking": {"type": "disabled"},
        "messages": [{"role": "user", "content": "private synthetic content"}],
        **overrides,
    }


def test_meter_preserves_usage_and_redacts_text():
    def reply(request):
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "private generated content",
                            "reasoning_content": "private reasoning",
                        },
                    }
                ],
            },
        )

    meter = MeteredTransport(httpx.MockTransport(reply))
    meter.phase = "test"
    with httpx.Client(transport=meter) as client:
        response = client.post(
            "https://api.deepseek.com/chat/completions",
            json=request_payload(),
            headers={"Authorization": "Bearer fake-test-secret"},
        )
    assert response.status_code == 200
    assert meter.calls[0]["usage"]["total_tokens"] == 25
    logged = json.dumps(meter.calls)
    assert "secret" not in logged and "private" not in logged and "reasoning_content" not in logged


@pytest.mark.parametrize(
    "change",
    [
        {"model": "deepseek-v4-pro"},
        {"thinking": {"type": "enabled"}},
        {"max_tokens": MAX_OUTPUT_TOKENS + 1},
    ],
)
def test_budget_rejects_expensive_requests_before_network(change):
    meter = MeteredTransport(httpx.MockTransport(lambda _: pytest.fail("network called")))
    with httpx.Client(transport=meter) as client, pytest.raises(httpx.TransportError):
        client.post("https://api.deepseek.com/chat/completions", json=request_payload(**change))
    assert not meter.calls


def test_call_count_limit_and_wrong_host_are_enforced():
    seen = []

    def reply(request):
        seen.append(request)
        return httpx.Response(429, json={"error": "private response"})

    meter = MeteredTransport(httpx.MockTransport(reply))
    with httpx.Client(transport=meter) as client:
        with pytest.raises(httpx.TransportError):
            client.post("https://wrong.example/chat/completions", json=request_payload())
        for _ in range(MAX_CALLS):
            assert (
                client.post(
                    "https://api.deepseek.com/chat/completions", json=request_payload()
                ).status_code
                == 429
            )
        with pytest.raises(httpx.TransportError):
            client.post("https://api.deepseek.com/chat/completions", json=request_payload())
    assert len(seen) == MAX_CALLS
    assert "private" not in json.dumps(meter.calls)
