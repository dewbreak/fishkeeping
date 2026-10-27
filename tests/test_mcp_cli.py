import sys
from types import SimpleNamespace

from fishkeeping import cli
from fishkeeping.config import Settings
from fishkeeping.models import Scope


def capture_server(monkeypatch):
    calls = {}

    def create(settings, scope, **kwargs):
        calls.update(settings=settings, scope=scope, **kwargs)
        return SimpleNamespace(run=lambda **options: calls.update(run=options))

    monkeypatch.setitem(
        sys.modules, "fishkeeping.mcp_server", SimpleNamespace(create_mcp_server=create)
    )
    monkeypatch.setattr(cli, "Settings", lambda: Settings(_env_file=None))
    return calls


def test_mcp_cli_defaults_to_memory_stdio(monkeypatch, capsys):
    calls = capture_server(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["fishkeeping", "mcp", "--user-id", "alice"])
    cli.main()
    assert calls["modules"] == ("memory",)
    assert calls["scope"] == Scope(user_id="alice")
    assert calls["run"] == {}
    assert capsys.readouterr().out == ""


def test_mcp_cli_selects_media_modules_and_http(monkeypatch, tmp_path):
    calls = capture_server(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fishkeeping",
            "mcp",
            "--user-id",
            "alice",
            "--namespace",
            "team",
            "--agent-id",
            "helper",
            "--modules",
            "image",
            "video",
            "--transport",
            "streamable-http",
            "--port",
            "8899",
            "--media-root",
            str(tmp_path),
        ],
    )
    cli.main()
    assert calls["modules"] == ("image", "video")
    assert calls["scope"] == Scope(namespace="team", user_id="alice", agent_id="helper")
    assert calls["media_root"] == tmp_path.resolve()
    assert calls["run"] == {
        "transport": "streamable-http",
        "host": "127.0.0.1",
        "port": 8899,
        "stateless_http": True,
        "json_response": True,
    }
