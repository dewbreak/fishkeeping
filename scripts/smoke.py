"""Exercise an actual HTTP server in an isolated directory, without provider calls."""

from __future__ import annotations

import socket
import tempfile
import threading
import time
from pathlib import Path

import httpx
import uvicorn

from fishkeeping.api import create_app
from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.models import Scope


def main() -> None:
    scratch = Path(".tmp").resolve()
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=scratch) as temporary:
        settings = Settings(
            _env_file=None,
            db_path=str(Path(temporary) / "smoke.sqlite3"),
            api_keys={},
            deepseek_api_key=None,
            embedding_base_url=None,
            embedding_model=None,
        )
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(create_app(settings), log_level="error", lifespan="on")
            )
            # Avoid Windows venv launcher child processes surviving terminate().
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            thread.start()
            try:
                base_url = f"http://127.0.0.1:{port}"
                with httpx.Client(base_url=base_url, timeout=1) as health:
                    deadline = time.monotonic() + 15
                    while True:
                        if not thread.is_alive():
                            raise RuntimeError("Memory server exited before becoming ready")
                        try:
                            if health.get("/readyz").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Memory server did not become ready")
                        time.sleep(0.1)
                with MemoryClient(base_url, Scope(user_id="smoke-user")) as client:
                    source = client.remember("用户喜欢Python开发与中文解释。", kind="episodic")
                    derived = client.consolidate([source.id])[0]
                    result = client.context("Python", max_chars=2000)
                    assert source.id in result.included_ids
                    assert result.chars_used == len(result.context) <= 2000
                    assert set(client.forget(source.id)) == {source.id, derived.id}
                    assert client.list() == []
                    print(
                        "HTTP smoke passed: readiness, SDK, mixed-language recall, cascade deletion"
                    )
            finally:
                server.should_exit = True
                thread.join(timeout=15)
                if thread.is_alive():
                    raise RuntimeError("Memory server did not stop")


if __name__ == "__main__":
    main()
