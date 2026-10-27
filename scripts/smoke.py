"""Exercise an actual HTTP server in an isolated directory, without provider calls."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Opt in to local model loading/CPU inference (no paid API)",
    )
    args = parser.parse_args()
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
            embedding_backend="local" if args.local_embeddings else "remote",
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
                    event = client.enqueue(
                        "smoke-event",
                        "smoke-session",
                        [{"role": "user", "content": "我每天骑自行车上班。"}],
                    )
                    assert event.job.state == "pending"
                    assert client.event("smoke-event").episode.id == event.episode.id
                    assert client.enqueue(
                        "smoke-event",
                        "smoke-session",
                        [{"role": "user", "content": "我每天骑自行车上班。"}],
                    ).replayed
                    if args.local_embeddings:
                        assert client.index().complete
                        semantic = client.recall("The user rides a bike to work every day.")
                        assert semantic.retrieval_mode == "hybrid"
                        assert event.episode.id in {hit.memory.id for hit in semantic.hits}
                    client.forget(event.episode.id)
                    source = client.remember("用户喜欢Python开发与中文解释。", kind="episodic")
                    derived = client.consolidate([source.id])[0]
                    assert client.sources(derived.id).items == [source.id]
                    result = client.context("Python", max_chars=2000)
                    assert source.id in result.included_ids
                    assert result.chars_used == len(result.context) <= 2000
                    assert set(client.forget(source.id)) == {source.id, derived.id}
                    assert client.list() == []
                    print(
                        "HTTP smoke passed: durable events, sources, SDK, recall, deletion; "
                        f"local_embeddings={args.local_embeddings}"
                    )
            finally:
                server.should_exit = True
                thread.join(timeout=15)
                if thread.is_alive():
                    raise RuntimeError("Memory server did not stop")


if __name__ == "__main__":
    main()
