"""Smoke real media through an isolated loopback HTTP server, with no paid model calls.

Uses only hash-verified public fixtures already cached by media_assets.py. It never
downloads assets or OCR weights, loads credentials, calls a model, or touches the
normal service database. Local OCR and OpenCV are required. Temporary decode files
stay in workspace .tmp so restricted Windows system TEMP permissions are avoided.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import socket
import tempfile
import threading
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import httpx
import uvicorn
from media_assets import DEFAULT_CACHE, load_manifest, prepare_case

from fishkeeping.api import create_app
from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.conversations import trusted_conversation_content
from fishkeeping.models import Scope
from fishkeeping.video import OpenCVDecoder, VideoProcessor

ROOT = Path(__file__).resolve().parents[1]


def _expect_status(status: int, action) -> None:
    try:
        action()
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != status:
            raise AssertionError(f"Expected HTTP {status}, got {exc.response.status_code}") from exc
    else:
        raise AssertionError(f"Expected HTTP {status}")


def _source_lifecycle(client, foreign, event, replay, conflict):
    assert event.job is None and not event.replayed
    assert trusted_conversation_content(event.episode) is None
    assert client.event(event.event_id).episode.id == event.episode.id
    repeated = replay()
    if hasattr(repeated, "event"):
        repeated = repeated.event
    assert repeated.replayed and repeated.episode.id == event.episode.id
    _expect_status(409, conflict)
    _expect_status(404, lambda: foreign.event(event.event_id))
    _expect_status(404, lambda: foreign.get(event.episode.id))
    marker = client.remember(
        "Smoke lifecycle marker for this public media source.",
        kind="semantic",
        source_ids=[event.episode.id],
    )
    assert client.sources(marker.id).items == [event.episode.id]
    assert set(client.forget(event.episode.id)) == {event.episode.id, marker.id}
    _expect_status(409, replay)
    assert client.list() == []


def run(cache_dir: Path = DEFAULT_CACHE) -> dict:
    """Exercise public photo/table/video bytes with real codecs and actual HTTP transport."""
    started = time.perf_counter()
    manifest = load_manifest()
    ids = ("real_coffee_spatial", "real_usda_rice_two_tables", "real_bbb_alternating_shots")
    cases = [
        next(case for case in manifest["cases"] if case["id"] == identifier) for identifier in ids
    ]
    prepared = [
        (case, prepare_case(case, cache_dir, download=False, manifest=manifest).read_bytes())
        for case in cases
    ]
    report = {"status": "running", "paid_calls": 0, "downloads": 0, "cases": []}
    scratch = ROOT / ".tmp"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="media-http-smoke-", dir=scratch) as temporary:
        key = "local-media-smoke-" + uuid4().hex
        scope = Scope(namespace="media-smoke", user_id="public-fixtures")
        settings = Settings(
            _env_file=None,
            db_path=str(Path(temporary) / "smoke.sqlite3"),
            api_keys={key: scope.namespace},
            deepseek_api_key=None,
            embedding_base_url=None,
            embedding_model=None,
            embedding_backend="remote",
        )
        app = create_app(settings)
        app.state.videos = VideoProcessor(OpenCVDecoder(temporary_root=temporary))
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            base_url = f"http://127.0.0.1:{listener.getsockname()[1]}"
            server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="on"))
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            thread.start()
            try:
                with httpx.Client(base_url=base_url, timeout=1, trust_env=False) as health:
                    deadline = time.monotonic() + 15
                    while True:
                        if not thread.is_alive():
                            raise RuntimeError("Media smoke server exited before readiness")
                        try:
                            if health.get("/readyz").status_code == 200:
                                break
                        except httpx.HTTPError:
                            pass
                        if time.monotonic() >= deadline:
                            raise RuntimeError("Media smoke server did not become ready")
                        time.sleep(0.1)
                # Explicit local transports avoid ambient HTTP proxy configuration.
                with (
                    MemoryClient(
                        base_url,
                        scope,
                        api_key=key,
                        transport=httpx.HTTPTransport(trust_env=False),
                    ) as client,
                    MemoryClient(
                        base_url,
                        scope.model_copy(update={"user_id": "another-user"}),
                        api_key=key,
                        transport=httpx.HTTPTransport(trust_env=False),
                    ) as foreign,
                ):
                    for case, raw in prepared:
                        item_started = time.perf_counter()
                        digest = hashlib.sha256(raw).hexdigest()
                        event_id, session = case["id"], "media-http-smoke"
                        if case["kind"] == "video":
                            options = {
                                "max_frames": 6,
                                "strategy": "adaptive",
                                "max_chars": 6000,
                                "start_seconds": case["start_seconds"],
                                "end_seconds": case["end_seconds"],
                            }
                            evidence = client.analyze_video(raw, case["question"], **options)
                            assert evidence.video_sha256 == digest
                            assert evidence.chars_used == len(evidence.packed) <= 6000
                            assert 2 <= len(evidence.selected_frames) <= 6
                            for frame in evidence.selected_frames:
                                assert (
                                    options["start_seconds"]
                                    <= frame.timestamp_seconds
                                    <= options["end_seconds"]
                                )
                                assert (
                                    hashlib.sha256(base64.b64decode(frame.image_base64)).hexdigest()
                                    == frame.image_sha256
                                )
                            receipt = client.enqueue_video(
                                raw, case["question"], event_id, session, **options
                            )
                            envelope = json.loads(receipt.episode.content[7:])
                            assert envelope["evidence"] == json.loads(evidence.packed)
                            assert "image_base64" not in receipt.episode.content
                            _source_lifecycle(
                                client,
                                foreign,
                                receipt,
                                partial(
                                    client.enqueue_video,
                                    raw,
                                    case["question"],
                                    event_id,
                                    session,
                                    **options,
                                ),
                                partial(
                                    client.enqueue_video,
                                    raw,
                                    "Changed question",
                                    event_id,
                                    session,
                                    **options,
                                ),
                            )
                            selected = len(evidence.selected_frames)
                        else:
                            options = (
                                {"task": "scene"}
                                if case["kind"] == "photo"
                                else {"layout": "table", "max_chars": 10000}
                            )
                            evidence = client.analyze_image(raw, case["question"], **options)
                            assert evidence.image_sha256 == digest
                            assert (
                                evidence.chars_used
                                == len(evidence.packed)
                                <= options.get("max_chars", 3000)
                            )
                            if case["kind"] == "photo":
                                assert evidence.requires_review and not evidence.regions
                            else:
                                assert evidence.regions and evidence.requires_review
                                assert evidence.selection["selection_mode"] == "full_table_fallback"
                            receipt = client.enqueue_image(
                                raw, case["question"], event_id, session, **options
                            )
                            assert receipt.evidence.packed == evidence.packed
                            _source_lifecycle(
                                client,
                                foreign,
                                receipt.event,
                                partial(
                                    client.enqueue_image,
                                    raw,
                                    case["question"],
                                    event_id,
                                    session,
                                    **options,
                                ),
                                partial(
                                    client.enqueue_image,
                                    raw,
                                    "Changed question",
                                    event_id,
                                    session,
                                    **options,
                                ),
                            )
                            selected = len(evidence.regions)
                        report["cases"].append(
                            {
                                "id": case["id"],
                                "kind": case["kind"],
                                "input_sha256": digest,
                                "input_bytes": len(raw),
                                "selected_evidence_count": selected,
                                "requires_review": evidence.requires_review,
                                "audit_chars": evidence.chars_used,
                                "lifecycle": (
                                    "analyze,event,replay,conflict,scope,source,forget,tombstone"
                                ),
                                "elapsed_seconds": round(time.perf_counter() - item_started, 3),
                            }
                        )
            finally:
                server.should_exit = True
                thread.join(timeout=15)
                if thread.is_alive():
                    raise RuntimeError("Media smoke server did not stop")
    report.update(status="passed", elapsed_seconds=round(time.perf_counter() - started, 3))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=ROOT / ".tmp" / "media-http-smoke.json")
    args = parser.parse_args()
    try:
        report = run(args.cache_dir)
    except Exception as exc:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"status": "failed", "paid_calls": 0, "error_type": type(exc).__name__})
            + "\n",
            encoding="utf-8",
        )
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
