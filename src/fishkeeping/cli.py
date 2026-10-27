"""Command-line entry point, safe local binding by default."""

import argparse
import json
import math
import time

import uvicorn

from fishkeeping.config import Settings
from fishkeeping.embedding_factory import create_embeddings
from fishkeeping.index_worker import IndexWorker
from fishkeeping.jobs import EventQueue
from fishkeeping.models import IndexRequest, Scope
from fishkeeping.providers import DeepSeekProvider
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def run_worker(settings: Settings, *, once: bool, max_jobs: int, poll_interval: float) -> None:
    if not settings.deepseek_api_key:
        raise SystemExit("Set FISHKEEPING_DEEPSEEK_API_KEY before starting an extraction worker")
    store = Store(settings.db_path)
    provider = DeepSeekProvider(settings)
    try:
        queue = EventQueue(store, settings)
        processed = 0
        while processed < max_jobs:
            job = queue.process_one(provider)
            if job is not None:
                processed += 1
                # Do not print event payloads, memory contents, secrets or upstream errors.
                print(
                    json.dumps(
                        {
                            "job_id": job["id"],
                            "state": job["state"],
                            "attempts": job["attempts"],
                            "code": job["last_error_code"],
                        }
                    )
                )
            if once or queue.budget_status()["remaining"] == 0:
                break
            if job is None:
                time.sleep(max(0.1, min(60.0, poll_interval)))
        print(json.dumps({"processed": processed, "budget": queue.budget_status()}))
    except KeyboardInterrupt:
        pass
    finally:
        provider.close()
        store.close()


def run_index(
    settings: Settings, scope: Scope, *, batch_size: int, max_batches: int, rebuild: bool = False
) -> None:
    embeddings = create_embeddings(settings)
    if embeddings is None:
        raise SystemExit("Configure a local or remote embedding provider before indexing")
    store = Store(settings.db_path)
    try:
        service = MemoryService(store, settings, embeddings=embeddings)
        for batch_number in range(max_batches):
            result = service.index(
                IndexRequest(
                    scope=scope, limit=min(batch_size, 1024), rebuild=rebuild and batch_number == 0
                )
            )
            print(result.model_dump_json())
            if result.complete or result.processed == 0:
                break
    finally:
        embeddings.close()
        store.close()


def run_index_worker(
    settings: Settings,
    scope: Scope,
    *,
    once: bool,
    batch_size: int,
    max_batches: int,
    daily_limit: int,
    poll_interval: float,
) -> None:
    if settings.embedding_backend != "local":
        raise SystemExit("index-worker requires FISHKEEPING_EMBEDDING_BACKEND=local")
    if not math.isfinite(poll_interval) or poll_interval <= 0:
        raise SystemExit("poll-interval must be finite and positive")
    embeddings = create_embeddings(settings)
    store = Store(settings.db_path)
    try:
        worker = IndexWorker(
            store, embeddings, scope, batch_size=batch_size, daily_limit=daily_limit
        )
        batches = processed = 0
        delay = min(60.0, max(0.1, poll_interval))
        previous_state = None
        while batches < max_batches:
            result = worker.process_one()
            if result["reserved"]:
                batches += 1
                processed += result["processed"]
            if result["reserved"] or result["state"] != previous_state:
                print(json.dumps(result))
            previous_state = result["state"]
            if once or result["budget"]["remaining"] == 0 or batches >= max_batches:
                break
            if result["state"] != "completed":
                time.sleep(delay)
                delay = min(60.0, delay * 2)
            else:
                delay = min(60.0, max(0.1, poll_interval))
        print(
            json.dumps(
                {"batches": batches, "processed": processed, "budget": worker.budget_status()}
            )
        )
    except KeyboardInterrupt:
        pass
    finally:
        embeddings.close()
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="fishkeeping")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Start the memory HTTP API")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    worker = commands.add_parser("worker", help="Process durable extraction jobs with a daily cap")
    worker.add_argument(
        "--once", action="store_true", help="Process at most one ready job and exit"
    )
    worker.add_argument("--max-jobs", type=positive_int, default=20)
    worker.add_argument("--poll-interval", type=float, default=2)
    index = commands.add_parser("index", help="Explicitly build a bounded embedding index")
    index.add_argument("--user-id", required=True)
    index.add_argument("--namespace", default="default")
    index.add_argument("--agent-id", default="default")
    index.add_argument("--batch-size", type=positive_int, default=64)
    index.add_argument("--max-batches", type=positive_int, default=20)
    index.add_argument("--rebuild", action="store_true", help="Clear this scope/model cache first")
    index_worker = commands.add_parser(
        "index-worker", help="Continuously index new local memories with persistent record budgets"
    )
    index_worker.add_argument("--user-id", required=True)
    index_worker.add_argument("--namespace", default="default")
    index_worker.add_argument("--agent-id", default="default")
    index_worker.add_argument(
        "--once", action="store_true", help="Process at most one batch and exit"
    )
    index_worker.add_argument("--batch-size", type=positive_int, choices=range(1, 33), default=32)
    index_worker.add_argument("--max-batches", type=positive_int, default=20)
    index_worker.add_argument("--daily-limit", type=positive_int, default=1000)
    index_worker.add_argument("--poll-interval", type=float, default=2)
    args = parser.parse_args()
    settings = Settings()
    if args.command == "worker":
        run_worker(
            settings, once=args.once, max_jobs=args.max_jobs, poll_interval=args.poll_interval
        )
        return
    if args.command == "index":
        run_index(
            settings,
            Scope(namespace=args.namespace, user_id=args.user_id, agent_id=args.agent_id),
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            rebuild=args.rebuild,
        )
        return
    if args.command == "index-worker":
        run_index_worker(
            settings,
            Scope(namespace=args.namespace, user_id=args.user_id, agent_id=args.agent_id),
            once=args.once,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
            daily_limit=args.daily_limit,
            poll_interval=args.poll_interval,
        )
        return
    uvicorn.run(
        "fishkeeping.api:create_app",
        factory=True,
        host=args.host or settings.host,
        port=args.port or settings.port,
    )


if __name__ == "__main__":
    main()
