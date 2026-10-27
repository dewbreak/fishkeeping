"""Explicitly opt-in, one-call real DeepSeek acceptance for the durable worker."""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx
from live_deepseek import MeteredTransport

from fishkeeping.config import Settings
from fishkeeping.jobs import EventQueue
from fishkeeping.models import EventRequest, RecallRequest, Scope
from fishkeeping.providers import DeepSeekProvider
from fishkeeping.service import MemoryService
from fishkeeping.storage import ConflictError, Store


def run(report_path: Path) -> dict:
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "max_calls": 1,
        "max_output_tokens": 1024,
        "synthetic_data_only": True,
        "checks": {},
    }
    meter = MeteredTransport(max_calls=1)

    def check(name, condition):
        report["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    try:
        settings = Settings(
            deepseek_model="deepseek-flash",
            deepseek_thinking="disabled",
            deepseek_max_tokens=1024,
            worker_daily_call_limit=1,
        )
        if (
            settings.deepseek_base_url != "https://api.deepseek.com"
            or not settings.deepseek_api_key
        ):
            raise ValueError("Configure the official DeepSeek endpoint and key")
        scope = Scope(namespace="live-durable", user_id="synthetic-user")
        request = EventRequest(
            scope=scope,
            event_id="durable-one",
            session_id="synthetic",
            messages=[
                {"role": "user", "content": "我偏好Python示例，回答使用中文。"},
                {"role": "assistant", "content": "我猜你住在火星。"},
            ],
        )
        scratch = Path(".tmp").resolve()
        scratch.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="events-live-", dir=scratch) as temporary:
            path = str(Path(temporary) / "events.sqlite3")
            store = Store(path)
            try:
                receipt = EventQueue(store, settings).enqueue(request)
                check(
                    "receipt_before_model", receipt["job"]["state"] == "pending" and not meter.calls
                )
            finally:
                store.close()
            store = Store(path)
            try:
                queue = EventQueue(store, settings)
                check(
                    "pending_survives_reopen",
                    queue.get(scope, request.event_id)["job"]["state"] == "pending",
                )
                with httpx.Client(transport=meter, timeout=settings.request_timeout) as upstream:
                    provider = DeepSeekProvider(settings, upstream)
                    meter.phase = "durable_extraction"
                    job = queue.process_one(provider)
                    check("worker_succeeded", job["state"] == "succeeded")
                    check("exactly_one_claim", job["attempts"] == len(meter.calls) == 1)
                    check("daily_budget_enforced", queue.budget_status()["remaining"] == 0)
                    facts = [store.get(scope, identifier) for identifier in job["result_ids"]]
                    check("real_facts_nonempty", bool(facts))
                    check(
                        "assistant_claim_excluded",
                        all("火星" not in fact.content for fact in facts),
                    )
                    check(
                        "facts_have_source",
                        all(receipt["episode"].id in fact.source_ids for fact in facts),
                    )
                    check("replay_no_second_job", queue.enqueue(request)["replayed"])
                    check("completed_job_not_reclaimed", queue.process_one(provider) is None)
                    recalled = MemoryService(store, settings).recall(
                        RecallRequest(scope=scope, query="Python", kinds=["semantic", "procedural"])
                    )
                    check(
                        "facts_recalled",
                        any("Python" in hit.memory.content for hit in recalled.hits),
                    )
                    check("single_model_call", len(meter.calls) == 1)
                    store.delete(scope, receipt["episode"].id)
                    check("forget_cascades", store.stats(scope)["total"] == 0)
                    try:
                        queue.enqueue(request)
                    except ConflictError:
                        check("tombstone_prevents_resurrection", True)
                    else:
                        check("tombstone_prevents_resurrection", False)
            finally:
                store.close()
        report["status"] = "passed"
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "code": getattr(exc, "code", None)}
    finally:
        meter.close()
        report["calls"] = meter.calls
        report["completed_at"] = datetime.now(UTC).isoformat()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize at most one paid call")
    parser.add_argument("--report", type=Path, default=Path(".tmp/events-live.json"))
    args = parser.parse_args()
    if not args.live:
        parser.error("Pass --live explicitly to authorize the one-call acceptance")
    report = run(args.report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "calls": len(report["calls"]),
                "checks_passed": sum(report["checks"].values()),
                "report": str(args.report),
            }
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
