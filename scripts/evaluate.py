"""Offline, fixed-clock synthetic retrieval diagnostics; never calls model providers."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fishkeeping.config import Settings
from fishkeeping.models import IndexRequest, MemoryInput, RecallRequest, Scope
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "recall_cases.json"


@contextmanager
def fixed_clock(anchor: datetime):
    """Keep expiry and recency deterministic without replacing storage/retrieval logic."""

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return anchor.astimezone(tz) if tz is not None else anchor.replace(tzinfo=None)

    with ExitStack() as stack:
        for module in ("storage", "retrieval", "service"):
            stack.enter_context(patch(f"fishkeeping.{module}.datetime", FrozenDateTime))
        yield


def calculate_metrics(results: list[dict], limit: int) -> dict:
    """Macro-average recall and reciprocal rank over positive-label queries only."""
    positives = [case for case in results if case["expected"]]
    negatives = [case for case in results if not case["expected"]]
    recall_values = []
    reciprocal_ranks = []
    for case in positives:
        expected = set(case["expected"])
        returned = case["returned"][:limit]
        recall_values.append(len(expected.intersection(returned)) / len(expected))
        first = next((rank for rank, key in enumerate(returned, 1) if key in expected), None)
        reciprocal_ranks.append(1 / first if first is not None else 0.0)
    return {
        "k": limit,
        "positive_cases": len(positives),
        "recall_at_k": sum(recall_values) / len(positives) if positives else None,
        "mrr_at_k": sum(reciprocal_ranks) / len(positives) if positives else None,
        "strict_empty_cases": len(negatives),
        "strict_empty_passed": sum(not case["returned"] for case in negatives),
        "forbidden_hits": sum(len(case.get("forbidden_hits", [])) for case in results),
    }


def evaluate(dataset: dict, limit: int = 3, *, embeddings=None) -> dict:
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    anchor = datetime.fromisoformat(dataset["clock"])
    if anchor.tzinfo is None:
        raise ValueError("fixture clock must include timezone")
    anchor = anchor.astimezone(UTC)
    scope = Scope.model_validate(dataset["scope"])
    record_keys = [record["key"] for record in dataset["records"]]
    if len(record_keys) != len(set(record_keys)):
        raise ValueError("fixture record keys must be unique")
    case_ids = [case["id"] for case in dataset["cases"]]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("fixture case IDs must be unique")
    known = set(record_keys)
    for case in dataset["cases"]:
        if not set(case["expected"] + case.get("forbidden", [])).issubset(known):
            raise ValueError("case labels must reference known record keys")

    # Do not read .env. Explicitly override provider settings inherited from the shell.
    settings = Settings(
        _env_file=None,
        db_path=":memory:",
        api_keys={},
        deepseek_api_key=None,
        embedding_api_key=None,
        embedding_base_url=None,
        embedding_model=None,
        candidate_limit=max(100, len(record_keys) + 1),
    )
    store = Store(":memory:")
    results = []
    try:
        service = MemoryService(store, settings, llm=None, embeddings=embeddings)
        key_by_id = {}
        with fixed_clock(anchor):
            for position, record in enumerate(dataset["records"]):
                payload = {key: value for key, value in record.items() if key != "key"}
                payload.setdefault("scope", scope.model_dump())
                # All records are at least one hour old; TTL=1 is deliberately expired.
                created_at = anchor - timedelta(hours=1, seconds=len(record_keys) - position)
                memory = store.put(MemoryInput.model_validate(payload), now=created_at)
                if memory.id in key_by_id:
                    raise ValueError("fixture records must not collapse into duplicate memories")
                key_by_id[memory.id] = record["key"]
            if embeddings is not None:
                # Only the tested scope is indexed; foreign-scope fixtures remain isolated.
                while not service.index(IndexRequest(scope=scope, limit=1024)).complete:
                    pass
            for case in dataset["cases"]:
                response = service.recall(
                    RecallRequest(
                        scope=Scope.model_validate(case.get("scope", scope.model_dump())),
                        query=case["query"],
                        session_id=case.get("session_id"),
                        limit=limit,
                    )
                )
                returned = [key_by_id[hit.memory.id] for hit in response.hits]
                expected = list(dict.fromkeys(case["expected"]))
                forbidden_hits = sorted(set(returned).intersection(case.get("forbidden", [])))
                results.append(
                    {
                        "id": case["id"],
                        "category": case.get("category", "unspecified"),
                        "query": case["query"],
                        "expected": expected,
                        "returned": returned,
                        "missing": [key for key in expected if key not in returned],
                        "forbidden_hits": forbidden_hits,
                        "retrieval_mode": response.retrieval_mode,
                        "warnings": response.warnings,
                    }
                )
    finally:
        store.close()
    metrics = calculate_metrics(results, limit)
    return {
        "fixture_version": dataset["version"],
        "clock": anchor.isoformat(),
        "record_count": len(record_keys),
        "case_count": len(results),
        "mode": "local-hybrid" if embeddings is not None else "offline-lexical",
        "model_calls": 0,  # Paid/API model calls; local encoding is opt-in below.
        "embedding_model_key": embeddings.model_key if embeddings is not None else None,
        "semantic_threshold": getattr(embeddings, "similarity_threshold", None),
        "metrics": metrics,
        "strict_checks_passed": metrics["forbidden_hits"] == 0
        and metrics["strict_empty_passed"] == metrics["strict_empty_cases"],
        "cases": results,
        "limitations": "Small hand-labeled synthetic diagnostic set, not a general benchmark.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        help="Download/load the pinned local model and evaluate semantic recall (no paid API)",
    )
    args = parser.parse_args()
    dataset = json.loads(args.fixture.read_text(encoding="utf-8"))
    embeddings = None
    start = time.monotonic()
    try:
        if args.local_embeddings:
            from fishkeeping.local_embeddings import LocalEmbeddingProvider

            embeddings = LocalEmbeddingProvider(Settings(_env_file=None))
        report = evaluate(dataset, args.k, embeddings=embeddings)
    finally:
        if embeddings is not None:
            embeddings.close()
    if args.local_embeddings:
        report["elapsed_seconds"] = round(time.monotonic() - start, 3)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {"records": report["record_count"], "cases": report["case_count"], **report["metrics"]}
        )
    )
    if not report["strict_checks_passed"]:
        raise SystemExit("Strict isolation/expiry/empty-result checks failed")


if __name__ == "__main__":
    main()
