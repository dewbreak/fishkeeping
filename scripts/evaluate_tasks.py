"""Frozen synthetic task timelines; actual retrieval, deterministic quote-only answer proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fishkeeping.config import Settings
from fishkeeping.models import ContextRequest, IndexRequest, MemoryInput, Scope
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/task_memory_cases.json"
FROZEN_SHA256 = "d3bd120dbb122a92610675c091e720f42a6841aa83ebfe8387808c20ce3f1175"


def fixture_hash(dataset: dict) -> str:
    return hashlib.sha256(
        json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@contextmanager
def fixed_clock(anchor: datetime):
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return anchor.astimezone(tz) if tz is not None else anchor.replace(tzinfo=None)

    with ExitStack() as stack:
        for module in ("storage", "retrieval", "service"):
            stack.enter_context(patch(f"fishkeeping.{module}.datetime", FrozenDateTime))
        yield


def quote_answer(hits) -> dict:
    """No gold labels: quote every retrieved record passing a fixed confidence cutoff."""
    quotes = sorted({hit.memory.content for hit in hits if hit.memory.confidence >= 0.7})
    return {"status": "EVIDENCE" if quotes else "INSUFFICIENT_EVIDENCE", "quotes": quotes}


def calculate_metrics(cases: list[dict], k: int) -> dict:
    positives = [case for case in cases if case["expected"]]
    negatives = [case for case in cases if not case["expected"]]
    recalls, precisions, reciprocal_ranks = [], [], []
    false_positive_hits = returned_hits = 0
    for case in cases:
        returned = case["returned"][:k]
        relevant = set(case["expected"])
        hits = len(set(returned) & relevant)
        false_positive_hits += len(returned) - hits
        returned_hits += len(returned)
        # Empty retrieval has undefined precision. Report macro over nonempty results.
        if returned:
            precisions.append(hits / len(returned))
        if relevant:
            recalls.append(hits / len(relevant))
            rank = next((i for i, key in enumerate(returned, 1) if key in relevant), None)
            reciprocal_ranks.append(1 / rank if rank is not None else 0.0)
    return {
        "k": k,
        "positive_cases": len(positives),
        "recall_at_k": statistics.mean(recalls) if recalls else None,
        "mrr_at_k": statistics.mean(reciprocal_ranks) if reciprocal_ranks else None,
        "precision_at_k_nonempty_macro": statistics.mean(precisions) if precisions else None,
        "precision_nonempty_cases": len(precisions),
        "returned_hits": returned_hits,
        "false_positive_hits": false_positive_hits,
        "false_discovery_fraction": false_positive_hits / returned_hits if returned_hits else None,
        "queries_with_false_positive": sum(
            bool(set(case["returned"][:k]) - set(case["expected"])) for case in cases
        ),
        "strict_empty_cases": len(negatives),
        "strict_empty_passed": sum(not case["returned"][:k] for case in negatives),
        "safety_forbidden_hits": sum(len(case["safety_forbidden_hits"]) for case in cases),
        "known_incorrect_hits": sum(len(case["known_incorrect_hits"]) for case in cases),
        "quote_answer_exact_passed": sum(case["answer_exact"] for case in cases),
        "quote_answer_exact_accuracy": (
            statistics.mean(case["answer_exact"] for case in cases) if cases else None
        ),
        "llm_final_answer_accuracy": None,
    }


class CountedEmbeddings:
    def __init__(self, provider):
        self.provider = provider
        self.model_key = provider.model_key
        self.similarity_threshold = provider.similarity_threshold
        self.document_calls = self.document_texts = self.query_calls = 0

    def embed(self, texts):
        self.document_calls += 1
        self.document_texts += len(texts)
        return self.provider.embed(texts)

    def embed_query(self, text):
        self.query_calls += 1
        return self.provider.embed_query(text)


def validate_fixture(dataset: dict) -> None:
    scenario_ids, query_ids, keys = set(), set(), set()
    for scenario in dataset["scenarios"]:
        if scenario["id"] in scenario_ids:
            raise ValueError("duplicate scenario ID")
        scenario_ids.add(scenario["id"])
        known = set()
        for step in scenario["steps"]:
            if step["op"] == "put":
                if step["key"] in keys:
                    raise ValueError("duplicate record key")
                if not set(step.get("source_keys", [])).issubset(known):
                    raise ValueError("source must reference an earlier record in its scenario")
                known.add(step["key"])
                keys.add(step["key"])
            elif step["op"] == "delete":
                if step["key"] not in known:
                    raise ValueError("delete must reference an earlier record")
            elif step["op"] == "query":
                if step["id"] in query_ids:
                    raise ValueError("duplicate query ID")
                query_ids.add(step["id"])
                for field in ("expected", "answer_expected", "safety_forbidden", "known_incorrect"):
                    if not set(step.get(field, [])).issubset(known):
                        raise ValueError("query labels must reference known record keys")
                if set(step["expected"]) & set(step.get("safety_forbidden", [])):
                    raise ValueError("relevant and forbidden labels overlap")
            else:
                raise ValueError("unknown timeline operation")


def evaluate(dataset: dict, k: int = 3, *, embeddings=None) -> dict:
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    validate_fixture(dataset)
    anchor = datetime.fromisoformat(dataset["clock"])
    if anchor.tzinfo is None:
        raise ValueError("clock must include timezone")
    anchor = anchor.astimezone(UTC)
    settings = Settings(
        _env_file=None,
        db_path=":memory:",
        api_keys={},
        deepseek_api_key=None,
        embedding_api_key=None,
        embedding_base_url=None,
        embedding_model=None,
        candidate_limit=5000,
    )
    counted = CountedEmbeddings(embeddings) if embeddings is not None else None
    store = Store(":memory:")
    service = MemoryService(store, settings, llm=None, embeddings=counted)
    results, operations = [], []
    index_seconds = 0.0
    try:
        for scenario in dataset["scenarios"]:
            scope = Scope(namespace="task-eval", user_id=scenario["id"])
            records, keys_by_id = {}, {}
            offset = 0
            for step in scenario["steps"]:
                next_offset = step.get("at_seconds", offset + 1)
                if next_offset < offset:
                    raise ValueError("timeline must not move backwards")
                offset = next_offset
                with fixed_clock(anchor + timedelta(seconds=offset)):
                    if step["op"] == "put":
                        payload = {
                            key: value
                            for key, value in step.items()
                            if key not in {"op", "key", "source_keys", "at_seconds"}
                        }
                        payload.setdefault("scope", scope.model_dump())
                        payload["source_ids"] = [
                            records[key].id for key in step.get("source_keys", [])
                        ]
                        memory = store.put(MemoryInput.model_validate(payload))
                        if memory.id in keys_by_id:
                            raise ValueError("fixture records must not deduplicate")
                        records[step["key"]] = memory
                        keys_by_id[memory.id] = step["key"]
                    elif step["op"] == "delete":
                        deleted = store.delete(scope, records[step["key"]].id)
                        operations.append(
                            {
                                "delete": step["key"],
                                "deleted": sorted(keys_by_id[i] for i in deleted),
                            }
                        )
                    else:
                        if counted is not None:
                            index_start = time.perf_counter()
                            while True:
                                progress = service.index(IndexRequest(scope=scope, limit=1024))
                                if progress.complete:
                                    break
                                if progress.processed == 0:
                                    raise RuntimeError("indexing made no progress")
                            index_seconds += time.perf_counter() - index_start
                        start = time.perf_counter()
                        response = service.context(
                            ContextRequest(
                                scope=scope,
                                query=step["query"],
                                session_id=step.get("session_id"),
                                kinds=step.get("kinds"),
                                limit=k,
                                max_chars=6000,
                            )
                        )
                        elapsed = time.perf_counter() - start
                        returned = [keys_by_id[hit.memory.id] for hit in response.hits]
                        expected = step["expected"]
                        answer = quote_answer(response.hits)
                        expected_quotes = sorted(
                            {records[key].content for key in step.get("answer_expected", expected)}
                        )
                        results.append(
                            {
                                "id": step["id"],
                                "scenario": scenario["id"],
                                "category": step["category"],
                                "query": step["query"],
                                "expected": expected,
                                "returned": returned,
                                "missing": sorted(set(expected) - set(returned)),
                                "false_positive_hits": sorted(set(returned) - set(expected)),
                                "safety_forbidden_hits": sorted(
                                    set(returned) & set(step.get("safety_forbidden", []))
                                ),
                                "known_incorrect_hits": sorted(
                                    set(returned) & set(step.get("known_incorrect", []))
                                ),
                                "answer": answer,
                                "answer_expected_quotes": expected_quotes,
                                "answer_exact": answer["quotes"] == expected_quotes,
                                "retrieval_mode": response.retrieval_mode,
                                "warnings": response.warnings,
                                "context_characters": response.chars_used,
                                "context_latency_ms": round(elapsed * 1000, 3),
                            }
                        )
    finally:
        store.close()
    latencies = [case["context_latency_ms"] for case in results]
    return {
        "fixture_version": dataset["version"],
        "fixture_sha256_canonical": fixture_hash(dataset),
        "clock": anchor.isoformat(),
        "scenario_count": len(dataset["scenarios"]),
        "record_count": sum(s["op"] == "put" for x in dataset["scenarios"] for s in x["steps"]),
        "case_count": len(results),
        "mode": "local-hybrid" if counted else "offline-lexical",
        "embedding_model_key": counted.model_key if counted else None,
        "semantic_threshold": counted.similarity_threshold if counted else None,
        "metrics": calculate_metrics(results, k),
        "cost": {
            "paid_api_requests": 0,
            "api_input_tokens": 0,
            "api_output_tokens": 0,
            "api_cost_usd": 0.0,
            "local_document_encoding_calls": counted.document_calls if counted else 0,
            "local_document_texts_encoded": counted.document_texts if counted else 0,
            "local_query_encoding_calls": counted.query_calls if counted else 0,
            "llm_final_answers_evaluated": 0,
            "token_estimate": None,
        },
        "timing": {
            "context_median_ms": statistics.median(latencies) if latencies else None,
            "context_max_ms": max(latencies) if latencies else None,
            "explicit_index_seconds_including_lazy_model_load": round(index_seconds, 3),
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "operations": operations,
        "cases": results,
        "limitations": (
            "Synthetic frozen timelines; quote-only evidence proxy is not LLM answer accuracy. "
            "Single local measurement, no network or LLM latency; no production SLA. "
            "Known-wrong extraction is deliberately retained as a factual verification limitation."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local-embeddings", action="store_true", help="Cache-only local model")
    args = parser.parse_args()
    dataset = json.loads(args.fixture.read_text(encoding="utf-8"))
    if (
        args.fixture.resolve() == DEFAULT_FIXTURE.resolve()
        and fixture_hash(dataset) != FROZEN_SHA256
    ):
        raise SystemExit("Frozen task labels changed; preserve original dataset and version")
    provider = None
    try:
        if args.local_embeddings:
            os.environ["HF_HUB_OFFLINE"] = "1"
            from fishkeeping.local_embeddings import LocalEmbeddingProvider

            provider = LocalEmbeddingProvider(
                Settings(_env_file=None, local_embedding_min_similarity=0.50)
            )
        report = evaluate(dataset, args.k, embeddings=provider)
    finally:
        if provider is not None:
            provider.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"mode": report["mode"], **report["metrics"], **report["timing"]}))
    if report["metrics"]["safety_forbidden_hits"]:
        raise SystemExit("Scope/lifecycle boundary violation")


if __name__ == "__main__":
    main()
