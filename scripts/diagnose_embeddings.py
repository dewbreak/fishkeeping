"""Cache-only multilingual cosine calibration on frozen synthetic examples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from evaluate import DEFAULT_FIXTURE, calculate_metrics, fixed_clock

from fishkeeping.config import Settings
from fishkeeping.local_embeddings import LocalEmbeddingProvider
from fishkeeping.models import MemoryInput, Scope
from fishkeeping.retrieval import query_terms, tokenize
from fishkeeping.storage import Store

THRESHOLDS = [0.35, 0.45, 0.50, 0.55, 0.60]

# Initially authored independently of the original 22 cases. Now used for
# calibration and stop-word diagnostics, this is a frozen regression set.
DEFAULT_HOLDOUT = DEFAULT_FIXTURE.with_name("multilingual_holdout.json")


def cosine(left, right) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True)) / (
        math.hypot(*left) * math.hypot(*right)
    )


def score_range(values) -> dict:
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def diagnose(dataset: dict, provider: LocalEmbeddingProvider) -> dict:
    scope = Scope.model_validate(dataset["scope"])
    anchor = datetime.fromisoformat(dataset["clock"]).astimezone(UTC)
    records = dataset["records"]
    document_vectors = provider.embed([record["content"] for record in records])
    vectors = dict(zip((record["key"] for record in records), document_vectors, strict=True))
    store = Store(":memory:")
    results = []
    try:
        with fixed_clock(anchor):
            keys = {}
            for i, record in enumerate(records):
                data = {key: value for key, value in record.items() if key != "key"}
                data.setdefault("scope", scope.model_dump())
                memory = store.put(
                    MemoryInput.model_validate(data),
                    now=anchor - timedelta(hours=1, seconds=len(records) - i),
                )
                keys[memory.id] = record["key"]
            for case in dataset["cases"]:
                session = case.get("session_id")
                eligible = store.list(
                    Scope.model_validate(case.get("scope", scope.model_dump())),
                    session_id=session,
                    kinds=None if session else ["episodic", "semantic", "procedural"],
                    limit=10_000,
                )
                eligible_keys = {keys[memory.id] for memory in eligible}
                query_vector = provider.embed_query(case["query"])
                query_tokens = set(query_terms(case["query"]))
                ranked = sorted(
                    [
                        {
                            "key": record["key"],
                            "cosine": cosine(query_vector, vectors[record["key"]]),
                            "eligible": record["key"] in eligible_keys,
                            "expected": record["key"] in case["expected"],
                            "lexical_overlap": bool(
                                query_tokens & set(tokenize(record["content"]))
                            ),
                        }
                        for record in records
                    ],
                    key=lambda item: (-item["cosine"], item["key"]),
                )
                expected = [item for item in ranked if item["expected"] and item["eligible"]]
                unexpected = [item for item in ranked if not item["expected"] and item["eligible"]]
                excluded = [item for item in ranked if not item["eligible"]]
                results.append(
                    {
                        **case,
                        "best_expected": expected[0] if expected else None,
                        "worst_expected": expected[-1] if expected else None,
                        "best_unexpected": unexpected[0] if unexpected else None,
                        "best_excluded": excluded[0] if excluded else None,
                        "all_pairs": ranked,
                    }
                )
    finally:
        store.close()

    thresholds = []
    for threshold in THRESHOLDS:
        semantic_cases = []
        gate_expected_pairs, expected_pairs = 0, 0
        lexical_empty_failures = []
        hybrid_gate_empty_passed = 0
        for case in results:
            eligible = [pair for pair in case["all_pairs"] if pair["eligible"]]
            semantic = [pair["key"] for pair in eligible if pair["cosine"] > threshold]
            gated = {
                pair["key"]
                for pair in eligible
                if pair["cosine"] > threshold or pair["lexical_overlap"]
            }
            semantic_cases.append({"expected": case["expected"], "returned": semantic[:3]})
            expected_pairs += len(case["expected"])
            gate_expected_pairs += len(set(case["expected"]) & gated)
            if not case["expected"]:
                hybrid_gate_empty_passed += not gated
                lexical_hits = [pair["key"] for pair in eligible if pair["lexical_overlap"]]
                if lexical_hits:
                    lexical_empty_failures.append({"id": case["id"], "hits": lexical_hits})
        thresholds.append(
            {
                "threshold": threshold,
                "semantic_only": calculate_metrics(semantic_cases, 3),
                "expected_pairs_surviving_semantic_or_lexical_gate": gate_expected_pairs,
                "expected_pairs": expected_pairs,
                "hybrid_gate_empty_passed": hybrid_gate_empty_passed,
                "negative_lexical_overlap_not_fixable_by_cosine_threshold": lexical_empty_failures,
            }
        )
    positives = [case for case in results if case["expected"]]
    negatives = [case for case in results if not case["expected"]]
    return {
        "record_count": len(records),
        "case_count": len(results),
        "ranges": {
            "positive_best_expected": score_range(
                [case["best_expected"]["cosine"] for case in positives if case["best_expected"]]
            ),
            "positive_worst_expected": score_range(
                [case["worst_expected"]["cosine"] for case in positives if case["worst_expected"]]
            ),
            "positive_best_unexpected": score_range(
                [case["best_unexpected"]["cosine"] for case in positives if case["best_unexpected"]]
            ),
            "negative_best_eligible": score_range(
                [case["best_unexpected"]["cosine"] for case in negatives if case["best_unexpected"]]
            ),
        },
        "threshold_comparison": thresholds,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--cache", type=Path, default=Path(".cache/embeddings"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Fail if the exact snapshot is absent. This diagnostic never downloads models.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    source = args.fixture.read_bytes()
    dataset = json.loads(source)
    holdout_source = args.holdout.read_bytes()
    independent = json.loads(holdout_source)
    settings = Settings(
        _env_file=None,
        local_embedding_cache=str(args.cache.resolve()),
        deepseek_api_key=None,
        embedding_api_key=None,
        embedding_base_url=None,
    )
    provider = LocalEmbeddingProvider(settings)
    started = time.monotonic()
    try:
        original = diagnose(dataset, provider)
        holdout_result = diagnose(independent, provider)
    finally:
        provider.close()
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model_key": provider.model_key,
        "cache_only": True,
        "paid_api_calls": 0,
        "fixture_sha256": hashlib.sha256(source).hexdigest(),
        "holdout_sha256": hashlib.sha256(holdout_source).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "method": "Raw cosine over all synthetic documents; scope, expiry, fact and session "
        "eligibility use the real Store. Threshold semantic metrics rank by cosine alone; "
        "gate metrics use the current query_terms lexical policy and are not full service "
        "ranking metrics.",
        "limitations": "Small synthetic regression/calibration set, not a benchmark. The second "
        "dataset was authored before the initial scores, then used to diagnose thresholds "
        "and lexical function words; it is no longer an unseen holdout. Do not edit its "
        "texts or labels after observing results. "
        "A global threshold cannot remove irrelevant lexical matches or distinguish all "
        "semantically related but incorrect facts.",
        "recommendation": {
            "local_default_threshold": 0.50,
            "rationale": "Prefer a conservative semantic gate. The 0.45, 0.50 and 0.55 trials "
            "retain the same holdout positives; 0.50 adds margin over 0.45 without the "
            "additional translation loss observed at 0.60. This is a provisional default, "
            "not a general optimum or a guarantee of empty results.",
            "known_tradeoffs": [
                "The original oolong-beverage positive scores below the Oslo-to-Berlin "
                "negative. No single global cutoff can recover the former while rejecting "
                "the latter; retain both original labels and report the missed positive.",
                "The independent commute question is a valid low-cosine positive. "
                "Conservative thresholds miss it; do not relabel or replace this example.",
                "Initial calibration exposed irrelevant lexical function-word matches. "
                "Semantic thresholds cannot fix lexical matches; residual lexical matches "
                "under the current query_terms policy are recorded separately.",
            ],
        },
        "original_fixture": original,
        "holdout_dataset": independent,
        "holdout": holdout_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    "ranges": report[name]["ranges"],
                    "thresholds": [
                        {
                            "threshold": item["threshold"],
                            **item["semantic_only"],
                            "gate_pairs": item["expected_pairs_surviving_semantic_or_lexical_gate"],
                            "total_pairs": item["expected_pairs"],
                            "hybrid_gate_empty_passed": item["hybrid_gate_empty_passed"],
                        }
                        for item in report[name]["threshold_comparison"]
                    ],
                }
                for name in ("original_fixture", "holdout")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
