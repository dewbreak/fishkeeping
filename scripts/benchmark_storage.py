"""Offline synthetic storage diagnostics; no .env, network, providers or user DB.

Run from the installed project environment, for example::

    python scripts/benchmark_storage.py --records 10000 \
        --output docs/validation/storage-scale-2026-10-28.json

The database is disposable SQLite :memory:. Document writes include validation,
provenance bookkeeping and FTS5 indexing. Dense 384-dimensional test vectors are
injected directly in one transaction; they are not model-generated embeddings.
Five measured queries report median/max, not production latency percentiles.

Vector candidate search still scans every eligible cached vector: O(N * D).
These measurements cannot establish a supported production size or SLA; ANN
indexing and concurrent, disk-backed workload measurements remain future work.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sqlite3
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fishkeeping.models import Memory, MemoryInput, Scope
from fishkeeping.storage import Store

DIMENSIONS = 384
REPETITIONS = 5
CANDIDATE_LIMIT = 100
MODEL_KEY = "synthetic-fixed-vectors-384-v1-no-model"
QUERY = "sapphire 茉莉花茶"


def measure(
    operation: Callable[[], list[Memory]], validate: Callable[[list[Memory]], None]
) -> dict:
    samples = []
    returned = []
    for _ in range(REPETITIONS):
        started = time.perf_counter()
        result = operation()
        samples.append(time.perf_counter() - started)
        validate(result)
        returned.append(len(result))
    return {
        "repetitions": REPETITIONS,
        "median_seconds": round(statistics.median(samples), 6),
        "max_seconds": round(max(samples), 6),
        "samples_seconds": [round(sample, 6) for sample in samples],
        "returned_records_per_query": returned,
        "candidate_limit": CANDIDATE_LIMIT,
    }


def benchmark(records: int) -> dict:
    if records < 2:
        raise ValueError("records must be at least 2")
    started = datetime.now(UTC)
    scope = Scope(namespace="synthetic-scale", user_id="own-user", agent_id="benchmark")
    other_scopes = [
        scope.model_copy(update={"namespace": "other-namespace"}),
        scope.model_copy(update={"user_id": "other-user"}),
        scope.model_copy(update={"agent_id": "other-agent"}),
    ]
    query_vector = [1.0] + [0.0] * (DIMENSIONS - 1)
    store = Store(":memory:")
    report = {
        "started_at": started.isoformat(),
        "status": "failed",
        "synthetic_data_only": True,
        "environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "os_version": platform.version(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "sqlite": sqlite3.sqlite_version,
            "database": ":memory:",
        },
        "dataset": {
            "scope_records": records,
            "old_relevant_records": 1,
            "newer_noise_records": records - 1,
            "foreign_scope_records": len(other_scopes),
            "total_database_records": records + len(other_scopes),
            "vector_dimensions": DIMENSIONS,
            "vector_source": "deterministic synthetic values, never a model provider",
            "vector_model_key": MODEL_KEY,
            "noise_vector_patterns": 257,
        },
        "cost": {"model_calls": 0, "network_requests": 0, "model_tokens": 0},
        "checks": {},
        "limitations": [
            "In-memory SQLite; excludes filesystem durability/fsync cost and disk contention.",
            "Single process and connection, without concurrent ingestion or deletion.",
            "Five repetitions describe this run's median/max, not production p95 or an SLA.",
            "Synthetic short documents and repeated dense vectors do not measure semantic quality.",
            "Vector retrieval remains exact O(N * dimensions); candidate limit caps returned "
            "records, not vectors scanned. Future ANN evaluation is required for larger workloads.",
            "No supported production record-count limit is inferred from this diagnostic.",
        ],
    }
    try:
        write_started = time.perf_counter()
        needle = store.put(
            MemoryInput(scope=scope, content="old unique sapphire memory: 我喜欢茉莉花茶"),
            now=started - timedelta(days=365),
        )
        identifiers = [needle.id]
        for offset in range(1, records, 256):
            batch = [
                MemoryInput(scope=scope, content=f"recent unrelated synthetic observation {index}")
                for index in range(offset, min(offset + 256, records))
            ]
            saved, _ = store.put_many(batch, now=started)
            identifiers.extend(memory.id for memory in saved)
        report["write_scope_memories_seconds"] = round(time.perf_counter() - write_started, 6)

        foreign = [
            store.put(MemoryInput(scope=other, content="foreign sapphire 茉莉花茶"), now=started)
            for other in other_scopes
        ]
        cache_started = time.perf_counter()
        # All noise vectors are orthogonal to the query, but dense in the other
        # 383 dimensions. Repeating 257 deterministic patterns avoids expensive
        # random-data setup while exercising all dimensions during exact search.
        noise_json = []
        for seed in range(257):
            values = [0.0] + [
                ((seed * 37 + column * 19) % 101 - 50) / 50 for column in range(1, DIMENSIONS)
            ]
            norm = math.hypot(*values)
            noise_json.append(json.dumps([round(value / norm, 9) for value in values]))
        needle_json = json.dumps(query_vector)
        with store._transaction():
            store._conn.executemany(
                "INSERT INTO embeddings(memory_id,model_key,vector_json) VALUES (?,?,?)",
                (
                    (
                        memory_id,
                        MODEL_KEY,
                        needle_json if index == 0 else noise_json[index % len(noise_json)],
                    )
                    for index, memory_id in enumerate(identifiers)
                ),
            )
            store._conn.executemany(
                "INSERT INTO embeddings(memory_id,model_key,vector_json) VALUES (?,?,?)",
                [(memory.id, MODEL_KEY, needle_json) for memory in foreign],
            )
        report["prepare_and_cache_vectors_seconds"] = round(time.perf_counter() - cache_started, 6)

        def validate(result: list[Memory]) -> None:
            if not result or result[0].id != needle.id:
                raise AssertionError("old relevant memory was not ranked first")
            if any(memory.scope != scope for memory in result):
                raise AssertionError("candidate escaped its requested scope")

        report["lexical_candidates"] = measure(
            lambda: store.lexical_candidates(scope, QUERY, limit=CANDIDATE_LIMIT), validate
        )
        report["vector_candidates"] = measure(
            lambda: store.vector_candidates(scope, MODEL_KEY, query_vector, limit=CANDIDATE_LIMIT),
            validate,
        )
        coverage = store.embedding_coverage(scope, MODEL_KEY)
        report["checks"]["complete_scope_vector_coverage"] = coverage == {
            "total": records,
            "indexed": records,
        }
        report["checks"]["old_needle_first_all_lexical_queries"] = True
        report["checks"]["old_needle_first_all_vector_queries"] = True
        report["checks"]["own_scope_queries_exclude_foreign_records"] = True
        for memory in foreign:
            lexical = store.lexical_candidates(memory.scope, QUERY, limit=CANDIDATE_LIMIT)
            vector = store.vector_candidates(
                memory.scope, MODEL_KEY, query_vector, limit=CANDIDATE_LIMIT
            )
            report["checks"][f"foreign_{other_scopes.index(memory.scope)}_scope_isolation"] = [
                candidate.id for candidate in lexical
            ] == [memory.id] and [candidate.id for candidate in vector] == [memory.id]
        if not all(report["checks"].values()):
            raise AssertionError("scope isolation or coverage diagnostic failed")
        report["status"] = "passed"
        report["completed_at"] = datetime.now(UTC).isoformat()
        return report
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--records", type=int, default=10000, help="Own-scope records; default 10000"
    )
    parser.add_argument(
        "--output", type=Path, help="Optional JSON report path; stdout always contains JSON"
    )
    args = parser.parse_args()
    if args.records < 2:
        parser.error("--records must be at least 2")
    report = benchmark(args.records)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
