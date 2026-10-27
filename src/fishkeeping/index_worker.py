"""Bounded local indexing, resumed from missing cache rows after every restart.

The worker never invokes remote embeddings. SQLite leases fence stale workers;
record attempts are reserved before CPU work and are not refunded after a crash.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from fishkeeping.local_embeddings import LocalEmbeddingProvider
from fishkeeping.models import Scope
from fishkeeping.providers import ProviderError
from fishkeeping.storage import Store, _valid_vector


class IndexWorker:
    """One scope/model worker sharing a database-wide UTC daily record budget.

    The effective daily limit can only decrease during a UTC day. A new day
    adopts the first worker's configured limit. Manual indexing is independent.
    """

    def __init__(
        self,
        store: Store,
        embeddings: LocalEmbeddingProvider,
        scope: Scope,
        *,
        daily_limit: int = 1000,
        batch_size: int = 32,
        lease_seconds: float = 300,
        clock: Callable[[], datetime] | None = None,
    ):
        if not isinstance(embeddings, LocalEmbeddingProvider):
            raise ValueError("Incremental indexing requires a local embedding provider")
        if not 0 <= daily_limit <= 1_000_000 or not 1 <= batch_size <= 32:
            raise ValueError("daily_limit must be 0..1000000 and batch_size must be 1..32")
        if not 1 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be 1..3600")
        self.store = store
        self.embeddings = embeddings
        self.scope = scope
        self.model_key = embeddings.model_key
        self.daily_limit = daily_limit
        self.batch_size = batch_size
        self.lease_seconds = lease_seconds
        self._clock = clock or (lambda: datetime.now(UTC))
        with store._transaction():
            store._conn.execute(
                "CREATE TABLE IF NOT EXISTS index_worker_schema (version INTEGER NOT NULL)"
            )
            versions = store._conn.execute("SELECT version FROM index_worker_schema").fetchall()
            if versions and (len(versions) != 1 or versions[0][0] != 1):
                raise ValueError("unsupported index worker schema version")
            if not versions:
                store._conn.execute("INSERT INTO index_worker_schema VALUES (1)")
            store._conn.execute(
                """CREATE TABLE IF NOT EXISTS index_worker_leases (
                    namespace TEXT NOT NULL,user_id TEXT NOT NULL,agent_id TEXT NOT NULL,
                    model_key TEXT NOT NULL,token TEXT NOT NULL,expires_at TEXT NOT NULL,
                    PRIMARY KEY(namespace,user_id,agent_id,model_key)
                )"""
            )
            store._conn.execute(
                """CREATE TABLE IF NOT EXISTS index_worker_daily_budget (
                    utc_day TEXT PRIMARY KEY,attempts INTEGER NOT NULL,daily_limit INTEGER NOT NULL
                )"""
            )

    def _now(self) -> datetime:
        now = self._clock()
        return now.replace(tzinfo=UTC) if now.tzinfo is None else now.astimezone(UTC)

    def _budget(self, now: datetime) -> dict:
        row = self.store._conn.execute(
            "SELECT attempts,daily_limit FROM index_worker_daily_budget WHERE utc_day=?",
            (now.date().isoformat(),),
        ).fetchone()
        attempts = row[0] if row else 0
        limit = min(self.daily_limit, row[1]) if row else self.daily_limit
        return {
            "utc_day": now.date().isoformat(),
            "attempts": attempts,
            "limit": limit,
            "remaining": max(0, limit - attempts),
        }

    def budget_status(self) -> dict:
        with self.store._lock:
            return self._budget(self._now())

    def _result(self, state: str, reserved: int = 0, processed: int = 0) -> dict:
        coverage = self.store.embedding_coverage(self.scope, self.model_key, now=self._now())
        return {
            "state": state,
            "reserved": reserved,
            "processed": processed,
            **coverage,
            "complete": coverage["indexed"] == coverage["total"],
            "model_key": self.model_key,
            "budget": self.budget_status(),
        }

    def _release(self, token: str) -> None:
        self.store._conn.execute(
            "DELETE FROM index_worker_leases WHERE namespace=? AND user_id=? AND agent_id=? "
            "AND model_key=? AND token=?",
            (*self.scope.key(), self.model_key, token),
        )

    def process_one(self) -> dict:
        """Reserve and encode at most 32 records / 32000 characters, then checkpoint."""
        token = str(uuid4())
        with self.store._transaction():
            now = self._now()
            stamp = now.isoformat(timespec="microseconds")
            self.store._conn.execute(
                "DELETE FROM index_worker_leases WHERE expires_at<=?", (stamp,)
            )
            leased = self.store._conn.execute(
                "SELECT 1 FROM index_worker_leases WHERE namespace=? AND user_id=? AND agent_id=? "
                "AND model_key=?",
                (*self.scope.key(), self.model_key),
            ).fetchone()
            if leased:
                return self._result("busy")
            budget = self._budget(now)
            self.store._conn.execute(
                "INSERT INTO index_worker_daily_budget VALUES (?,?,?) "
                "ON CONFLICT(utc_day) DO UPDATE SET "
                "daily_limit=MIN(daily_limit,excluded.daily_limit)",
                (budget["utc_day"], 0, self.daily_limit),
            )
            if budget["remaining"] == 0:
                return self._result("budget_exhausted")
            candidates = self.store.embedding_missing(
                self.scope,
                self.model_key,
                limit=min(self.batch_size, budget["remaining"]),
                now=now,
            )
            batch, characters = [], 0
            for memory in candidates:
                if characters + len(memory.content) > 32_000:
                    break
                batch.append(memory)
                characters += len(memory.content)
            if not batch:
                return self._result("idle")
            self.store._conn.execute(
                "INSERT INTO index_worker_leases VALUES (?,?,?,?,?,?)",
                (
                    *self.scope.key(),
                    self.model_key,
                    token,
                    (now + timedelta(seconds=self.lease_seconds)).isoformat(
                        timespec="microseconds"
                    ),
                ),
            )
            self.store._conn.execute(
                "UPDATE index_worker_daily_budget SET attempts=attempts+? WHERE utc_day=?",
                (len(batch), budget["utc_day"]),
            )
        try:
            vectors = self.embeddings.embed([memory.content for memory in batch])
            if len(vectors) != len(batch) or any(not _valid_vector(vector) for vector in vectors):
                raise ProviderError("Invalid local embedding batch")
            if len({len(vector) for vector in vectors}) != 1:
                raise ProviderError("Inconsistent local embedding dimensions")
        except (ProviderError, ValueError):
            with self.store._transaction():
                self._release(token)
            return self._result("provider_error", reserved=len(batch))
        except BaseException:
            with self.store._transaction():
                self._release(token)
            raise
        processed = 0
        with self.store._transaction():
            stamp = self._now().isoformat(timespec="microseconds")
            owns_lease = self.store._conn.execute(
                "SELECT 1 FROM index_worker_leases WHERE namespace=? AND user_id=? AND agent_id=? "
                "AND model_key=? AND token=? AND expires_at>?",
                (*self.scope.key(), self.model_key, token, stamp),
            ).fetchone()
            if not owns_lease:
                self._release(token)
                return self._result("lease_lost", reserved=len(batch))
            for memory, vector in zip(batch, vectors, strict=True):
                # The eligibility check and cache insert share the lease's transaction.
                cursor = self.store._conn.execute(
                    "INSERT OR IGNORE INTO embeddings(memory_id,model_key,vector_json) "
                    "SELECT id,?,? FROM memories WHERE id=? AND namespace=? AND user_id=? "
                    "AND agent_id=? AND status='active' AND (expires_at IS NULL OR expires_at>?)",
                    (self.model_key, json.dumps(vector), memory.id, *self.scope.key(), stamp),
                )
                processed += cursor.rowcount
            self._release(token)
        return self._result("completed", reserved=len(batch), processed=processed)
