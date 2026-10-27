"""Durable event receipts and bounded background extraction.

An accepted event is committed before provider work. Claims and model-attempt
budgets share SQLite transactions across workers. A lost lease is deliberately
not retried automatically: the upstream request may already have been billed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from pydantic import ValidationError

from fishkeeping.config import Settings
from fishkeeping.conversations import encode_conversation, trusted_conversation_content
from fishkeeping.models import Extraction, IngestRequest, Memory, MemoryInput, Message, Scope
from fishkeeping.providers import ProviderError
from fishkeeping.storage import ConflictError, InvalidSourceError, Store

_RETRYABLE = {"rate_limited", "upstream_unavailable"}
_ERROR_CODES = {
    "not_configured",
    "authentication_failed",
    "insufficient_balance",
    "permission_denied",
    "model_unavailable",
    "invalid_request",
    "rate_limited",
    "upstream_unavailable",
    "timeout",
    "network_error",
    "outcome_unknown",
    "invalid_response",
    "truncated_response",
    "http_error",
    "provider_error",
}


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _scope(row: sqlite3.Row) -> Scope:
    return Scope(namespace=row["namespace"], user_id=row["user_id"], agent_id=row["agent_id"])


class EventQueue:
    """A persistent work queue using the Store connection and transaction lock."""

    def __init__(
        self, store: Store, settings: Settings, *, clock: Callable[[], datetime] | None = None
    ):
        self.store = store
        self.settings = settings
        self._clock = clock or (lambda: datetime.now(UTC))
        with store._transaction():
            store._conn.execute(
                "CREATE TABLE IF NOT EXISTS extraction_queue_schema (version INTEGER NOT NULL)"
            )
            versions = store._conn.execute("SELECT version FROM extraction_queue_schema").fetchall()
            if versions and (len(versions) != 1 or versions[0][0] != 1):
                raise ValueError("unsupported extraction queue schema version")
            if not versions:
                store._conn.execute("INSERT INTO extraction_queue_schema VALUES (1)")
            store._conn.execute(
                """CREATE TABLE IF NOT EXISTS durable_events (
                    namespace TEXT NOT NULL, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    event_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    episode_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                    PRIMARY KEY(namespace,user_id,agent_id,event_id)
                )"""
            )
            store._conn.execute(
                """CREATE TABLE IF NOT EXISTS extraction_jobs (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL, user_id TEXT NOT NULL, agent_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    episode_id TEXT REFERENCES memories(id) ON DELETE SET NULL,
                    state TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    lease_token TEXT, lease_expires_at TEXT,
                    next_attempt_at TEXT, last_error_code TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, completed_at TEXT,
                    result_ids TEXT NOT NULL DEFAULT '[]',
                    UNIQUE(namespace,user_id,agent_id,event_id)
                )"""
            )
            store._conn.execute(
                "CREATE INDEX IF NOT EXISTS extraction_jobs_ready "
                "ON extraction_jobs(state,next_attempt_at,created_at)"
            )
            store._conn.execute(
                """CREATE TABLE IF NOT EXISTS extraction_daily_budget (
                    utc_day TEXT PRIMARY KEY, attempts INTEGER NOT NULL
                )"""
            )

    def _now(self) -> datetime:
        current = self._clock()
        return current.replace(tzinfo=UTC) if current.tzinfo is None else current.astimezone(UTC)

    @staticmethod
    def _public_job(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {
            key: row[key]
            for key in (
                "id",
                "event_id",
                "state",
                "attempts",
                "max_attempts",
                "next_attempt_at",
                "last_error_code",
                "created_at",
                "updated_at",
            )
        } | {"result_ids": json.loads(row["result_ids"])}

    def _job(self, scope: Scope, event_id: str) -> sqlite3.Row | None:
        return self.store._conn.execute(
            "SELECT * FROM extraction_jobs WHERE namespace=? AND user_id=? AND agent_id=? "
            "AND event_id=?",
            (*scope.key(), event_id),
        ).fetchone()

    def _refresh_states(self, now: datetime) -> None:
        timestamp = _stamp(now)
        # Foreign-key SET NULL is immediate, even if a worker is currently in a provider call.
        self.store._conn.execute(
            """UPDATE extraction_jobs SET state='cancelled', last_error_code='source_unavailable',
                lease_token=NULL,lease_expires_at=NULL,next_attempt_at=NULL,result_ids='[]',
                updated_at=?,completed_at=?
                WHERE state!='cancelled' AND (episode_id IS NULL OR
                    (state IN ('pending','processing') AND episode_id NOT IN (
                        SELECT id FROM memories WHERE status='active'
                        AND (expires_at IS NULL OR expires_at>?)
                    )))""",
            (timestamp, timestamp, timestamp),
        )
        self.store._conn.execute(
            """UPDATE extraction_jobs SET state='failed',last_error_code='outcome_unknown',
                lease_token=NULL,lease_expires_at=NULL,next_attempt_at=NULL,
                updated_at=?,completed_at=?
                WHERE state='processing' AND lease_expires_at<=?""",
            (timestamp, timestamp, timestamp),
        )

    def _receipt(self, scope: Scope, event_id: str, replayed: bool) -> dict | None:
        durable = self.store._conn.execute(
            "SELECT * FROM durable_events WHERE namespace=? AND user_id=? AND agent_id=? "
            "AND event_id=?",
            (*scope.key(), event_id),
        ).fetchone()
        if durable is None:
            return None
        if self.store._event(scope, event_id, durable["payload_hash"]) is None:
            raise ConflictError("event receipt is no longer available")
        episode = self.store._fetch(scope, durable["episode_id"]) if durable["episode_id"] else None
        if episode is None:
            raise ConflictError("event source was deleted; use a new event_id for new data")
        return {
            "event_id": event_id,
            "scope": scope,
            "episode": episode,
            "job": self._public_job(self._job(scope, event_id)),
            "replayed": replayed,
        }

    def enqueue(self, request: IngestRequest) -> dict:
        payload = {"delivery": "durable-v1", "request": request.model_dump(mode="json")}
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        transcript, provenance = encode_conversation(request.messages)
        with self.store._transaction():
            now = self._now()
            self._refresh_states(now)
            previous = self.store._event(request.scope, request.event_id, digest)
            if previous is not None:
                receipt = self._receipt(request.scope, request.event_id, replayed=True)
                if receipt is None:
                    raise ConflictError("event_id belongs to another ingestion mode")
                return receipt
            episode = self.store._put(
                MemoryInput(
                    scope=request.scope,
                    content=transcript,
                    kind="episodic",
                    session_id=request.session_id,
                    metadata={"event_id": request.event_id, **provenance},
                ),
                now,
            )
            self.store._conn.execute(
                "INSERT INTO events VALUES (?,?,?,?,?,?,?,0)",
                (
                    *request.scope.key(),
                    request.event_id,
                    digest,
                    json.dumps([episode.model_dump(mode="json")], ensure_ascii=False),
                    json.dumps([episode.id]),
                ),
            )
            self.store._conn.execute(
                "INSERT INTO durable_events VALUES (?,?,?,?,?,?)",
                (*request.scope.key(), request.event_id, digest, episode.id),
            )
            if request.extract:
                self.store._conn.execute(
                    """INSERT INTO extraction_jobs (
                        id,namespace,user_id,agent_id,event_id,episode_id,state,max_attempts,
                        next_attempt_at,created_at,updated_at
                    ) VALUES (?,?,?,?,?,?,'pending',?,?,?,?)""",
                    (
                        uuid4().hex,
                        *request.scope.key(),
                        request.event_id,
                        episode.id,
                        getattr(self.settings, "worker_max_attempts", 3),
                        _stamp(now),
                        _stamp(now),
                        _stamp(now),
                    ),
                )
            return self._receipt(request.scope, request.event_id, replayed=False)

    def get(self, scope: Scope, event_id: str) -> dict | None:
        with self.store._transaction():
            self._refresh_states(self._now())
            return self._receipt(scope, event_id, replayed=False)

    def retry(self, scope: Scope, event_id: str) -> dict:
        with self.store._transaction():
            now = self._now()
            self._refresh_states(now)
            receipt = self._receipt(scope, event_id, replayed=False)
            if receipt is None or receipt["job"] is None:
                raise ConflictError("no extraction job exists in this scope")
            job = self._job(scope, event_id)
            if job["state"] == "pending":
                return self._public_job(job)
            if job["state"] != "failed" or job["attempts"] >= job["max_attempts"]:
                raise ConflictError("job cannot be retried in its current state or attempt budget")
            self.store._conn.execute(
                """UPDATE extraction_jobs SET state='pending',next_attempt_at=?,updated_at=?,
                    completed_at=NULL,lease_token=NULL,lease_expires_at=NULL WHERE id=?""",
                (_stamp(now), _stamp(now), job["id"]),
            )
            return self._public_job(self._job(scope, event_id))

    def budget_status(self) -> dict:
        with self.store._lock:
            day = self._now().date().isoformat()
            row = self.store._conn.execute(
                "SELECT attempts FROM extraction_daily_budget WHERE utc_day=?", (day,)
            ).fetchone()
            attempts = row[0] if row else 0
            limit = getattr(self.settings, "worker_daily_call_limit", 20)
            return {
                "utc_day": day,
                "attempts": attempts,
                "limit": limit,
                "remaining": max(0, limit - attempts),
            }

    def _claim(self) -> tuple[sqlite3.Row, list[Message] | None] | None:
        with self.store._transaction():
            now = self._now()
            self._refresh_states(now)
            row = self.store._conn.execute(
                """SELECT * FROM extraction_jobs WHERE state='pending' AND next_attempt_at<=?
                    ORDER BY next_attempt_at,created_at,id LIMIT 1""",
                (_stamp(now),),
            ).fetchone()
            if row is None:
                return None
            scope = _scope(row)
            episode = self.store.get(scope, row["episode_id"], now=now)
            try:
                content = trusted_conversation_content(episode)
                messages = (
                    []
                    if content is None
                    else [
                        Message(role="user", content=value)
                        for value in json.loads(content)["user_messages"]
                    ]
                )
            except (ValueError, KeyError, TypeError, AttributeError):
                self.store._conn.execute(
                    """UPDATE extraction_jobs SET state='failed',last_error_code='invalid_source',
                        next_attempt_at=NULL,updated_at=?,completed_at=? WHERE id=?""",
                    (_stamp(now), _stamp(now), row["id"]),
                )
                return self._job(scope, row["event_id"]), None
            if not messages:
                self.store._conn.execute(
                    """UPDATE extraction_jobs SET state='succeeded',next_attempt_at=NULL,
                        updated_at=?,completed_at=? WHERE id=?""",
                    (_stamp(now), _stamp(now), row["id"]),
                )
                return self._job(scope, row["event_id"]), None
            if row["attempts"] >= row["max_attempts"]:
                self.store._conn.execute(
                    """UPDATE extraction_jobs
                        SET state='failed',last_error_code='attempts_exhausted',
                        next_attempt_at=NULL,updated_at=?,completed_at=? WHERE id=?""",
                    (_stamp(now), _stamp(now), row["id"]),
                )
                return self._job(scope, row["event_id"]), None
            budget = self.budget_status()
            if budget["remaining"] <= 0:
                return None
            self.store._conn.execute(
                """INSERT INTO extraction_daily_budget VALUES (?,1)
                    ON CONFLICT(utc_day) DO UPDATE SET attempts=attempts+1""",
                (budget["utc_day"],),
            )
            lease = uuid4().hex
            lease_seconds = self.settings.request_timeout * 2 + 30
            self.store._conn.execute(
                """UPDATE extraction_jobs SET state='processing',attempts=attempts+1,
                    lease_token=?,lease_expires_at=?,next_attempt_at=NULL,last_error_code=NULL,
                    updated_at=? WHERE id=?""",
                (lease, _stamp(now + timedelta(seconds=lease_seconds)), _stamp(now), row["id"]),
            )
            return self._job(scope, row["event_id"]), messages

    def _current_claim(self, claimed: sqlite3.Row) -> sqlite3.Row | None:
        row = self._job(_scope(claimed), claimed["event_id"])
        if row and row["state"] == "processing" and row["lease_token"] == claimed["lease_token"]:
            return row
        return None

    def _finish(self, claimed: sqlite3.Row, facts: list) -> dict:
        with self.store._transaction():
            now = self._now()
            self._refresh_states(now)
            row = self._current_claim(claimed)
            if row is None:
                return self._public_job(self._job(_scope(claimed), claimed["event_id"]))
            scope = _scope(row)
            results: list[Memory] = []
            for fact in facts:
                results.append(
                    self.store._put(
                        MemoryInput(
                            scope=scope,
                            **fact.model_dump(),
                            source_ids=[row["episode_id"]],
                            metadata={"method": "deepseek_extraction", "event_id": row["event_id"]},
                        ),
                        now,
                        require_active_sources=True,
                    )
                )
            result_ids = list(dict.fromkeys(memory.id for memory in results))
            self.store._conn.execute(
                """UPDATE extraction_jobs SET state='succeeded',result_ids=?,lease_token=NULL,
                    lease_expires_at=NULL,next_attempt_at=NULL,last_error_code=NULL,
                    updated_at=?,completed_at=? WHERE id=?""",
                (json.dumps(result_ids), _stamp(now), _stamp(now), row["id"]),
            )
            # Deleting any derived result also tombstones this receipt in Store.delete.
            self.store._conn.execute(
                "UPDATE events SET memory_ids=? WHERE namespace=? AND user_id=? AND agent_id=? "
                "AND event_id=?",
                (json.dumps([row["episode_id"], *result_ids]), *scope.key(), row["event_id"]),
            )
            return self._public_job(self._job(scope, row["event_id"]))

    def _fail(self, claimed: sqlite3.Row, code: str) -> dict:
        with self.store._transaction():
            now = self._now()
            self._refresh_states(now)
            row = self._current_claim(claimed)
            if row is None:
                return self._public_job(self._job(_scope(claimed), claimed["event_id"]))
            retry = code in _RETRYABLE and row["attempts"] < row["max_attempts"]
            next_attempt = now + timedelta(seconds=2 ** row["attempts"]) if retry else None
            self.store._conn.execute(
                """UPDATE extraction_jobs SET state=?,last_error_code=?,lease_token=NULL,
                    lease_expires_at=NULL,next_attempt_at=?,updated_at=?,completed_at=?
                    WHERE id=?""",
                (
                    "pending" if retry else "failed",
                    code,
                    _stamp(next_attempt) if next_attempt else None,
                    _stamp(now),
                    None if retry else _stamp(now),
                    row["id"],
                ),
            )
            return self._public_job(self._job(_scope(row), row["event_id"]))

    def process_one(self, provider) -> dict | None:
        """Process one ready job. No provider means no claim and no budget debit."""
        if provider is None:
            return None
        claimed = self._claim()
        if claimed is None:
            return None
        row, messages = claimed
        if messages is None:
            return self._public_job(row)
        try:
            output = provider.extract(messages)
            facts = Extraction.model_validate(
                {
                    "memories": [
                        fact.model_dump() if hasattr(fact, "model_dump") else fact
                        for fact in output
                    ]
                },
                strict=True,
            ).memories
        except ProviderError as exc:
            code = exc.code if exc.code in _ERROR_CODES else "provider_error"
            return self._fail(row, code)
        except (ValidationError, TypeError, ValueError):
            return self._fail(row, "invalid_response")
        except Exception:
            return self._fail(row, "provider_error")
        try:
            return self._finish(row, facts)
        except InvalidSourceError:
            return self._fail(row, "source_unavailable")
        except (sqlite3.Error, ValidationError, ValueError):
            # Provider already ran. A failed local commit is not safe to auto-repeat.
            return self._fail(row, "commit_failed")
