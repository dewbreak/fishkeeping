"""Transactional SQLite storage with scoped provenance and idempotent events.

One connection is serialized within a Store. ``BEGIN IMMEDIATE`` also serializes
writers across Store instances/processes. Event tombstones prevent deleted data
from returning through a retry, while discarding the deleted response payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fishkeeping.models import Kind, Memory, MemoryInput, Scope


class ConflictError(ValueError):
    """An event ID has another payload or its original result was deleted."""


class InvalidSourceError(ValueError):
    """A provenance reference is missing, crosses a scope, or creates a cycle."""


class NotFoundError(LookupError):
    """A requested memory does not exist in the caller's scope."""


def _time(value: datetime | None = None) -> datetime:
    value = value or datetime.now(UTC)
    # Clock injection is an internal API; interpret naive test clocks as UTC.
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds")


def _fingerprint(item: MemoryInput) -> str:
    # Preserve case and internal whitespace (which may be executable-code data).
    content = unicodedata.normalize("NFC", item.content.strip())
    value = [item.kind, item.session_id, item.fact_key, content]
    if (
        item.metadata.get("method") == "conversation"
        and "conversation_fingerprint" in item.metadata
    ):
        # Identical readable text can represent different message roles/boundaries.
        # Legacy records keep their original fingerprint until explicitly re-ingested.
        value.append(item.metadata["conversation_fingerprint"])
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode("utf-8")).hexdigest()


class Store:
    """Single-node durable memory repository. Call ``close`` when done."""

    def __init__(self, path: str):
        if path != ":memory:":
            Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
            path = str(Path(path).expanduser())
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            path, timeout=30, isolation_level=None, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            with self._transaction():
                version = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if version not in (0, 1):
                    raise ValueError(f"unsupported database schema version: {version}")
                statements = [
                    """CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        namespace TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        status TEXT NOT NULL,
                        session_id TEXT,
                        fact_key TEXT,
                        fingerprint TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        expires_at TEXT,
                        document TEXT NOT NULL
                    )""",
                    """CREATE INDEX IF NOT EXISTS memories_scope
                        ON memories(namespace,user_id,agent_id,status,created_at DESC)""",
                    """CREATE UNIQUE INDEX IF NOT EXISTS memories_active_fingerprint
                        ON memories(namespace,user_id,agent_id,fingerprint)
                        WHERE status='active'""",
                    """CREATE UNIQUE INDEX IF NOT EXISTS memories_active_fact
                        ON memories(namespace,user_id,agent_id,fact_key)
                        WHERE status='active' AND fact_key IS NOT NULL""",
                    """CREATE TABLE IF NOT EXISTS sources (
                        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        source_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        PRIMARY KEY(memory_id,source_id)
                    )""",
                    "CREATE INDEX IF NOT EXISTS sources_parent ON sources(source_id)",
                    """CREATE TABLE IF NOT EXISTS events (
                        namespace TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        agent_id TEXT NOT NULL,
                        event_id TEXT NOT NULL,
                        payload_hash TEXT NOT NULL,
                        result_json TEXT NOT NULL,
                        memory_ids TEXT NOT NULL,
                        tombstoned INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(namespace,user_id,agent_id,event_id)
                    )""",
                    """CREATE TABLE IF NOT EXISTS embeddings (
                        memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                        model_key TEXT NOT NULL,
                        vector_json TEXT NOT NULL,
                        PRIMARY KEY(memory_id,model_key)
                    )""",
                ]
                for statement in statements:
                    self._conn.execute(statement)
                self._conn.execute("PRAGMA user_version=1")
        except BaseException:
            self._conn.close()
            raise

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _fetch(self, scope: Scope, memory_id: str) -> Memory | None:
        row = self._conn.execute(
            "SELECT document FROM memories WHERE namespace=? AND user_id=? AND agent_id=? AND id=?",
            (*scope.key(), memory_id),
        ).fetchone()
        return Memory.model_validate_json(row[0]) if row else None

    def _save(self, memory: Memory) -> None:
        self._conn.execute(
            """INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status, expires_at=excluded.expires_at,
                document=excluded.document""",
            (
                memory.id,
                *memory.scope.key(),
                memory.kind,
                memory.status,
                memory.session_id,
                memory.fact_key,
                _fingerprint(memory),
                _timestamp(memory.created_at),
                _timestamp(memory.expires_at) if memory.expires_at else None,
                memory.model_dump_json(),
            ),
        )
        for source_id in memory.source_ids:
            self._conn.execute(
                "INSERT OR IGNORE INTO sources(memory_id,source_id) VALUES (?,?)",
                (memory.id, source_id),
            )

    def _validate_sources(
        self,
        scope: Scope,
        source_ids: list[str],
        memory_id: str | None = None,
        require_active: bool = False,
        now: datetime | None = None,
    ) -> None:
        for source_id in source_ids:
            source = self._fetch(scope, source_id)
            if source is None:
                raise InvalidSourceError("source does not exist in the requested scope")
            if require_active and (
                source.status != "active"
                or (source.expires_at is not None and source.expires_at <= _time(now))
            ):
                raise InvalidSourceError("source is no longer active in the requested scope")
            if memory_id and self._would_create_cycle(memory_id, source_id):
                raise InvalidSourceError("source would create a provenance cycle")

    def _would_create_cycle(self, memory_id: str, source_id: str) -> bool:
        # Following a proposed source's parents must not reach this memory.
        row = self._conn.execute(
            """WITH RECURSIVE ancestors(id) AS (
                SELECT ? UNION SELECT s.source_id FROM sources s
                JOIN ancestors a ON s.memory_id=a.id
            ) SELECT 1 FROM ancestors WHERE id=? LIMIT 1""",
            (source_id, memory_id),
        ).fetchone()
        return row is not None

    def _archive_descendants(self, scope: Scope, memory_id: str, now: datetime) -> None:
        """Invalidate conclusions derived from a superseded fact, retaining history."""
        rows = self._conn.execute(
            """WITH RECURSIVE descendants(id) AS (
                SELECT s.memory_id FROM sources s WHERE s.source_id=?
                UNION SELECT s.memory_id FROM sources s
                JOIN descendants d ON s.source_id=d.id
            ) SELECT m.document FROM descendants d JOIN memories m ON m.id=d.id
            WHERE m.namespace=? AND m.user_id=? AND m.agent_id=? AND m.status='active'""",
            (memory_id, *scope.key()),
        ).fetchall()
        for row in rows:
            derived = Memory.model_validate_json(row[0])
            derived.status = "archived"
            derived.updated_at = now
            self._save(derived)

    def _put(
        self,
        item: MemoryInput,
        now: datetime,
        require_active_sources: bool = False,
        consolidation: bool = False,
    ) -> Memory:
        self._validate_sources(
            item.scope, item.source_ids, require_active=require_active_sources, now=now
        )
        fingerprint = _fingerprint(item)
        rows = self._conn.execute(
            """SELECT document FROM memories WHERE namespace=? AND user_id=?
                AND agent_id=? AND status='active'
                AND (fingerprint=? OR (fact_key IS NOT NULL AND fact_key=?))""",
            (*item.scope.key(), fingerprint, item.fact_key),
        ).fetchall()
        existing: list[Memory] = []
        for row in rows:
            memory = Memory.model_validate_json(row[0])
            if memory.expires_at is not None and memory.expires_at <= now:
                if item.fact_key is not None:
                    self._archive_descendants(item.scope, memory.id, now)
                memory.status = "archived"
                memory.updated_at = now
                self._save(memory)
            else:
                existing.append(memory)
        for memory in existing:
            if _fingerprint(memory) == fingerprint:
                requested_sources = item.source_ids
                if consolidation:
                    # An LLM may restate a source or its ancestor exactly. Reuse
                    # that knowledge instead of adding circular evidence links.
                    # Scope, existence and optional active checks ran above for
                    # EVERY requested source, including links omitted here.
                    requested_sources = [
                        source_id
                        for source_id in requested_sources
                        if not self._would_create_cycle(memory.id, source_id)
                    ]
                new_sources = list(dict.fromkeys([*memory.source_ids, *requested_sources]))
                if len(new_sources) > 100:
                    raise InvalidSourceError("a memory cannot reference more than 100 sources")
                self._validate_sources(item.scope, new_sources, memory.id)
                if new_sources != memory.source_ids:
                    memory.source_ids = new_sources
                    memory.updated_at = now
                    self._save(memory)
                return memory
        memory = Memory(
            **item.model_dump(),
            id=uuid4().hex,
            created_at=now,
            updated_at=now,
            expires_at=(now + timedelta(seconds=item.ttl_seconds)) if item.ttl_seconds else None,
        )
        for old in existing:
            self._archive_descendants(item.scope, old.id, now)
            old.status = "superseded"
            old.superseded_by = memory.id
            old.updated_at = now
            self._save(old)
        self._save(memory)
        return memory

    def put(self, item: MemoryInput, now: datetime | None = None) -> Memory:
        return self.put_many([item], now=now)[0][0]

    def _event(self, scope: Scope, event_id: str, event_hash: str) -> list[Memory] | None:
        row = self._conn.execute(
            "SELECT * FROM events WHERE namespace=? AND user_id=? AND agent_id=? AND event_id=?",
            (*scope.key(), event_id),
        ).fetchone()
        if row is None:
            return None
        if row["payload_hash"] != event_hash:
            raise ConflictError("event_id was already used with a different payload")
        if row["tombstoned"]:
            raise ConflictError("event result was deleted; use a new event_id for new data")
        ids = json.loads(row["memory_ids"])
        if any(self._fetch(scope, memory_id) is None for memory_id in ids):
            raise ConflictError("event result is no longer available")
        return [Memory.model_validate(item) for item in json.loads(row["result_json"])]

    def get_event(self, scope: Scope, event_id: str, event_hash: str) -> list[Memory] | None:
        with self._lock:
            return self._event(scope, event_id, event_hash)

    def put_many(
        self,
        items: list[MemoryInput],
        event_id: str | None = None,
        event_hash: str | None = None,
        now: datetime | None = None,
        source_event: bool = False,
        require_active_sources: bool = False,
        consolidation: bool = False,
    ) -> tuple[list[Memory], bool]:
        if consolidation and (source_event or event_id is not None or event_hash is not None):
            raise ValueError("consolidation cannot be combined with event ingestion")
        if not items:
            if event_id is not None or source_event:
                raise ValueError("an event must contain at least one memory")
            return [], False
        if (event_id is None) != (event_hash is None):
            raise ValueError("event_id and event_hash must be supplied together")
        scope = items[0].scope
        if any(item.scope != scope for item in items):
            raise ValueError("a batch must belong to one scope")
        if source_event and items[0].kind != "episodic":
            raise ValueError("the first source event memory must be episodic")
        with self._transaction():
            # Evaluate expiry after waiting for the write lock, not before it.
            clock = _time(now)
            if event_id is not None:
                original = self._event(scope, event_id, event_hash)
                if original is not None:
                    return original, True
            result: list[Memory] = []
            for index, item in enumerate(items):
                if index and source_event:
                    source_ids = list(dict.fromkeys([*item.source_ids, result[0].id]))
                    item = MemoryInput.model_validate(
                        {**item.model_dump(), "source_ids": source_ids}
                    )
                result.append(self._put(item, clock, require_active_sources, consolidation))
            if event_id is not None:
                self._conn.execute(
                    "INSERT INTO events VALUES (?,?,?,?,?,?,?,0)",
                    (
                        *scope.key(),
                        event_id,
                        event_hash,
                        json.dumps([m.model_dump(mode="json") for m in result], ensure_ascii=False),
                        json.dumps([m.id for m in result]),
                    ),
                )
            return result, False

    def get(
        self,
        scope: Scope,
        memory_id: str,
        include_inactive: bool = False,
        now: datetime | None = None,
    ) -> Memory | None:
        with self._lock:
            memory = self._fetch(scope, memory_id)
            if memory is None or include_inactive:
                return memory
            if memory.status != "active" or (
                memory.expires_at is not None and memory.expires_at <= _time(now)
            ):
                return None
            return memory

    def list(
        self,
        scope: Scope,
        include_inactive: bool = False,
        session_id: str | None = None,
        kinds: list[Kind] | None = None,
        limit: int = 5000,
        offset: int = 0,
        now: datetime | None = None,
    ) -> list[Memory]:
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be nonnegative")
        clauses = ["namespace=?", "user_id=?", "agent_id=?"]
        params: list = list(scope.key())
        if not include_inactive:
            clauses.extend(["status='active'", "(expires_at IS NULL OR expires_at>?)"])
            params.append(_timestamp(_time(now)))
        if session_id is not None:
            clauses.append("(kind!='working' OR session_id=?)")
            params.append(session_id)
        if kinds is not None:
            if not kinds:
                return []
            clauses.append("kind IN (" + ",".join("?" for _ in kinds) + ")")
            params.extend(kinds)
        query = "SELECT document FROM memories WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
            return [Memory.model_validate_json(row[0]) for row in rows]

    def rehearse(self, scope: Scope, memory_id: str, now: datetime | None = None) -> Memory | None:
        with self._transaction():
            clock = _time(now)
            memory = self.get(scope, memory_id, now=clock)
            if memory is None:
                return None
            memory.access_count += 1
            memory.last_accessed_at = clock
            memory.updated_at = clock
            self._save(memory)
            return memory

    def refresh_active(
        self, scope: Scope, ids: list[str], now: datetime | None = None
    ) -> list[Memory]:
        """Reload a candidate set after external work, discarding stale/expired rows.

        A read transaction gives all chunks one SQLite snapshot. The returned order
        follows the requested IDs, so refreshing never changes candidate priority.
        """
        if not ids:
            return []
        current: dict[str, Memory] = {}
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                clock = _timestamp(_time(now))
                unique_ids = list(dict.fromkeys(ids))
                for start in range(0, len(unique_ids), 500):
                    chunk = unique_ids[start : start + 500]
                    placeholders = ",".join("?" for _ in chunk)
                    rows = self._conn.execute(
                        "SELECT id,document FROM memories WHERE namespace=? AND user_id=? "
                        "AND agent_id=? AND status='active' "
                        "AND (expires_at IS NULL OR expires_at>?) "
                        f"AND id IN ({placeholders})",
                        (*scope.key(), clock, *chunk),
                    ).fetchall()
                    current.update(
                        {row["id"]: Memory.model_validate_json(row["document"]) for row in rows}
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        return [current[memory_id] for memory_id in ids if memory_id in current]

    def archive(
        self,
        scope: Scope,
        ids: list[str],
        now: datetime | None = None,
        expected_updated_at: dict[str, datetime] | None = None,
    ) -> list[str]:
        clock = _time(now)
        archived = []
        with self._transaction():
            for memory_id in dict.fromkeys(ids):
                memory = self._fetch(scope, memory_id)
                if memory is not None and memory.status == "active":
                    if expected_updated_at is not None and (
                        memory.id not in expected_updated_at
                        or memory.updated_at != expected_updated_at[memory.id]
                    ):
                        continue
                    memory.status = "archived"
                    memory.updated_at = clock
                    self._save(memory)
                    archived.append(memory_id)
        return archived

    def delete(self, scope: Scope, memory_id: str) -> list[str]:
        with self._transaction():
            if self._fetch(scope, memory_id) is None:
                return []
            rows = self._conn.execute(
                """WITH RECURSIVE descendants(id) AS (
                    SELECT ? UNION SELECT s.memory_id FROM sources s
                    JOIN descendants d ON s.source_id=d.id
                ) SELECT m.id FROM descendants d JOIN memories m ON m.id=d.id
                WHERE m.namespace=? AND m.user_id=? AND m.agent_id=?""",
                (memory_id, *scope.key()),
            ).fetchall()
            ids = {row[0] for row in rows}
            # Do not retain forgotten content inside an idempotency snapshot.
            events = self._conn.execute(
                "SELECT event_id,memory_ids FROM events WHERE namespace=? AND user_id=? "
                "AND agent_id=? AND tombstoned=0",
                scope.key(),
            ).fetchall()
            for event in events:
                if ids.intersection(json.loads(event["memory_ids"])):
                    self._conn.execute(
                        "UPDATE events SET tombstoned=1,result_json='[]',memory_ids='[]' "
                        "WHERE namespace=? AND user_id=? AND agent_id=? AND event_id=?",
                        (*scope.key(), event["event_id"]),
                    )
            # Keep historical versions superseded but remove dangling pointers.
            survivors = self._conn.execute(
                "SELECT document FROM memories WHERE namespace=? AND user_id=? AND agent_id=?",
                scope.key(),
            ).fetchall()
            for row in survivors:
                memory = Memory.model_validate_json(row[0])
                if memory.id not in ids and memory.superseded_by in ids:
                    memory.superseded_by = None
                    self._save(memory)
            for deleted_id in ids:
                self._conn.execute("DELETE FROM memories WHERE id=?", (deleted_id,))
            return sorted(ids)

    def stats(self, scope: Scope) -> dict:
        with self._lock:
            rows = self._conn.execute(
                "SELECT kind,status,expires_at FROM memories "
                "WHERE namespace=? AND user_id=? AND agent_id=?",
                scope.key(),
            ).fetchall()
            now = _timestamp(_time())
            by_kind = {kind: 0 for kind in ("working", "episodic", "semantic", "procedural")}
            by_status = {status: 0 for status in ("active", "archived", "superseded")}
            expired = 0
            for row in rows:
                by_kind[row["kind"]] += 1
                by_status[row["status"]] += 1
                if row["expires_at"] is not None and row["expires_at"] <= now:
                    expired += 1
            return {
                "total": len(rows),
                "by_kind": by_kind,
                "by_status": by_status,
                "expired": expired,
            }

    def get_embeddings(self, ids: list[str], model_key: str) -> dict[str, list[float]]:
        result = {}
        with self._lock:
            # Chunk to stay below SQLite's variable limit on older distributions.
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = self._conn.execute(
                    "SELECT memory_id,vector_json FROM embeddings WHERE model_key=? "
                    f"AND memory_id IN ({placeholders})",
                    [model_key, *chunk],
                ).fetchall()
                result.update({row[0]: json.loads(row[1]) for row in rows})
        return result

    def set_embedding(self, memory_id: str, model_key: str, vector: list[float]) -> None:
        if not vector or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in vector
        ):
            raise ValueError("embedding must be a nonempty list of finite numbers")
        with self._transaction():
            exists = self._conn.execute(
                "SELECT 1 FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
            if exists is None:
                # A concurrent deletion must not recreate a memory through its cache.
                return
            self._conn.execute(
                "INSERT INTO embeddings VALUES (?,?,?) ON CONFLICT(memory_id,model_key) "
                "DO UPDATE SET vector_json=excluded.vector_json",
                (memory_id, model_key, json.dumps(vector, allow_nan=False)),
            )
