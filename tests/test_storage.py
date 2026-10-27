import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping.models import MemoryInput, Scope
from fishkeeping.storage import ConflictError, InvalidSourceError, Store

NOW = datetime(2026, 10, 28, tzinfo=UTC)
SCOPE = Scope(namespace="team", user_id="alice", agent_id="helper")


@pytest.fixture
def store():
    instance = Store(":memory:")
    yield instance
    instance.close()


def item(content="I prefer tea", **kwargs):
    return MemoryInput(scope=SCOPE, content=content, **kwargs)


def test_persistence_and_scope_isolation(tmp_path):
    path = str(tmp_path / "nested" / "memory.sqlite3")
    first = Store(path)
    memory = first.put(item(kind="semantic"), now=NOW)
    first.close()
    second = Store(path)
    try:
        assert second.get(SCOPE, memory.id) == memory
        other_scopes = [
            SCOPE.model_copy(update={"namespace": "another"}),
            SCOPE.model_copy(update={"user_id": "bob"}),
            SCOPE.model_copy(update={"agent_id": "another"}),
        ]
        for scope in other_scopes:
            assert second.get(scope, memory.id, include_inactive=True) is None
            assert second.list(scope, include_inactive=True) == []
            assert second.delete(scope, memory.id) == []
            assert second.rehearse(scope, memory.id) is None
    finally:
        second.close()


def test_duplicate_preserves_identity_and_count(store):
    first = store.put(item(kind="semantic"), now=NOW)
    again = store.put(item(content="  I prefer tea  ", kind="semantic"), now=NOW)
    assert again.id == first.id
    assert store.stats(SCOPE)["total"] == 1
    distinct = store.put(item(kind="semantic", fact_key="drink"), now=NOW)
    assert distinct.id != first.id
    # Case and indentation may be meaningful procedural/code data.
    assert store.put(item(content="I prefer Tea", kind="semantic"), now=NOW).id != first.id


def test_fact_versions_and_expired_replacement(store):
    old = store.put(item(kind="semantic", fact_key="drink"), now=NOW)
    new = store.put(item("I prefer coffee", kind="semantic", fact_key="drink"), now=NOW)
    assert store.get(SCOPE, old.id, now=NOW) is None
    history = store.get(SCOPE, old.id, include_inactive=True)
    assert history.status == "superseded"
    assert history.superseded_by == new.id
    assert store.list(SCOPE, now=NOW) == [new]
    expiring = store.put(
        item("short", kind="semantic", fact_key="temporary", ttl_seconds=2), now=NOW
    )
    replacement = store.put(
        item("short", kind="semantic", fact_key="temporary", ttl_seconds=2),
        now=NOW + timedelta(seconds=3),
    )
    assert replacement.id != expiring.id
    assert store.get(SCOPE, expiring.id, include_inactive=True).status == "archived"


def test_working_session_and_expiration(store):
    one = store.put(item("now", kind="working", session_id="one"), now=NOW)
    two = store.put(item("elsewhere", kind="working", session_id="two"), now=NOW)
    long_term = store.put(item("remember", session_id="two"), now=NOW)
    assert one.expires_at == NOW + timedelta(hours=1)
    assert {m.id for m in store.list(SCOPE, now=NOW)} == {one.id, two.id, long_term.id}
    assert {m.id for m in store.list(SCOPE, session_id="one", now=NOW)} == {one.id, long_term.id}
    assert store.get(SCOPE, one.id, now=NOW + timedelta(hours=1)) is None
    assert store.rehearse(SCOPE, one.id, now=NOW + timedelta(hours=1)) is None
    assert store.get(SCOPE, one.id, include_inactive=True) is not None


def test_event_replay_and_conflict(store):
    batch = [item("User: I prefer tea", session_id="chat"), item(kind="semantic")]
    result, replayed = store.put_many(batch, event_id="event", event_hash="hash", source_event=True)
    assert not replayed
    assert result[1].source_ids == [result[0].id]
    replay, replayed = store.put_many(batch, event_id="event", event_hash="hash", source_event=True)
    assert replayed and replay == result
    assert store.get_event(SCOPE, "event", "hash") == result
    with pytest.raises(ConflictError):
        store.put_many(batch, event_id="event", event_hash="different")
    assert store.stats(SCOPE)["total"] == 2


def test_failed_batch_rolls_back_everything(store):
    with pytest.raises(InvalidSourceError):
        store.put_many(
            [item("before"), item("after", source_ids=["missing"])],
            event_id="event",
            event_hash="hash",
        )
    assert store.list(SCOPE, include_inactive=True) == []
    assert store.get_event(SCOPE, "event", "hash") is None


def test_cross_scope_and_cyclic_sources_are_rejected(store):
    other = store.put(MemoryInput(scope=Scope(user_id="bob"), content="secret"))
    with pytest.raises(InvalidSourceError):
        store.put(item(source_ids=[other.id]))
    parent = store.put(item("parent"))
    child = store.put(item("child", source_ids=[parent.id]))
    with pytest.raises(InvalidSourceError):
        store.put(item("parent", source_ids=[child.id]))
    with pytest.raises(InvalidSourceError):
        store.put(item("parent", source_ids=[parent.id]))


def test_duplicate_merges_provenance_and_delete_cascades(store):
    parent = store.put(item("first observation"))
    second = store.put(item("another observation"))
    child = store.put(item(kind="semantic", source_ids=[parent.id]))
    duplicate = store.put(item(kind="semantic", source_ids=[second.id]))
    assert duplicate.id == child.id
    assert duplicate.source_ids == [parent.id, second.id]
    grandchild = store.put(item("higher summary", source_ids=[child.id]))
    store.set_embedding(child.id, "model", [0.5, 0.25])
    deleted = store.delete(SCOPE, parent.id)
    assert set(deleted) == {parent.id, child.id, grandchild.id}
    assert store.get_embeddings([child.id], "model") == {}
    assert store.get(SCOPE, second.id) is not None


def test_deleted_event_is_tombstoned_and_cannot_restore_content(store):
    memories, _ = store.put_many(
        [item("sensitive original"), item("sensitive derived", kind="semantic")],
        event_id="event",
        event_hash="hash",
        source_event=True,
    )
    store.delete(SCOPE, memories[0].id)
    with pytest.raises(ConflictError):
        store.get_event(SCOPE, "event", "hash")
    with pytest.raises(ConflictError):
        store.put_many([item("sensitive original")], event_id="event", event_hash="hash")
    row = store._conn.execute("SELECT result_json,memory_ids,tombstoned FROM events").fetchone()
    assert tuple(row) == ("[]", "[]", 1)
    assert store.stats(SCOPE)["total"] == 0


def test_rehearsal_is_explicit_and_archive_does_not_overwrite_history(store):
    old = store.put(item(kind="semantic", fact_key="drink"), now=NOW)
    store.get(SCOPE, old.id, now=NOW)
    store.list(SCOPE, now=NOW)
    assert store.get(SCOPE, old.id, now=NOW).access_count == 0
    reinforced = store.rehearse(SCOPE, old.id, now=NOW + timedelta(seconds=1))
    assert reinforced.access_count == 1
    assert reinforced.last_accessed_at == NOW + timedelta(seconds=1)
    current = store.put(item("coffee", kind="semantic", fact_key="drink"), now=NOW)
    assert store.archive(SCOPE, [old.id, current.id], now=NOW) == [current.id]
    assert store.get(SCOPE, old.id, include_inactive=True).status == "superseded"


def test_multiple_connections_serialize_idempotent_writes(tmp_path):
    path = str(tmp_path / "memory.sqlite3")
    stores = [Store(path) for _ in range(4)]
    try:

        def write(index):
            return stores[index % len(stores)].put_many(
                [item()], event_id="same", event_hash="same"
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(write, range(24)))
        assert len({result[0][0].id for result in results}) == 1
        assert sum(not result[1] for result in results) == 1
        assert stores[0].stats(SCOPE)["total"] == 1
    finally:
        for instance in stores:
            instance.close()


def test_embeddings_cache_version_validation_and_deleted_race(store):
    memory = store.put(item())
    store.set_embedding(memory.id, "model-v1", [0.0, 1.0])
    store.set_embedding(memory.id, "model-v2", [1.0, 0.0, 0.0])
    assert store.get_embeddings([memory.id], "model-v1") == {memory.id: [0.0, 1.0]}
    assert store.get_embeddings([], "model-v1") == {}
    for bad in ([], [float("nan")], [float("inf")], [True]):
        with pytest.raises(ValueError):
            store.set_embedding(memory.id, "bad", bad)
    store.delete(SCOPE, memory.id)
    store.set_embedding(memory.id, "model-v1", [1.0, 2.0])
    assert store.get_embeddings([memory.id], "model-v1") == {}


def test_unknown_schema_is_rejected(tmp_path):
    path = str(tmp_path / "newer.sqlite3")
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA user_version=42")
    connection.close()
    with pytest.raises(ValueError, match="unsupported database schema"):
        Store(path)


def test_consolidation_sources_must_still_be_active_inside_transaction(store):
    expired = store.put(item("short lived", ttl_seconds=2), now=NOW)
    archived = store.put(item("archived source"), now=NOW)
    store.archive(SCOPE, [archived.id], now=NOW)
    for source in (expired, archived):
        with pytest.raises(InvalidSourceError, match="no longer active"):
            store.put_many(
                [item("new candidate", source_ids=[source.id])],
                require_active_sources=True,
                now=NOW + timedelta(seconds=3),
            )
    assert store.stats(SCOPE)["total"] == 2
    # Explicit historical links remain available when no active-source condition is requested.
    historical = store.put(item("historical annotation", source_ids=[archived.id]), now=NOW)
    assert historical.source_ids == [archived.id]


def test_event_source_created_in_same_transaction_is_active(store):
    memories, _ = store.put_many(
        [item("raw observation"), item("derived knowledge", kind="semantic")],
        source_event=True,
        require_active_sources=True,
        now=NOW,
    )
    assert memories[1].source_ids == [memories[0].id]


def test_fact_supersession_archives_all_derived_conclusions(store):
    old = store.put(item("I live in Paris", kind="semantic", fact_key="city"), now=NOW)
    child = store.put(item("User lives in France", kind="semantic", source_ids=[old.id]), now=NOW)
    grandchild = store.put(item("Choose French timezone", source_ids=[child.id]), now=NOW)
    unrelated = store.put(item("unrelated"), now=NOW)
    current = store.put(item("I live in Berlin", kind="semantic", fact_key="city"), now=NOW)
    assert store.get(SCOPE, current.id, now=NOW) is not None
    assert store.get(SCOPE, unrelated.id, now=NOW) is not None
    assert store.get(SCOPE, old.id, include_inactive=True).status == "superseded"
    for derived in (child, grandchild):
        assert store.get(SCOPE, derived.id, now=NOW) is None
        history = store.get(SCOPE, derived.id, include_inactive=True)
        assert history.status == "archived"
        assert history.source_ids == derived.source_ids
    # A new supported claim gets a fresh ID even if an old summary had identical text.
    revised = store.put(item("User lives in France", kind="semantic", source_ids=[current.id]))
    assert revised.id != child.id


def test_archive_skips_records_updated_after_snapshot(store):
    before = store.put(item("before"), now=NOW)
    untouched = store.put(item("untouched"), now=NOW)
    store.rehearse(SCOPE, before.id, now=NOW + timedelta(seconds=1))
    archived = store.archive(
        SCOPE,
        [before.id, untouched.id],
        now=NOW + timedelta(seconds=2),
        expected_updated_at={before.id: before.updated_at, untouched.id: untouched.updated_at},
    )
    assert archived == [untouched.id]
    assert store.get(SCOPE, before.id, now=NOW + timedelta(seconds=2)) is not None


def test_refresh_active_is_scoped_ordered_and_filters_invalidated_records(store):
    first = store.put(item("first"), now=NOW)
    second = store.put(item("second"), now=NOW)
    expired = store.put(item("expired", ttl_seconds=1), now=NOW)
    archived = store.put(item("archived"), now=NOW)
    deleted = store.put(item("deleted"), now=NOW)
    foreign = store.put(MemoryInput(scope=Scope(user_id="bob"), content="foreign"), now=NOW)
    store.archive(SCOPE, [archived.id], now=NOW)
    store.delete(SCOPE, deleted.id)
    store.rehearse(SCOPE, first.id, now=NOW + timedelta(seconds=1))
    result = store.refresh_active(
        SCOPE,
        [second.id, expired.id, archived.id, first.id, deleted.id, foreign.id, "missing"],
        now=NOW + timedelta(seconds=2),
    )
    assert [memory.id for memory in result] == [second.id, first.id]
    assert result[1].access_count == 1
    assert store.refresh_active(SCOPE, []) == []


def test_refresh_active_handles_more_than_one_sqlite_chunk(store):
    memories, _ = store.put_many([item(f"item-{index}") for index in range(510)], now=NOW)
    ids = [memory.id for memory in reversed(memories)]
    assert [memory.id for memory in store.refresh_active(SCOPE, ids, now=NOW)] == ids
