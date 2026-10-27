import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping.models import MemoryInput, Scope
from fishkeeping.storage import ConflictError, InvalidSourceError, Store

NOW = datetime(2026, 10, 28, tzinfo=UTC)
SCOPE = Scope(namespace="team", user_id="alice", agent_id="helper")
OTHER = Scope(namespace="team", user_id="bob", agent_id="helper")


@pytest.fixture
def store():
    instance = Store(":memory:")
    yield instance
    instance.close()


def item(content, **kwargs):
    return MemoryInput(scope=SCOPE, content=content, **kwargs)


def many_sources(store):
    parents, _ = store.put_many([item(f"evidence {n}") for n in range(151)], now=NOW)
    ids = [memory.id for memory in parents]
    result = store.put(item("shared conclusion", kind="semantic", source_ids=ids[:100]), now=NOW)
    merged = store.put(item("shared conclusion", kind="semantic", source_ids=ids[95:]), now=NOW)
    assert merged.id == result.id
    return merged, ids


def test_source_preview_full_pagination_and_all_read_paths(store):
    memory, ids = many_sources(store)
    assert memory.source_count == 151
    assert memory.source_ids == ids[:100]
    assert store.get(SCOPE, memory.id).source_count == 151
    duplicate = store.put(item("shared conclusion", kind="semantic", source_ids=ids[-20:]))
    assert duplicate.source_count == 151
    assert len(duplicate.source_ids) == 100
    assert store.refresh_active(SCOPE, [memory.id])[0].source_count == 151
    assert store.list(SCOPE, kinds=["semantic"])[0].source_count == 151
    assert store.lexical_candidates(SCOPE, "conclusion")[0].source_count == 151
    store.set_embedding(memory.id, "vectors", [1, 0])
    assert store.vector_candidates(SCOPE, "vectors", [1, 0])[0].source_count == 151
    page = store.source_page(SCOPE, memory.id, limit=100)
    assert page == {
        "memory_id": memory.id,
        "items": ids[:100],
        "total": 151,
        "limit": 100,
        "offset": 0,
        "has_more": True,
    }
    assert store.source_page(SCOPE, memory.id, offset=100)["items"] == ids[100:]
    assert not store.source_page(SCOPE, memory.id, offset=100)["has_more"]
    empty = store.source_page(SCOPE, memory.id, offset=999)
    assert empty["items"] == [] and empty["total"] == 151 and not empty["has_more"]
    assert store.source_page(OTHER, memory.id) is None
    assert store.source_page(SCOPE, "missing") is None
    for limit, offset in [(0, 0), (101, 0), (5, -1)]:
        with pytest.raises(ValueError):
            store.source_page(SCOPE, memory.id, limit=limit, offset=offset)
    store.rehearse(SCOPE, memory.id)
    store.archive(SCOPE, [memory.id])
    assert store.source_page(SCOPE, memory.id, offset=100)["items"] == ids[100:]
    assert store.get(SCOPE, memory.id, include_inactive=True).source_count == 151


def test_source_outside_preview_cascades_and_tombstones_events(store):
    memory, ids = many_sources(store)
    grandchild, _ = store.put_many(
        [item("grandchild", source_ids=[memory.id])], event_id="derived", event_hash="hash"
    )
    deleted = store.delete(SCOPE, ids[-1])
    assert set(deleted) == {ids[-1], memory.id, grandchild[0].id}
    assert store.lexical_candidates(SCOPE, "conclusion grandchild") == []
    assert store.get_embeddings([memory.id], "vectors") == {}
    assert store.source_page(SCOPE, memory.id) is None
    with pytest.raises(ConflictError, match="deleted"):
        store.get_event(SCOPE, "derived", "hash")
    assert store.get(SCOPE, ids[0]) is not None


def test_source_merge_scope_cycle_checks_use_complete_graph_and_rollback(store):
    memory, ids = many_sources(store)
    with pytest.raises(InvalidSourceError, match="cycle"):
        store.put(item("evidence 150", source_ids=[memory.id]))
    foreign = store.put(MemoryInput(scope=OTHER, content="foreign source"))
    good = store.put(item("valid extra evidence"))
    with pytest.raises(InvalidSourceError, match="scope"):
        store.put(item("shared conclusion", kind="semantic", source_ids=[good.id, foreign.id]))
    assert store.get(SCOPE, memory.id).source_count == 151
    assert store.source_page(SCOPE, memory.id, offset=100)["items"] == ids[100:]


def test_old_relevant_memory_survives_more_than_5000_recent_noise_records(store):
    old = store.put(item("古老的记忆：我喜欢茉莉花茶 sapphire"), now=NOW - timedelta(days=30))
    noise, _ = store.put_many([item(f"unrelated noise {n}") for n in range(5001)], now=NOW)
    assert old.id not in {m.id for m in store.list(SCOPE)}
    assert [m.id for m in store.lexical_candidates(SCOPE, "茉莉花茶", limit=1)] == [old.id]
    assert [m.id for m in store.lexical_candidates(SCOPE, "sapphire", limit=1)] == [old.id]
    store.set_embedding(old.id, "local", [1.0, 0.0])
    # Cache all the newer records too: top-k must run before the limit.
    with store._transaction():
        store._conn.executemany(
            "INSERT INTO embeddings VALUES (?,?,?)",
            [(memory.id, "local", "[0,1]") for memory in noise],
        )
    assert [m.id for m in store.vector_candidates(SCOPE, "local", [1, 0], limit=1)] == [old.id]
    assert store.count_active(SCOPE) == 5002


@pytest.mark.parametrize("query", ['"tea" OR *', 'tea ) NOT ("coffee', "tea:NEAR(x)", "tea'--"])
def test_match_syntax_is_literal_data(store, query):
    tea = store.put(item("tea preference"))
    assert tea.id in {m.id for m in store.lexical_candidates(SCOPE, query)}
    assert store.lexical_candidates(SCOPE, '" * : () --') == []


def test_application_tokens_preserved_in_full_text_index(store):
    hyphenated = store.put(item("re-index don't foo_bar"))
    separate = store.put(item("re index don t foo bar"))
    for query in ["re-index", "don't", "foo_bar"]:
        assert [m.id for m in store.lexical_candidates(SCOPE, query)] == [hyphenated.id]
    assert separate.id in {m.id for m in store.lexical_candidates(SCOPE, "index")}


def test_candidate_filters_apply_before_limits_and_counts(store):
    active = store.put(item("tea semantic", kind="semantic"), now=NOW)
    session = store.put(item("tea working", kind="working", session_id="a"), now=NOW)
    store.put(item("tea private", kind="working", session_id="b"), now=NOW)
    expired = store.put(item("tea expired", ttl_seconds=1), now=NOW - timedelta(seconds=2))
    archived = store.put(item("tea archived"), now=NOW)
    store.archive(SCOPE, [archived.id], now=NOW)
    foreign = store.put(MemoryInput(scope=OTHER, content="tea foreign"), now=NOW)
    for memory in [active, session, expired, archived, foreign]:
        store.set_embedding(memory.id, "local", [1, 0])
    assert [m.id for m in store.lexical_candidates(SCOPE, "tea", now=NOW)] == [active.id]
    assert [m.id for m in store.vector_candidates(SCOPE, "local", [1, 0], now=NOW)] == [active.id]
    assert store.count_active(SCOPE, now=NOW) == 1
    assert store.count_active(SCOPE, session_id="a", now=NOW) == 2
    assert store.embedding_coverage(SCOPE, "local", now=NOW) == {"total": 1, "indexed": 1}
    for method, args in [
        (store.lexical_candidates, ["tea"]),
        (store.vector_candidates, ["local", [1, 0]]),
    ]:
        assert {m.id for m in method(SCOPE, *args, session_id="a", now=NOW)} == {
            active.id,
            session.id,
        }
        assert [m.id for m in method(SCOPE, *args, session_id="a", kinds=["working"], now=NOW)] == [
            session.id
        ]
        assert method(SCOPE, *args, kinds=[], now=NOW) == []
        assert method(SCOPE, *args, limit=0, now=NOW) == []
        with pytest.raises(ValueError, match="limit"):
            method(SCOPE, *args, limit=-1, now=NOW)


def test_missing_embeddings_cover_oldest_long_term_scope_and_model(store):
    old = store.put(item("old missing", kind="semantic"), now=NOW - timedelta(days=10))
    newer = store.put(item("new missing"), now=NOW)
    cached = store.put(item("cached missing other model"), now=NOW)
    store.set_embedding(cached.id, "v1", [1, 0])
    store.put(item("working excluded", kind="working", session_id="s"), now=NOW)
    store.put(MemoryInput(scope=OTHER, content="other user"), now=NOW)
    assert [m.id for m in store.embedding_missing(SCOPE, "v1", limit=1, now=NOW)] == [old.id]
    assert {m.id for m in store.embedding_missing(SCOPE, "v1", now=NOW)} == {old.id, newer.id}
    assert store.embedding_coverage(SCOPE, "v1", now=NOW) == {"total": 3, "indexed": 1}
    assert store.embedding_coverage(SCOPE, "v2", now=NOW) == {"total": 3, "indexed": 0}
    assert store.embedding_coverage(SCOPE, "v1", kinds=["semantic"], now=NOW) == {
        "total": 1,
        "indexed": 0,
    }
    assert store.embedding_coverage(SCOPE, "v1", kinds=[], now=NOW) == {
        "total": 0,
        "indexed": 0,
    }
    assert store.embedding_missing(SCOPE, "v1", limit=0) == []
    with pytest.raises(ValueError):
        store.embedding_missing(SCOPE, "v1", limit=-1)


def test_vector_validation_and_invalid_cache_is_reported(store):
    valid = store.put(item("valid vector"))
    store.set_embedding(valid.id, "local", [1, 0])
    for n, vector_json in enumerate(
        ["[1,0,0]", "[true,0]", "[NaN,0]", "[]", "{}", "garbage", "[0,0]"]
    ):
        invalid = store.put(item(f"invalid cached vector {n}"))
        with store._transaction():
            store._conn.execute(
                "INSERT INTO embeddings VALUES (?,?,?)", (invalid.id, "local", vector_json)
            )
        with pytest.raises(ValueError, match="embedding cache"):
            store.vector_candidates(SCOPE, "local", [1, 0])
        store.delete(SCOPE, invalid.id)
    assert [m.id for m in store.vector_candidates(SCOPE, "local", [1, 0])] == [valid.id]
    for vector in [[], [0, 0], [True, 0], [float("nan"), 0], [float("inf"), 0], ["bad", 0]]:
        with pytest.raises(ValueError, match="embedding"):
            store.vector_candidates(SCOPE, "local", vector)


def test_provenance_beyond_preview_survives_restart(tmp_path):
    path = str(tmp_path / "many-sources.sqlite3")
    original = Store(path)
    memory, ids = many_sources(original)
    original.close()
    reopened = Store(path)
    try:
        assert reopened.get(SCOPE, memory.id).source_count == 151
        assert reopened.source_page(SCOPE, memory.id, offset=100)["items"] == ids[100:]
        assert memory.id in reopened.delete(SCOPE, ids[-1])
    finally:
        reopened.close()


def test_missing_fts5_has_actionable_diagnostic_and_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "no-fts.sqlite3"
    real_connect = sqlite3.connect

    class WithoutFTS(sqlite3.Connection):
        def execute(self, statement, *args, **kwargs):
            if statement.startswith("CREATE VIRTUAL TABLE"):
                raise sqlite3.OperationalError("no such module: fts5")
            return super().execute(statement, *args, **kwargs)

    def connect(*args, **kwargs):
        return real_connect(*args, **kwargs, factory=WithoutFTS)

    monkeypatch.setattr(sqlite3, "connect", connect)
    with pytest.raises(RuntimeError, match="ENABLE_FTS5"):
        Store(str(path))
    with real_connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='memories'").fetchone()
            is None
        )


def make_v1(path):
    original = Store(str(path))
    parent = original.put(item("migration parent"), now=NOW)
    child = original.put(item("migration child 茉莉花茶", source_ids=[parent.id]), now=NOW)
    original.close()
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE memory_fts")
        connection.execute("DELETE FROM sources")
        for memory in [parent, child]:
            document = memory.model_dump(mode="json")
            document.pop("source_count")
            connection.execute(
                "UPDATE memories SET document=? WHERE id=?", (json.dumps(document), memory.id)
            )
        connection.execute("PRAGMA user_version=1")
    return parent, child


def test_v1_migration_backfills_sources_and_fts_without_rewriting_documents(tmp_path):
    path = tmp_path / "v1.sqlite3"
    parent, child = make_v1(path)
    with sqlite3.connect(path) as connection:
        before = connection.execute("SELECT id,document FROM memories ORDER BY id").fetchall()
    for _ in range(2):
        migrated = Store(str(path))
        try:
            assert migrated._conn.execute("PRAGMA user_version").fetchone()[0] == 2
            assert migrated.source_page(SCOPE, child.id)["items"] == [parent.id]
            assert migrated.get(SCOPE, child.id).source_count == 1
            assert [m.id for m in migrated.lexical_candidates(SCOPE, "茉莉花茶")] == [child.id]
            assert [
                tuple(row)
                for row in migrated._conn.execute("SELECT id,document FROM memories ORDER BY id")
            ] == before
        finally:
            migrated.close()


def test_failed_migration_rolls_back_schema_and_backfills(tmp_path):
    path = tmp_path / "broken-v1.sqlite3"
    _, child = make_v1(path)
    with sqlite3.connect(path) as connection:
        document = json.loads(
            connection.execute("SELECT document FROM memories WHERE id=?", (child.id,)).fetchone()[
                0
            ]
        )
        document["source_ids"] = ["missing"]
        connection.execute(
            "UPDATE memories SET document=? WHERE id=?", (json.dumps(document), child.id)
        )
    with pytest.raises(sqlite3.IntegrityError):
        Store(str(path))
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM sources").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE name='memory_fts'").fetchone()
            is None
        )


def test_fts_index_changes_only_with_content_and_deletes(store):
    memory = store.put(item("original token"))
    statements = []
    store._conn.set_trace_callback(statements.append)
    try:
        store.rehearse(SCOPE, memory.id)
    finally:
        store._conn.set_trace_callback(None)
    assert not any(
        statement.startswith(("DELETE FROM memory_fts", "INSERT INTO memory_fts"))
        for statement in statements
    )
    memory.content = "replacement token"
    with store._transaction():
        store._save(memory)
    assert store.lexical_candidates(SCOPE, "original") == []
    assert [m.id for m in store.lexical_candidates(SCOPE, "replacement")] == [memory.id]
    store.delete(SCOPE, memory.id)
    assert store._conn.execute("SELECT COUNT(*) FROM memory_fts").fetchone()[0] == 0


def test_legacy_event_snapshot_source_count_is_compatible(store):
    parent = store.put(item("event source"))
    result, _ = store.put_many(
        [item("event child", source_ids=[parent.id])], event_id="legacy", event_hash="hash"
    )
    snapshot = result[0].model_dump(mode="json")
    snapshot.pop("source_count")
    with store._transaction():
        store._conn.execute("UPDATE events SET result_json=?", (json.dumps([snapshot]),))
    extra = store.put(item("new evidence"))
    store.put(item("event child", source_ids=[extra.id]))
    replay = store.get_event(SCOPE, "legacy", "hash")
    assert replay[0].source_count == 1  # historical event snapshot is immutable
    assert store.get(SCOPE, result[0].id).source_count == 2


def test_embedding_write_rechecks_activity_and_expiry_inside_transaction(store):
    active = store.put(item("active cache target"), now=NOW)
    expiring = store.put(item("expiring cache target", ttl_seconds=10), now=NOW)
    archived = store.put(item("archived cache target"), now=NOW)
    store.archive(SCOPE, [archived.id], now=NOW)
    assert store.set_embedding(active.id, "local", [1, 0], now=NOW)
    assert not store.set_embedding(archived.id, "local", [1, 0], now=NOW)
    assert not store.set_embedding(expiring.id, "local", [1, 0], now=NOW + timedelta(seconds=10))
    assert not store.set_embedding("missing", "local", [1, 0], now=NOW)
    assert store.get_embeddings([active.id, expiring.id, archived.id], "local") == {
        active.id: [1, 0]
    }


def test_clear_embeddings_is_scoped_model_specific_and_includes_history(store):
    active = store.put(item("active cached target"))
    archived = store.put(item("archived cached target"))
    foreign = store.put(MemoryInput(scope=OTHER, content="foreign cached target"))
    for memory in [active, archived, foreign]:
        store.set_embedding(memory.id, "local", [1, 0])
        store.set_embedding(memory.id, "other-model", [0, 1])
    store.archive(SCOPE, [archived.id])
    assert store.clear_embeddings(SCOPE, "local") == 2
    assert store.clear_embeddings(SCOPE, "local") == 0
    ids = [active.id, archived.id, foreign.id]
    assert store.get_embeddings(ids, "local") == {foreign.id: [1, 0]}
    assert len(store.get_embeddings(ids, "other-model")) == 3
    assert store.get(SCOPE, archived.id, include_inactive=True).status == "archived"
