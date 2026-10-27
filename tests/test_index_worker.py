import json
from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping import cli
from fishkeeping.config import Settings
from fishkeeping.index_worker import IndexWorker
from fishkeeping.local_embeddings import LocalEmbeddingProvider
from fishkeeping.models import MemoryInput, RecallRequest, Scope
from fishkeeping.providers import ProviderError
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

SCOPE = Scope(user_id="index-worker")


class FakeLocal(LocalEmbeddingProvider):
    def __init__(self, key="local-test-v1", callback=None, error=None):
        super().__init__(Settings(_env_file=None))
        self.model_key = key
        self.calls = []
        self.callback = callback
        self.error = error

    def embed(self, texts):
        self.calls.append(texts)
        if self.callback:
            self.callback()
        if self.error:
            raise self.error
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


@pytest.fixture
def store(tmp_path):
    result = Store(str(tmp_path / "index-worker.sqlite3"))
    yield result
    result.close()


def add(store, number=0, **kwargs):
    return store.put(MemoryInput(scope=SCOPE, content=f"Document {number}", **kwargs))


def test_restart_resumes_only_missing_and_auto_discovers_later_writes(tmp_path):
    path = str(tmp_path / "restart.sqlite3")
    store = Store(path)
    for number in range(5):
        add(store, number)
    first = FakeLocal()
    worker = IndexWorker(store, first, SCOPE, batch_size=2)
    assert worker.process_one()["processed"] == 2
    store.close()
    store = Store(path)
    try:
        second = FakeLocal()
        worker = IndexWorker(store, second, SCOPE, batch_size=2)
        assert worker.process_one()["processed"] == 2
        assert worker.process_one()["processed"] == 1
        assert worker.process_one()["state"] == "idle"
        later = add(store, "later")
        assert worker.process_one()["processed"] == 1
        assert second.calls[-1] == [later.content]
        assert worker.budget_status()["attempts"] == 6
        assert len({text for call in [*first.calls, *second.calls] for text in call}) == 6
    finally:
        store.close()


def test_budget_is_persistent_global_and_does_not_increase_midday(store):
    for number in range(5):
        add(store, number)
    other_scope = Scope(user_id="second-scope")
    store.put(MemoryInput(scope=other_scope, content="other document"))
    clock = [datetime.now(UTC)]
    first = IndexWorker(
        store, FakeLocal(), SCOPE, daily_limit=3, batch_size=2, clock=lambda: clock[0]
    )
    assert first.process_one()["processed"] == 2
    other = IndexWorker(store, FakeLocal(), other_scope, daily_limit=100, clock=lambda: clock[0])
    assert other.process_one()["processed"] == 1
    assert other.budget_status()["limit"] == 3
    assert first.process_one()["state"] == "budget_exhausted"
    clock[0] += timedelta(days=1)
    assert first.process_one()["processed"] == 2
    assert first.budget_status()["attempts"] == 2


def test_failures_consume_budget_but_missing_work_is_retryable(store):
    add(store)
    provider = FakeLocal(error=ProviderError("private upstream error"))
    worker = IndexWorker(store, provider, SCOPE, daily_limit=2)
    result = worker.process_one()
    assert result["state"] == "provider_error"
    assert "private" not in json.dumps(result)
    assert result["budget"]["attempts"] == 1
    provider.error = None
    assert worker.process_one()["processed"] == 1
    assert worker.budget_status()["attempts"] == 2


def test_concurrent_connection_cannot_claim_active_lease(store, tmp_path):
    add(store)
    second_store = Store(str(tmp_path / "index-worker.sqlite3"))
    try:
        second_provider = FakeLocal()
        second = IndexWorker(second_store, second_provider, SCOPE)
        nested = []
        first_provider = FakeLocal(callback=lambda: nested.append(second.process_one()))
        result = IndexWorker(store, first_provider, SCOPE).process_one()
        assert result["processed"] == 1
        assert nested[0]["state"] == "busy"
        assert second_provider.calls == []
        assert second.budget_status()["attempts"] == 1
    finally:
        second_store.close()


def test_expired_lease_resumes_and_fences_original_worker(store, tmp_path):
    add(store)
    now = [datetime.now(UTC)]
    second_store = Store(str(tmp_path / "index-worker.sqlite3"))
    second = IndexWorker(second_store, FakeLocal(), SCOPE, clock=lambda: now[0], lease_seconds=1)
    nested = []

    def lose_lease():
        now[0] += timedelta(seconds=2)
        nested.append(second.process_one())

    first = IndexWorker(
        store, FakeLocal(callback=lose_lease), SCOPE, clock=lambda: now[0], lease_seconds=1
    )
    try:
        result = first.process_one()
        assert result["state"] == "lease_lost"
        assert result["processed"] == 0
        assert nested[0]["processed"] == 1
        assert first.budget_status()["attempts"] == 2
    finally:
        second_store.close()


@pytest.mark.parametrize("mutation", ["delete", "archive", "expire", "replace"])
def test_lifecycle_change_during_encoding_cannot_recreate_cache(store, mutation):
    memory = add(store, kind="semantic", fact_key="preference")

    def change():
        if mutation == "delete":
            store.delete(SCOPE, memory.id)
        elif mutation == "archive":
            store.archive(SCOPE, [memory.id])
        elif mutation == "expire":
            memory.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            with store._transaction():
                store._save(memory)
        else:
            add(store, "replacement", kind="semantic", fact_key="preference")

    provider = FakeLocal(callback=change)
    worker = IndexWorker(store, provider, SCOPE)
    result = worker.process_one()
    assert result["processed"] == 0
    assert store.get_embeddings([memory.id], provider.model_key) == {}
    if mutation == "replace":
        provider.callback = None
        assert worker.process_one()["processed"] == 1


def test_skips_expired_working_and_foreign_records_and_separates_models(store):
    add(store)
    add(store, "session", kind="working", session_id="session")
    store.put(
        MemoryInput(scope=SCOPE, content="expired", ttl_seconds=1),
        now=datetime.now(UTC) - timedelta(seconds=2),
    )
    store.put(MemoryInput(scope=Scope(user_id="foreign"), content="foreign"))
    first = IndexWorker(store, FakeLocal(), SCOPE)
    assert first.process_one()["processed"] == 1
    assert first.process_one()["state"] == "idle"
    second = IndexWorker(store, FakeLocal(key="new-model"), SCOPE)
    assert second.process_one()["processed"] == 1
    assert second.process_one()["complete"]


def test_batch_character_and_record_caps(store):
    for number in range(4):
        store.put(MemoryInput(scope=SCOPE, content=str(number) + "x" * 15999))
    provider = FakeLocal()
    worker = IndexWorker(store, provider, SCOPE, batch_size=32)
    assert worker.process_one()["reserved"] == 2
    assert sum(map(len, provider.calls[0])) == 32_000
    with pytest.raises(ValueError, match="batch_size"):
        IndexWorker(store, provider, SCOPE, batch_size=33)
    with pytest.raises(ValueError, match="local"):
        IndexWorker(store, object(), SCOPE)


def test_unknown_worker_schema_rejected_without_changing_memory_schema(store):
    add(store)
    IndexWorker(store, FakeLocal(), SCOPE)
    store._conn.execute("UPDATE index_worker_schema SET version=2")
    with pytest.raises(ValueError, match="schema version"):
        IndexWorker(store, FakeLocal(), SCOPE)
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert store.stats(SCOPE)["total"] == 1


def test_recall_reports_partial_then_worker_completes_without_query_document_encoding(store):
    memory = add(store)
    provider = FakeLocal()
    service = MemoryService(store, Settings(_env_file=None), embeddings=provider)
    result = service.recall(RecallRequest(scope=SCOPE, query="Document"))
    assert result.warnings and provider.calls == []
    assert IndexWorker(store, provider, SCOPE).process_one()["complete"]
    result = service.recall(RecallRequest(scope=SCOPE, query="Document"))
    assert result.retrieval_mode == "hybrid"
    assert result.hits[0].memory.id == memory.id
    assert len(provider.calls) == 1


def test_cli_once_is_local_only_bounded_and_redacts(tmp_path, monkeypatch, capsys):
    settings = Settings(_env_file=None, db_path=str(tmp_path / "cli.sqlite3"))
    with pytest.raises(SystemExit, match="BACKEND=local"):
        cli.run_index_worker(
            settings, SCOPE, once=True, batch_size=1, max_batches=2, daily_limit=5, poll_interval=1
        )
    settings.embedding_backend = "local"
    store = Store(settings.db_path)
    add(store, "private content")
    add(store, "another")
    store.close()
    provider = FakeLocal()
    monkeypatch.setattr(cli, "create_embeddings", lambda settings: provider)
    monkeypatch.setattr(cli.time, "sleep", lambda seconds: pytest.fail("--once slept"))
    cli.run_index_worker(
        settings, SCOPE, once=True, batch_size=1, max_batches=2, daily_limit=5, poll_interval=1
    )
    output = capsys.readouterr().out
    assert "private content" not in output
    assert json.loads(output.splitlines()[-1])["processed"] == 1
    assert provider._closed


def test_cli_empty_poll_uses_backoff_and_discovers_later_memory(tmp_path, monkeypatch, capsys):
    settings = Settings(
        _env_file=None, db_path=str(tmp_path / "poll.sqlite3"), embedding_backend="local"
    )
    monkeypatch.setattr(cli, "create_embeddings", lambda settings: FakeLocal())
    delays = []

    def sleep(delay):
        delays.append(delay)
        if len(delays) == 3:
            store = Store(settings.db_path)
            add(store)
            store.close()

    monkeypatch.setattr(cli.time, "sleep", sleep)
    cli.run_index_worker(
        settings, SCOPE, once=False, batch_size=1, max_batches=1, daily_limit=5, poll_interval=2
    )
    assert delays == [2, 4, 8]
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item.get("state") for item in output] == ["idle", "completed", None]
