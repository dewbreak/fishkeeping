"""Cross-layer regressions found during independent storage/service review."""

from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping.config import Settings
from fishkeeping.models import (
    ConsolidateRequest,
    ExtractedFact,
    MaintenanceRequest,
    MemoryInput,
    RecallRequest,
    Scope,
)
from fishkeeping.service import MemoryService
from fishkeeping.storage import InvalidSourceError, Store


@pytest.fixture
def store():
    instance = Store(":memory:")
    yield instance
    instance.close()


def test_working_memories_do_not_consume_long_term_candidate_budget(store):
    scope = Scope(user_id="alice")
    now = datetime.now(UTC)
    durable = store.put(
        MemoryInput(scope=scope, content="The preferred database is SQLite", kind="semantic"),
        now=now - timedelta(days=1),
    )
    for index in range(105):
        store.put(
            MemoryInput(
                scope=scope,
                content=f"Current scratchpad item {index}",
                kind="working",
                session_id="busy-session",
            ),
            now=now,
        )
    service = MemoryService(store, Settings(candidate_limit=100, _env_file=None))
    response = service.recall(RecallRequest(scope=scope, query="SQLite"))
    assert [hit.memory.id for hit in response.hits] == [durable.id]


def test_consolidation_rechecks_active_sources_after_provider_returns(store):
    scope = Scope(user_id="alice")
    original = store.put(MemoryInput(scope=scope, content="An observation"))

    class RacingProvider:
        def consolidate(self, memories):
            # Another request can archive a source while the provider is running.
            store.archive(scope, [original.id])
            return [ExtractedFact(content="A derived claim")]

    service = MemoryService(store, Settings(_env_file=None), llm=RacingProvider())
    with pytest.raises(InvalidSourceError):
        service.consolidate(ConsolidateRequest(scope=scope, source_ids=[original.id]))
    assert store.stats(scope)["total"] == 1


def test_maintenance_does_not_archive_memory_rehearsed_after_snapshot(store, monkeypatch):
    scope = Scope(user_id="alice")
    original = store.put(
        MemoryInput(scope=scope, content="A rehearsable memory", kind="episodic"),
        now=datetime.now(UTC) - timedelta(days=365),
    )
    archive = store.archive

    def rehearse_then_archive(scope, ids, *args, **kwargs):
        store.rehearse(scope, original.id)
        return archive(scope, ids, *args, **kwargs)

    monkeypatch.setattr(store, "archive", rehearse_then_archive)
    service = MemoryService(store, Settings(_env_file=None))
    response = service.maintain(MaintenanceRequest(scope=scope, dry_run=False))
    assert response["archived_ids"] == []
    assert store.get(scope, original.id) is not None


@pytest.mark.parametrize("change", ["delete", "supersede"])
def test_recall_discards_memory_invalidated_during_embedding_request(store, change):
    scope = Scope(user_id="alice")
    original = store.put(
        MemoryInput(scope=scope, content="User lives in Paris", kind="semantic", fact_key="city")
    )
    summary = store.put(
        MemoryInput(
            scope=scope,
            content="Paris is the user's home city",
            kind="semantic",
            source_ids=[original.id],
        )
    )

    class RacingEmbeddings:
        model_key = "test-embedding-race"
        changed = False

        def embed(self, texts):
            if not self.changed:
                self.changed = True
                if change == "delete":
                    store.delete(scope, original.id)
                else:
                    store.put(
                        MemoryInput(
                            scope=scope,
                            content="User lives in Berlin",
                            kind="semantic",
                            fact_key="city",
                        )
                    )
            return [[1.0, 0.0] for _ in texts]

    service = MemoryService(store, Settings(_env_file=None), embeddings=RacingEmbeddings())
    response = service.recall(RecallRequest(scope=scope, query="Paris"))
    returned_ids = {hit.memory.id for hit in response.hits}
    assert original.id not in returned_ids
    assert summary.id not in returned_ids
