from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping.config import Settings
from fishkeeping.models import IndexRequest, MemoryInput, RecallRequest, Scope
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

SCOPE = Scope(user_id="index-regressions")


class Embeddings:
    model_key = "test-model"

    def __init__(self, before_query=None):
        self.before_query = before_query
        self.document_calls = []
        self.query_calls = []

    def embed(self, texts):
        self.document_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query):
        self.query_calls.append(query)
        if self.before_query is not None:
            self.before_query()
        return [1.0, 0.0]


@pytest.fixture
def service():
    store = Store(":memory:")
    instance = MemoryService(
        store, Settings(_env_file=None, candidate_limit=100), embeddings=Embeddings()
    )
    yield instance
    store.close()


def remember(service, content, **kwargs):
    return service.store.put(MemoryInput(scope=SCOPE, content=content, **kwargs))


def test_recall_semantic_candidate_union_recovers_old_memory_without_embedding_documents(service):
    old = service.store.put(
        MemoryInput(scope=SCOPE, content="用户一直喜欢茉莉花茶"),
        now=datetime.now(UTC) - timedelta(days=60),
    )
    newer, _ = service.store.put_many(
        [MemoryInput(scope=SCOPE, content=f"unrelated recent record {n}") for n in range(150)]
    )
    service.store.set_embedding(old.id, "test-model", [1, 0])
    for memory in newer:
        service.store.set_embedding(memory.id, "test-model", [0, 1])
    result = service.recall(RecallRequest(scope=SCOPE, query="preferred beverage"))
    assert [hit.memory.id for hit in result.hits] == [old.id]
    assert result.retrieval_mode == "hybrid"
    assert service.embeddings.document_calls == []
    assert service.embeddings.query_calls == ["preferred beverage"]


@pytest.mark.parametrize("mutation", ["delete", "archive", "expire"])
def test_recall_query_embedding_cannot_return_memory_that_became_inactive(service, mutation):
    memory = remember(service, "tea preference")
    service.store.set_embedding(memory.id, "test-model", [1, 0])

    def mutate():
        if mutation == "delete":
            service.store.delete(SCOPE, memory.id)
        elif mutation == "archive":
            service.store.archive(SCOPE, [memory.id])
        else:
            memory.expires_at = datetime.now(UTC) - timedelta(seconds=1)
            with service.store._transaction():
                service.store._save(memory)

    service.embeddings.before_query = mutate
    result = service.recall(RecallRequest(scope=SCOPE, query="tea"))
    assert result.hits == []
    assert service.embeddings.document_calls == []


def test_recall_refreshes_provenance_added_while_query_embedding_runs(service):
    first = remember(service, "first observation")
    second = remember(service, "second observation")
    memory = remember(service, "tea preference", source_ids=[first.id])
    service.store.set_embedding(memory.id, "test-model", [1, 0])
    service.embeddings.before_query = lambda: remember(
        service, "tea preference", source_ids=[second.id]
    )
    result = service.recall(RecallRequest(scope=SCOPE, query="tea"))
    assert len(result.hits) == 1
    assert result.hits[0].memory.source_count == 2
    assert result.hits[0].memory.source_ids == [first.id, second.id]


def test_index_does_not_count_or_cache_record_archived_after_refresh(service, monkeypatch):
    memory = remember(service, "encode then archive")
    original = service.store.refresh_active

    def refresh_then_archive(*args, **kwargs):
        result = original(*args, **kwargs)
        service.store.archive(SCOPE, [memory.id])
        return result

    monkeypatch.setattr(service.store, "refresh_active", refresh_then_archive)
    result = service.index(IndexRequest(scope=SCOPE))
    assert result.processed == 0
    assert service.store.get_embeddings([memory.id], "test-model") == {}


def test_rebuild_recovers_corrupt_cache_and_retains_other_scope_and_model(service):
    memory = remember(service, "tea preference")
    foreign_scope = Scope(user_id="another-user")
    foreign = service.store.put(MemoryInput(scope=foreign_scope, content="private preference"))
    service.store.set_embedding(memory.id, "test-model", [1, 0, 0])
    service.store.set_embedding(memory.id, "other-model", [1, 0, 0])
    service.store.set_embedding(foreign.id, "test-model", [1, 0, 0])
    before = service.recall(RecallRequest(scope=SCOPE, query="tea"))
    assert before.retrieval_mode == "lexical"
    assert before.warnings
    assert [hit.memory.id for hit in before.hits] == [memory.id]
    indexed = service.index(IndexRequest(scope=SCOPE, rebuild=True))
    assert indexed.complete and indexed.processed == 1
    after = service.recall(RecallRequest(scope=SCOPE, query="tea"))
    assert after.retrieval_mode == "hybrid"
    assert service.store.get_embeddings([memory.id], "test-model") == {memory.id: [1, 0]}
    assert service.store.get_embeddings([memory.id], "other-model") == {memory.id: [1, 0, 0]}
    assert service.store.get_embeddings([foreign.id], "test-model") == {foreign.id: [1, 0, 0]}
