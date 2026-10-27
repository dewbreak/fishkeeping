from types import SimpleNamespace

import pytest

from fishkeeping.config import Settings
from fishkeeping.local_embeddings import DIMENSIONS, LocalEmbeddingProvider
from fishkeeping.models import IndexRequest, MemoryInput, Scope
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

SCOPE = Scope(user_id="index-batch-test")


class CharacterTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(ids=range(len(text)), offsets=[(i, i + 1) for i in range(len(text))])


class RecordingModel:
    """Exercise the real adapter's chunking without a model download or paid API."""

    tokenizer = CharacterTokenizer()

    def __init__(self):
        self.chunks = []
        self.query_calls = 0

    def passage_embed(self, texts, **kwargs):
        self.chunks.append(list(texts))
        return iter([[1.0] + [0.0] * (DIMENSIONS - 1) for _ in texts])

    def query_embed(self, texts, **kwargs):
        self.query_calls += 1
        pytest.fail("Indexing must never embed a query")


class RecordingLocalProvider(LocalEmbeddingProvider):
    def __init__(self, settings, model):
        super().__init__(settings, model)
        self.batches = []

    def embed(self, texts):
        self.batches.append(list(texts))
        return super().embed(texts)


@pytest.fixture
def service(tmp_path):
    settings = Settings(_env_file=None, db_path=str(tmp_path / "index.sqlite3"))
    store = Store(settings.db_path)
    provider = RecordingLocalProvider(settings, RecordingModel())
    service = MemoryService(store, settings, embeddings=provider)
    yield service
    provider.close()
    store.close()


def add_long_records(service, count):
    records = []
    for index in range(count):
        prefix, tail = f"record-{index:02d}", f"唯一尾部-{index:02d}"
        content = prefix + "记" * (16_000 - len(prefix) - len(tail)) + tail
        records.append(service.store.put(MemoryInput(scope=SCOPE, content=content)))
    return records


def test_32_maximum_length_memories_are_indexed_in_bounded_complete_batches(service):
    records = add_long_records(service, 32)
    result = service.index(IndexRequest(scope=SCOPE, limit=32))

    assert result.complete
    assert result.processed == result.indexed == result.total == 32
    provider = service.embeddings
    assert [len(batch) for batch in provider.batches] == [2] * 16
    assert all(sum(map(len, batch)) <= 32_000 for batch in provider.batches)
    assert sorted(text for batch in provider.batches for text in batch) == sorted(
        record.content for record in records
    )
    # Check text reaching the model, including the tail of the last 16k record.
    for documents, chunks in zip(provider.batches, provider._model.chunks, strict=True):
        assert "".join(documents) == "".join(chunks)
    assert any(chunk.endswith("唯一尾部-31") for call in provider._model.chunks for chunk in call)
    assert provider._model.query_calls == 0
    assert len(service.store.get_embeddings([r.id for r in records], provider.model_key)) == 32


def test_character_limited_batches_respect_deadline_and_resume_without_repeating(
    service, monkeypatch
):
    records = add_long_records(service, 17)
    values = iter([1_000.0, 1_001.0, 1_046.0])
    monkeypatch.setattr(
        "fishkeeping.service.time", SimpleNamespace(monotonic=lambda: next(values, 1_046.0))
    )
    result = service.index(IndexRequest(scope=SCOPE, limit=17))
    assert not result.complete
    assert result.processed == result.indexed == 2
    assert result.total == 17

    monkeypatch.setattr("fishkeeping.service.time", SimpleNamespace(monotonic=lambda: 2_000.0))
    resumed = service.index(IndexRequest(scope=SCOPE, limit=17))
    assert resumed.complete
    assert resumed.processed == 15
    assert resumed.indexed == 17
    assert [len(batch) for batch in service.embeddings.batches] == [2] * 8 + [1]
    supplied = [text for batch in service.embeddings.batches for text in batch]
    assert len(supplied) == len(set(supplied)) == len(records)
    assert service.embeddings._model.query_calls == 0
