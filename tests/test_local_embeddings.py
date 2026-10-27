import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from fishkeeping.local_embeddings import (
    DIMENSIONS,
    MODEL_FILES,
    MODEL_NAME,
    MODEL_REVISION,
    LocalEmbeddingProvider,
)
from fishkeeping.providers import ProviderError


def settings(**overrides):
    return SimpleNamespace(
        **{
            "local_embedding_model": MODEL_NAME,
            "local_embedding_cache": ".cache/embeddings",
            "local_embedding_threads": 2,
            "local_embedding_min_similarity": 0.50,
            **overrides,
        }
    )


class CharacterTokenizer:
    """One token per codepoint, including whitespace: boundaries are observable."""

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return SimpleNamespace(
            ids=list(range(len(text))), offsets=[(i, i + 1) for i in range(len(text))]
        )

    def no_truncation(self):
        pass

    def no_padding(self):
        pass


def vector(x=3, y=4):
    return [x, y] + [0] * (DIMENSIONS - 2)


class FakeModel:
    def __init__(self, make_vector=None):
        self.tokenizer = CharacterTokenizer()
        self.calls = []
        self.make_vector = make_vector or (lambda text: vector())

    def _encode(self, texts, route, kwargs):
        self.calls.append((route, list(texts), kwargs))
        return (self.make_vector(text) for text in texts)

    def passage_embed(self, texts, **kwargs):
        return self._encode(texts, "passage", kwargs)

    def query_embed(self, texts, **kwargs):
        return self._encode(texts, "query", kwargs)


def install_fake_packages(monkeypatch, tmp_path, *, construct=None, download=None):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    for name in MODEL_FILES:
        (directory / name).write_text("test fixture", encoding="utf-8")
    calls = {"construct": [], "download": []}
    model = FakeModel()

    class MissingSnapshot(Exception):
        pass

    def snapshot_download(**kwargs):
        calls["download"].append(kwargs)
        if download:
            return download(kwargs, directory, MissingSnapshot)
        return str(directory)

    def text_embedding(**kwargs):
        calls["construct"].append(kwargs)
        return construct(kwargs) if construct else model

    monkeypatch.setitem(sys.modules, "fastembed", SimpleNamespace(TextEmbedding=text_embedding))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.errors",
        SimpleNamespace(LocalEntryNotFoundError=MissingSnapshot),
    )
    monkeypatch.setitem(
        sys.modules,
        "tokenizers",
        SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda path: CharacterTokenizer())),
    )
    return calls, model


def test_short_chinese_and_english_use_distinct_routes_without_e5_prefixes():
    model = FakeModel()
    provider = LocalEmbeddingProvider(settings(), model)
    documents = ["我喜欢乌龙茶。", "I prefer a silent office."]
    assert provider.embed(documents) == [vector(0.6, 0.8), vector(0.6, 0.8)]
    assert provider.embed_query("oolong beverage") == vector(0.6, 0.8)
    assert [(route, texts) for route, texts, _ in model.calls] == [
        ("passage", documents),
        ("query", ["oolong beverage"]),
    ]
    assert MODEL_NAME in provider.model_key and MODEL_REVISION in provider.model_key
    assert "token126-weighted-chunks-v1" in provider.model_key


def test_long_content_keeps_tail_and_weights_chunks_by_actual_tokens():
    model = FakeModel(lambda text: vector(1, 0) if "尾" not in text else vector(0, 1))
    provider = LocalEmbeddingProvider(settings(), model)
    content = "头" * 126 + "尾" * 63
    result = provider.embed([content])[0]
    assert model.calls[0][1] == ["头" * 126, "尾" * 63]
    assert result[:2] == pytest.approx([2 / math.sqrt(5), 1 / math.sqrt(5)])
    assert math.hypot(*result) == pytest.approx(1)
    provider.embed_query(content)
    assert model.calls[1][0] == "query"
    assert "".join(model.calls[1][1]) == content


def test_overlapping_unicode_offsets_and_retokenization_preserve_all_content():
    class OverlappingTokenizer(CharacterTokenizer):
        def encode(self, text, *, add_special_tokens=False):
            # A substring boundary can need extra tokens. Repeated spans model
            # byte fallback without pretending tokens map one-to-one to characters.
            offsets = [(i, i + 1) for i in range(len(text)) for _ in range(3)]
            if not text.startswith("首"):
                offsets = offsets * 2
            return SimpleNamespace(ids=list(range(len(offsets))), offsets=offsets)

    model = FakeModel()
    model.tokenizer = OverlappingTokenizer()
    provider = LocalEmbeddingProvider(settings(), model)
    content = "首" + "𠮷中文🙂" * 60 + "尾"
    provider.embed([content])
    chunks = model.calls[0][1]
    assert "".join(chunks) == content
    assert all(len(model.tokenizer.encode(chunk).ids) <= 126 for chunk in chunks)


def test_init_and_empty_batch_do_not_import_or_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    provider = LocalEmbeddingProvider(settings())
    assert provider.embed([]) == []
    with pytest.raises(ProviderError, match=r"pip install '\.\[local\]'") as error:
        provider.embed_query("hello")
    assert error.value.code == "not_configured"


def test_pinned_snapshot_is_offline_first_and_cpu_session_reused(monkeypatch, tmp_path):
    calls, _ = install_fake_packages(monkeypatch, tmp_path)
    provider = LocalEmbeddingProvider(settings())
    assert not calls["download"]
    provider.embed(["hello"])
    provider.embed_query("你好")
    assert len(calls["download"]) == len(calls["construct"]) == 1
    assert calls["download"][0]["revision"] == MODEL_REVISION
    assert calls["download"][0]["local_files_only"] is True
    assert set(calls["download"][0]["allow_patterns"]) == set(MODEL_FILES)
    assert calls["construct"][0]["providers"] == ["CPUExecutionProvider"]
    assert calls["construct"][0]["threads"] == 2
    assert calls["construct"][0]["local_files_only"] is True


def test_auxiliary_cache_defaults_stay_under_configured_cache(monkeypatch, tmp_path):
    install_fake_packages(monkeypatch, tmp_path)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    configured = tmp_path / "embedding-cache"
    provider = LocalEmbeddingProvider(settings(local_embedding_cache=str(configured)))
    assert "HF_HOME" not in os.environ
    assert "HF_XET_CACHE" not in os.environ
    provider.embed_query("hello")
    assert os.environ["HF_HOME"] == str(configured.resolve() / "hub")
    assert os.environ["HF_XET_CACHE"] == str(configured.resolve() / "xet")


def test_auxiliary_cache_respects_explicit_environment(monkeypatch, tmp_path):
    install_fake_packages(monkeypatch, tmp_path)
    monkeypatch.setenv("HF_HOME", "explicit-hf-home")
    monkeypatch.setenv("HF_XET_CACHE", "explicit-xet-cache")
    LocalEmbeddingProvider(settings()).embed_query("hello")
    assert os.environ["HF_HOME"] == "explicit-hf-home"
    assert os.environ["HF_XET_CACHE"] == "explicit-xet-cache"


def test_missing_snapshot_downloads_pinned_files_only(monkeypatch, tmp_path):
    def download(kwargs, directory, error):
        if kwargs.get("local_files_only"):
            raise error("not cached")
        return str(directory)

    calls, _ = install_fake_packages(monkeypatch, tmp_path, download=download)
    LocalEmbeddingProvider(settings()).embed_query("hello")
    assert len(calls["download"]) == 2
    assert all(call["revision"] == MODEL_REVISION for call in calls["download"])
    assert calls["download"][1].get("local_files_only") is None


def test_concurrent_first_requests_load_once_and_serialize_generator_consumption(
    monkeypatch, tmp_path
):
    active = 0
    peak = 0

    class SlowModel(FakeModel):
        def query_embed(self, texts, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                time.sleep(0.01)
                yield vector()
            finally:
                active -= 1

    calls, _ = install_fake_packages(monkeypatch, tmp_path, construct=lambda _: SlowModel())
    provider = LocalEmbeddingProvider(settings())
    with ThreadPoolExecutor(max_workers=6) as executor:
        outputs = list(executor.map(provider.embed_query, ["test"] * 6))
    assert len(outputs) == 6
    assert peak == 1
    assert len(calls["construct"]) == len(calls["download"]) == 1


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [1],
        [1] * 385,
        vector(True, 1),
        vector("1", 1),
        vector(float("nan"), 1),
        vector(float("inf"), 1),
        vector(0, 0),
        None,
    ],
)
def test_bad_vectors_never_reach_storage(bad):
    provider = LocalEmbeddingProvider(settings(), FakeModel(lambda _: bad))
    with pytest.raises(ProviderError) as error:
        provider.embed(["private content"])
    assert error.value.code == "invalid_response"
    assert "private content" not in str(error.value)


@pytest.mark.parametrize("count", [0, 2])
def test_wrong_output_count_is_rejected(count):
    model = FakeModel()
    model.query_embed = lambda *args, **kwargs: iter([vector()] * count)
    with pytest.raises(ProviderError):
        LocalEmbeddingProvider(settings(), model).embed_query("query")


def test_loading_failure_is_sanitized_and_can_be_retried(monkeypatch, tmp_path):
    attempts = 0

    def construct(kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("secret local path and private content")
        return FakeModel()

    install_fake_packages(monkeypatch, tmp_path, construct=construct)
    provider = LocalEmbeddingProvider(settings())
    with pytest.raises(ProviderError) as error:
        provider.embed_query("first")
    assert str(error.value) == "Local embedding model could not be loaded"
    assert error.value.code == "model_unavailable"
    assert provider.embed_query("second") == vector(0.6, 0.8)


def test_closed_and_oversized_requests_do_not_start_inference():
    model = FakeModel()
    provider = LocalEmbeddingProvider(settings(), model)
    with pytest.raises(ProviderError) as error:
        provider.embed(["x" * 16_001])
    assert error.value.code == "invalid_request"
    assert not model.calls
    provider.close()
    provider.close()
    with pytest.raises(ProviderError) as error:
        provider.embed_query("hello")
    assert error.value.code == "provider_closed"


def test_unreviewed_model_is_rejected_without_import_or_download(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastembed", None)
    with pytest.raises(ProviderError) as error:
        LocalEmbeddingProvider(settings(local_embedding_model="unknown/custom-code"))
    assert error.value.code == "not_configured"
