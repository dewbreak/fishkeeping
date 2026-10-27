"""Optional, pinned CPU embeddings; importing this module never loads a model."""

from __future__ import annotations

import math
import os
from importlib.metadata import PackageNotFoundError, version
from itertools import islice
from numbers import Real
from pathlib import Path
from threading import Lock

from fishkeeping.config import Settings
from fishkeeping.providers import ProviderError

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REPOSITORY = "Qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q"
MODEL_REVISION = "faf4aa4225822f3bc6376869cb1164e8e3feedd0"
MODEL_FILES = (
    "model_optimized.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
DIMENSIONS = 384
CHUNK_TOKENS = 126  # The model's 128-token limit includes CLS and SEP.
MAX_TEXT_CHARACTERS = 16_000
MAX_BATCH_CHARACTERS = 262_144
MAX_CHUNKS = 2_048


def _invalid_output() -> ProviderError:
    return ProviderError("Invalid local embedding output", code="invalid_response")


def _vectors(output, count: int) -> list[list[float]]:
    """Bound iteration, validate ONNX/numpy scalars, and normalize each chunk."""
    rows = list(islice(output, count + 1))
    if len(rows) != count:
        raise _invalid_output()
    result = []
    for row in rows:
        values = list(islice(row, DIMENSIONS + 1))
        if len(values) != DIMENSIONS or any(
            isinstance(value, bool) or not isinstance(value, Real) for value in values
        ):
            raise _invalid_output()
        vector = [float(value) for value in values]
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise _invalid_output()
        result.append([value / norm for value in vector])
    return result


class LocalEmbeddingProvider:
    """One lazily loaded session, serialized inference, no billable API calls.

    ``model`` is an optional caller-owned test double exposing passage_embed,
    query_embed and an untruncated tokenizer. Production uses a separately loaded
    tokenizer so inspecting full-text offsets cannot alter FastEmbed's tokenizer.
    """

    def __init__(self, settings: Settings, model=None):
        if settings.local_embedding_model != MODEL_NAME:
            raise ProviderError("Unsupported local embedding model", code="not_configured")
        self._cache = settings.local_embedding_cache
        self._threads = settings.local_embedding_threads
        self.similarity_threshold = settings.local_embedding_min_similarity
        self._model = model
        self._tokenizer = getattr(model, "tokenizer", None)
        self._lock = Lock()
        self._closed = False
        try:
            library_version = version("fastembed")
        except PackageNotFoundError:
            library_version = "not-installed"
        self.model_key = (
            f"local:fastembed-{library_version}|{MODEL_NAME}@{MODEL_REVISION}"
            "|fp16-mean-l2-384|raw-query-passage|token126-weighted-chunks-v1"
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        # Hugging Face/Xet read these defaults at import time. Keep auxiliary
        # caches alongside the configured model cache, respecting explicit envs.
        cache = Path(self._cache).resolve()
        os.environ.setdefault("HF_HOME", str(cache / "hub"))
        os.environ.setdefault("HF_XET_CACHE", str(cache / "xet"))
        try:
            from fastembed import TextEmbedding
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError
            from tokenizers import Tokenizer
        except ImportError:
            raise ProviderError(
                "Local embeddings require: pip install '.[local]'", code="not_configured"
            ) from None
        except Exception:
            raise ProviderError(
                "Local embedding runtime could not be loaded", code="model_unavailable"
            ) from None

        try:
            download = {
                "repo_id": MODEL_REPOSITORY,
                "revision": MODEL_REVISION,
                "cache_dir": self._cache,
                "allow_patterns": list(MODEL_FILES),
                "max_workers": 2,
            }
            try:
                directory = snapshot_download(**download, local_files_only=True)
                if not all((Path(directory) / name).is_file() for name in MODEL_FILES):
                    raise LocalEntryNotFoundError("Incomplete snapshot")
            except LocalEntryNotFoundError:
                directory = snapshot_download(**download)
            tokenizer = Tokenizer.from_file(str(Path(directory) / "tokenizer.json"))
            tokenizer.no_truncation()
            tokenizer.no_padding()
            model = TextEmbedding(
                model_name=MODEL_NAME,
                cache_dir=self._cache,
                threads=self._threads,
                providers=["CPUExecutionProvider"],
                specific_model_path=directory,
                local_files_only=True,
            )
        except Exception:
            raise ProviderError(
                "Local embedding model could not be loaded", code="model_unavailable"
            ) from None
        self._model, self._tokenizer = model, tokenizer

    def _chunks(self, text: str) -> list[tuple[str, int]]:
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        offsets = encoding.offsets
        if len(encoding.ids) <= CHUNK_TOKENS:
            return [(text, max(1, len(encoding.ids)))]
        chunks = []
        start, index = 0, 0
        while start < len(text):
            # Several tokens may refer to one Unicode span. Never split that span.
            while index < len(offsets) and offsets[index][1] <= start:
                index += 1
            last = min(index + CHUNK_TOKENS - 1, len(offsets) - 1)
            end = len(text) if last == len(offsets) - 1 else offsets[last][1]
            if end <= start:
                raise _invalid_output()
            while True:
                part = text[start:end]
                size = len(self._tokenizer.encode(part, add_special_tokens=False).ids)
                if size <= CHUNK_TOKENS:
                    break
                # Retokenizing a substring can change word boundaries. Recheck its
                # actual token budget, including byte-fallback / overlapping offsets.
                if end - start <= 1:
                    raise _invalid_output()
                end = start + (end - start) // 2
            chunks.append((part, max(1, size)))
            if len(chunks) > MAX_CHUNKS:
                raise ProviderError("Local embedding batch is too large", code="invalid_request")
            start = end
        return chunks

    def _embed(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if not texts:
            return []
        if (
            len(texts) > 1_024
            or any(not isinstance(text, str) or len(text) > MAX_TEXT_CHARACTERS for text in texts)
            or sum(len(text) for text in texts) > MAX_BATCH_CHARACTERS
        ):
            raise ProviderError("Local embedding batch is too large", code="invalid_request")
        with self._lock:
            if self._closed:
                raise ProviderError("Local embedding provider is closed", code="provider_closed")
            self._load()
            try:
                groups = []
                chunk_count = 0
                for text in texts:
                    group = self._chunks(text)
                    chunk_count += len(group)
                    if chunk_count > MAX_CHUNKS:
                        raise ProviderError(
                            "Local embedding batch is too large", code="invalid_request"
                        )
                    groups.append(group)
                chunks = [part for group in groups for part, _ in group]
                # MiniLM is symmetric: no E5-style prefixes. Keep routes explicit
                # so changing to an asymmetric model requires changing the contract.
                encode = self._model.query_embed if query else self._model.passage_embed
                vectors = _vectors(encode(chunks, batch_size=32, parallel=None), chunk_count)
                result, offset = [], 0
                for group in groups:
                    group_vectors = vectors[offset : offset + len(group)]
                    if len(group) == 1:
                        result.append(group_vectors[0])
                    else:
                        total_weight = sum(weight for _, weight in group)
                        mean = [
                            math.fsum(
                                vector[column] * weight / total_weight
                                for vector, (_, weight) in zip(group_vectors, group, strict=True)
                            )
                            for column in range(DIMENSIONS)
                        ]
                        result.extend(_vectors([mean], 1))
                    offset += len(group)
                return result
            except ProviderError:
                raise
            except Exception:
                raise ProviderError(
                    "Local embedding inference failed", code="invalid_response"
                ) from None

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, query=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], query=True)[0]

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._model = None
            self._tokenizer = None
