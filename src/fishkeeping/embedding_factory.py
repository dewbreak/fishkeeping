"""Construct optional providers without loading a local model until it is used."""

from fishkeeping.config import Settings
from fishkeeping.providers import EmbeddingProvider


def create_embeddings(settings: Settings):
    if settings.embedding_backend == "local":
        from fishkeeping.local_embeddings import LocalEmbeddingProvider

        return LocalEmbeddingProvider(settings)
    if settings.embedding_base_url or settings.embedding_model:
        if not (settings.embedding_base_url and settings.embedding_model):
            raise ValueError("Both embedding_base_url and embedding_model must be configured")
        return EmbeddingProvider(settings)
    return None
