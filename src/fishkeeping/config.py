"""Environment configuration; no network calls or DB writes at import time."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FISHKEEPING_", env_file=".env", extra="ignore", env_file_encoding="utf-8"
    )

    db_path: str = "data/fishkeeping.sqlite3"
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    api_keys: dict[str, str] = Field(default_factory=dict, repr=False)
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    deepseek_thinking: Literal["disabled", "enabled", "omit"] = "disabled"
    deepseek_max_tokens: int = Field(default=2048, ge=128, le=16384)
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    request_timeout: float = Field(default=45, ge=1, le=300)
    candidate_limit: int = Field(default=5000, ge=100, le=100000)
    embedding_backend: Literal["remote", "local"] = "remote"
    local_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    local_embedding_cache: str = ".cache/embeddings"
    local_embedding_threads: int = Field(default=2, ge=1, le=32)
    local_embedding_min_similarity: float = Field(default=0.50, ge=0, lt=1)
    embedding_min_similarity: float = Field(default=0.35, ge=0, lt=1)
    worker_daily_call_limit: int = Field(default=20, ge=0, le=10000)
    worker_max_attempts: int = Field(default=3, ge=1, le=10)

    @field_validator("deepseek_api_key", "embedding_api_key", mode="before")
    @classmethod
    def normalize_optional_key(cls, value: str | SecretStr | None) -> str | None:
        if value is None:
            return None
        raw = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw, str):
            raise ValueError("API key must be a string")
        raw = raw.strip()
        if not raw or raw.lower() in {
            "your-api-key",
            "your_api_key",
            "your-provider-key",
            "replace-with-your-provider-key",
            "replace-with-your-embedding-key",
            "你的真实api密钥",
        }:
            return None
        return raw

    @field_validator("deepseek_base_url", "embedding_base_url")
    @classmethod
    def check_url(cls, value: str | None) -> str | None:
        if value is not None:
            from urllib.parse import urlparse

            parsed = urlparse(value)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise ValueError("provider base URL must be an absolute HTTP(S) URL")
            if parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError(
                    "provider base URL must not contain credentials, query or fragment"
                )
            return value.rstrip("/")
        return value

    @field_validator("api_keys")
    @classmethod
    def check_keys(cls, value: dict[str, str]) -> dict[str, str]:
        from fishkeeping.models import Scope

        for key, namespace in value.items():
            if len(key.strip()) < 16:
                raise ValueError("API keys must contain at least 16 characters")
            Scope(namespace=namespace, user_id="validation")
        return value
