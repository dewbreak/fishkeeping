"""Environment configuration; no network calls or DB writes at import time."""

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
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None
    embedding_model: str | None = None
    request_timeout: float = Field(default=45, ge=1, le=300)
    candidate_limit: int = Field(default=5000, ge=100, le=100000)
    embedding_max_new_per_recall: int = Field(default=64, ge=1, le=1024)

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
