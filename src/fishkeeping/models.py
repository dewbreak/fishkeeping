"""Shared wire and storage schemas. All public inputs reject unknown fields."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Identifier = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[\w.:-]+$")]
Kind = Literal["working", "episodic", "semantic", "procedural"]
Status = Literal["active", "archived", "superseded"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class Scope(Model):
    namespace: Identifier = "default"
    user_id: Identifier
    agent_id: Identifier = "default"

    def key(self) -> tuple[str, str, str]:
        return self.namespace, self.user_id, self.agent_id


class MemoryInput(Model):
    scope: Scope
    content: str = Field(min_length=1, max_length=16000)
    kind: Kind = "episodic"
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    session_id: Identifier | None = None
    fact_key: Identifier | None = None
    source_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    metadata: dict = Field(default_factory=dict)
    ttl_seconds: int | None = Field(default=None, ge=1, le=315360000)
    pinned: bool = False

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict) -> dict:
        try:
            size = len(json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        except (TypeError, ValueError) as exc:
            raise ValueError("metadata must be finite JSON data") from exc
        if size > 16000:
            raise ValueError("metadata must not exceed 16000 UTF-8 bytes")
        return value

    @model_validator(mode="after")
    def validate_kind(self) -> MemoryInput:
        if self.kind == "working":
            if not self.session_id:
                raise ValueError("working memory requires session_id")
            if self.ttl_seconds is None:
                self.ttl_seconds = 3600
        if self.fact_key is not None and self.kind != "semantic":
            raise ValueError("fact_key is only supported for semantic memory")
        self.source_ids = list(dict.fromkeys(self.source_ids))
        return self


class Memory(MemoryInput):
    id: str
    status: Status = "active"
    created_at: datetime
    updated_at: datetime
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    access_count: int = 0
    superseded_by: str | None = None
    source_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def count_preview_sources(self) -> Memory:
        self.source_count = max(self.source_count, len(self.source_ids))
        return self


class Message(Model):
    role: Literal["user", "assistant", "system", "tool"]
    content: str = Field(min_length=1, max_length=12000)


class IngestRequest(Model):
    scope: Scope
    event_id: Identifier
    session_id: Identifier
    messages: list[Message] = Field(min_length=1, max_length=100)
    extract: bool = False

    @model_validator(mode="after")
    def check_size(self) -> IngestRequest:
        if sum(len(m.content) for m in self.messages) > 12000:
            raise ValueError("event content must not exceed 12000 characters")
        return self


class IngestResponse(Model):
    memories: list[Memory]
    replayed: bool


class EventRequest(IngestRequest):
    extract: bool = True


class JobInfo(Model):
    id: str
    event_id: str
    state: Literal["pending", "processing", "succeeded", "failed", "cancelled"]
    attempts: int
    max_attempts: int
    next_attempt_at: datetime | None = None
    last_error_code: str | None = None
    created_at: datetime
    updated_at: datetime
    result_ids: list[str] = Field(default_factory=list)


class EventResponse(Model):
    event_id: str
    scope: Scope
    episode: Memory
    job: JobInfo | None = None
    replayed: bool = False


class SourcePage(Model):
    memory_id: str
    items: list[str]
    total: int
    limit: int
    offset: int
    has_more: bool


class IndexRequest(Model):
    scope: Scope
    limit: int = Field(default=64, ge=1, le=1024)
    rebuild: bool = False


class IndexResponse(Model):
    processed: int
    total: int
    indexed: int
    complete: bool
    model_key: str


class RecallRequest(Model):
    scope: Scope
    query: str = Field(min_length=1, max_length=4000)
    session_id: Identifier | None = None
    kinds: list[Kind] | None = Field(default=None, min_length=1, max_length=4)
    limit: int = Field(default=8, ge=1, le=100)


class RecallHit(Model):
    memory: Memory
    score: float
    relevance: float
    retention: float


class RecallResponse(Model):
    hits: list[RecallHit]
    retrieval_mode: str
    warnings: list[str] = Field(default_factory=list)


class ContextRequest(RecallRequest):
    max_chars: int = Field(default=6000, ge=256, le=32000)


class ContextResponse(RecallResponse):
    context: str
    included_ids: list[str]
    chars_used: int


class ConsolidateRequest(Model):
    scope: Scope
    source_ids: list[Identifier] = Field(min_length=1, max_length=20)


class MaintenanceRequest(Model):
    scope: Scope
    dry_run: bool = True
    retention_threshold: float = Field(default=0.1, ge=0, le=1)


class ExtractedFact(Model):
    content: str = Field(min_length=1, max_length=4000)
    kind: Literal["semantic", "procedural"] = "semantic"
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=0.7, ge=0, le=1)


class Extraction(Model):
    memories: list[ExtractedFact] = Field(default_factory=list, max_length=20)


class ConsolidateResponse(Model):
    memories: list[Memory]
    method: Literal["extractive", "deepseek"]


class MaintenanceResponse(Model):
    dry_run: bool
    candidate_ids: list[str]
    archived_ids: list[str]


class DeleteResponse(Model):
    deleted_ids: list[str]


class StatsResponse(Model):
    total: int
    by_kind: dict[str, int]
    by_status: dict[str, int]
    expired: int
