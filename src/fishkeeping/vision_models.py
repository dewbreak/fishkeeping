"""Bounded image evidence schemas; image contents never become instructions."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from fishkeeping.models import EventResponse, Identifier, Model, Scope


class Box(Model):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(gt=0)
    y2: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> Box:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("region must have positive area")
        return self


class ImageRequest(Model):
    scope: Scope
    image_base64: str = Field(min_length=1, max_length=5_592_408, repr=False)
    question: str = Field(min_length=1, max_length=1000)
    max_chars: int = Field(default=3000, ge=800, le=10000)
    task: Literal["document", "scene"] = "document"
    region: Box | None = None


class ImageEventRequest(ImageRequest):
    event_id: Identifier
    session_id: Identifier


class TextRegion(Model):
    id: str
    box: Box
    text: str = Field(min_length=1, max_length=2000)
    confidence: float = Field(ge=0, le=1)


class ImageEvidence(Model):
    image_sha256: str
    width: int
    height: int
    method: str
    method_version: str
    question: str
    regions: list[TextRegion]
    total_regions: int
    omitted_regions: int
    requires_review: bool
    warnings: list[str]
    packed: str
    chars_used: int
    cache_hit: bool = False


class ImageEventResponse(Model):
    evidence: ImageEvidence
    event: EventResponse


class ImageAnswer(Model):
    answer: str = Field(max_length=4000)
    uncertain: bool
    evidence_ids: list[str] = Field(default_factory=list, max_length=256)
    usage: dict[str, int] = Field(default_factory=dict)
    model: str
