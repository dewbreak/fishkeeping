"""Bounded video frame evidence. No video or image payload is a trusted instruction."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from fishkeeping.models import Identifier, Model, Scope


class VideoRequest(Model):
    scope: Scope
    video_base64: str = Field(min_length=1, max_length=33_554_432, repr=False)
    question: str = Field(min_length=1, max_length=1000)
    max_frames: int = Field(default=6, ge=2, le=12)
    strategy: Literal["adaptive", "uniform"] = "adaptive"
    max_chars: int = Field(default=6000, ge=2000, le=10000)
    start_seconds: float = Field(default=0, ge=0, lt=180)
    end_seconds: float | None = Field(default=None, gt=0, le=180)

    @model_validator(mode="after")
    def valid_window(self) -> VideoRequest:
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class VideoEventRequest(VideoRequest):
    event_id: Identifier
    session_id: Identifier


class VideoFrame(Model):
    id: str
    frame_index: int = Field(ge=0)
    timestamp_seconds: float = Field(ge=0)
    image_sha256: str
    image_base64: str = Field(min_length=1, max_length=5_592_408, repr=False)
    mime_type: Literal["image/jpeg"] = "image/jpeg"
    width: int = Field(ge=1, le=1280)
    height: int = Field(ge=1, le=1280)
    selection_reason: str


class VideoEvidence(Model):
    video_sha256: str
    question: str
    method: str
    method_version: str
    strategy: Literal["adaptive", "uniform"]
    duration_seconds: float = Field(ge=0)
    container_duration_seconds: float = Field(gt=0)
    decoded_frames: int = Field(ge=2, le=6000)
    candidate_count: int = Field(ge=2, le=90)
    window_candidate_count: int = Field(ge=2, le=90)
    analysis_start_seconds: float = Field(ge=0)
    analysis_end_seconds: float = Field(gt=0)
    duplicate_candidates: int = Field(ge=0, le=89)
    sample_interval_seconds: float = Field(gt=0)
    selected_frames: list[VideoFrame] = Field(min_length=2, max_length=12)
    max_gap_seconds: float = Field(ge=0)
    observed_transitions: int = Field(ge=0)
    represented_transitions: int = Field(ge=0)
    requires_review: bool
    warnings: list[str]
    packed: str
    chars_used: int
