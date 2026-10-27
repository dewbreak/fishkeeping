"""Small synchronous HTTP SDK, independent of the caller's agent framework."""

from __future__ import annotations

import base64
from urllib.parse import quote

import httpx

from fishkeeping.models import (
    ConsolidateRequest,
    ContextRequest,
    ContextResponse,
    EventRequest,
    EventResponse,
    IndexRequest,
    IndexResponse,
    IngestRequest,
    IngestResponse,
    JobInfo,
    MaintenanceRequest,
    Memory,
    MemoryInput,
    Message,
    RecallRequest,
    RecallResponse,
    Scope,
    SourcePage,
)
from fishkeeping.video_models import VideoEventRequest, VideoEvidence, VideoRequest
from fishkeeping.vision_models import (
    ImageEventRequest,
    ImageEventResponse,
    ImageEvidence,
    ImageRequest,
)


class MemoryClient:
    """Use one scope per client. Call close() or use as a context manager."""

    def __init__(
        self,
        base_url: str,
        scope: Scope,
        api_key: str | None = None,
        timeout: float = 120,
        transport: httpx.BaseTransport | None = None,
    ):
        self.scope = scope
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"X-API-Key": api_key} if api_key else {},
            timeout=timeout,
            transport=transport,
        )

    def _request(self, method: str, path: str, **kwargs):
        response = self._http.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    def remember(self, content: str, **kwargs) -> Memory:
        body = MemoryInput(scope=self.scope, content=content, **kwargs)
        return Memory.model_validate(
            self._request("POST", "/v1/memories", json=body.model_dump(mode="json"))
        )

    def analyze_image(self, image: bytes, question: str, **kwargs) -> ImageEvidence:
        body = ImageRequest(
            scope=self.scope,
            image_base64=base64.b64encode(image).decode("ascii"),
            question=question,
            **kwargs,
        )
        return ImageEvidence.model_validate(
            self._request("POST", "/v1/images/analyze", json=body.model_dump(mode="json"))
        )

    def enqueue_image(
        self, image: bytes, question: str, event_id: str, session_id: str, **kwargs
    ) -> ImageEventResponse:
        body = ImageEventRequest(
            scope=self.scope,
            image_base64=base64.b64encode(image).decode("ascii"),
            question=question,
            event_id=event_id,
            session_id=session_id,
            **kwargs,
        )
        return ImageEventResponse.model_validate(
            self._request("POST", "/v1/images/events", json=body.model_dump(mode="json"))
        )

    def get(self, memory_id: str, include_inactive: bool = False) -> Memory:
        return Memory.model_validate(
            self._request(
                "GET",
                f"/v1/memories/{quote(memory_id, safe='')}",
                params={**self.scope.model_dump(), "include_inactive": include_inactive},
            )
        )

    def analyze_video(self, video: bytes, question: str, **kwargs) -> VideoEvidence:
        body = VideoRequest(
            scope=self.scope,
            video_base64=base64.b64encode(video).decode("ascii"),
            question=question,
            **kwargs,
        )
        return VideoEvidence.model_validate(
            self._request("POST", "/v1/videos/analyze", json=body.model_dump(mode="json"))
        )

    def enqueue_video(
        self, video: bytes, question: str, event_id: str, session_id: str, **kwargs
    ) -> EventResponse:
        body = VideoEventRequest(
            scope=self.scope,
            video_base64=base64.b64encode(video).decode("ascii"),
            question=question,
            event_id=event_id,
            session_id=session_id,
            **kwargs,
        )
        return EventResponse.model_validate(
            self._request("POST", "/v1/videos/events", json=body.model_dump(mode="json"))
        )

    def list(
        self, include_inactive: bool = False, limit: int = 100, offset: int = 0
    ) -> list[Memory]:
        result = self._request(
            "GET",
            "/v1/memories",
            params={
                **self.scope.model_dump(),
                "include_inactive": include_inactive,
                "limit": limit,
                "offset": offset,
            },
        )
        return [Memory.model_validate(item) for item in result]

    def recall(self, query: str, **kwargs) -> RecallResponse:
        body = RecallRequest(scope=self.scope, query=query, **kwargs)
        return RecallResponse.model_validate(
            self._request("POST", "/v1/recall", json=body.model_dump(mode="json"))
        )

    def context(self, query: str, **kwargs) -> ContextResponse:
        body = ContextRequest(scope=self.scope, query=query, **kwargs)
        return ContextResponse.model_validate(
            self._request("POST", "/v1/context", json=body.model_dump(mode="json"))
        )

    def ingest(
        self,
        event_id: str,
        session_id: str,
        messages: list[Message] | list[dict],
        extract: bool = False,
    ) -> IngestResponse:
        body = IngestRequest(
            scope=self.scope,
            event_id=event_id,
            session_id=session_id,
            messages=messages,
            extract=extract,
        )
        return IngestResponse.model_validate(
            self._request("POST", "/v1/ingest", json=body.model_dump(mode="json"))
        )

    def consolidate(self, source_ids: list[str]) -> list[Memory]:
        body = ConsolidateRequest(scope=self.scope, source_ids=source_ids)
        response = self._request("POST", "/v1/consolidate", json=body.model_dump(mode="json"))
        return [Memory.model_validate(item) for item in response["memories"]]

    def enqueue(
        self,
        event_id: str,
        session_id: str,
        messages: list[Message] | list[dict],
        extract: bool = True,
    ) -> EventResponse:
        body = EventRequest(
            scope=self.scope,
            event_id=event_id,
            session_id=session_id,
            messages=messages,
            extract=extract,
        )
        return EventResponse.model_validate(
            self._request("POST", "/v1/events", json=body.model_dump(mode="json"))
        )

    def event(self, event_id: str) -> EventResponse:
        return EventResponse.model_validate(
            self._request(
                "GET", f"/v1/events/{quote(event_id, safe='')}", params=self.scope.model_dump()
            )
        )

    def retry_event(self, event_id: str) -> JobInfo:
        return JobInfo.model_validate(
            self._request(
                "POST", f"/v1/events/{quote(event_id, safe='')}/retry", json=self.scope.model_dump()
            )
        )

    def sources(self, memory_id: str, limit: int = 100, offset: int = 0) -> SourcePage:
        return SourcePage.model_validate(
            self._request(
                "GET",
                f"/v1/memories/{quote(memory_id, safe='')}/sources",
                params={**self.scope.model_dump(), "limit": limit, "offset": offset},
            )
        )

    def index(self, limit: int = 64, *, rebuild: bool = False) -> IndexResponse:
        body = IndexRequest(scope=self.scope, limit=limit, rebuild=rebuild)
        return IndexResponse.model_validate(
            self._request("POST", "/v1/index", json=body.model_dump(mode="json"))
        )

    def forget(self, memory_id: str) -> list[str]:
        response = self._request(
            "DELETE",
            f"/v1/memories/{quote(memory_id, safe='')}",
            params=self.scope.model_dump(),
        )
        return response["deleted_ids"]

    def rehearse(self, memory_id: str) -> Memory:
        return Memory.model_validate(
            self._request(
                "POST",
                f"/v1/memories/{quote(memory_id, safe='')}/rehearse",
                json=self.scope.model_dump(),
            )
        )

    def maintain(self, dry_run: bool = True, retention_threshold: float = 0.1) -> dict:
        body = MaintenanceRequest(
            scope=self.scope, dry_run=dry_run, retention_threshold=retention_threshold
        )
        return self._request("POST", "/v1/maintenance", json=body.model_dump(mode="json"))

    def stats(self) -> dict:
        return self._request("GET", "/v1/stats", params=self.scope.model_dump())

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MemoryClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
