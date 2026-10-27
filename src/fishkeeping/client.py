"""Small synchronous HTTP SDK, independent of the caller's agent framework."""

from __future__ import annotations

from urllib.parse import quote

import httpx

from fishkeeping.models import (
    ConsolidateRequest,
    ContextRequest,
    ContextResponse,
    IngestRequest,
    IngestResponse,
    MaintenanceRequest,
    Memory,
    MemoryInput,
    Message,
    RecallRequest,
    RecallResponse,
    Scope,
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

    def get(self, memory_id: str, include_inactive: bool = False) -> Memory:
        return Memory.model_validate(
            self._request(
                "GET",
                f"/v1/memories/{quote(memory_id, safe='')}",
                params={**self.scope.model_dump(), "include_inactive": include_inactive},
            )
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
