from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from fishkeeping.api import create_app
from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.models import Scope
from fishkeeping.providers import ProviderError
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

SCOPE = {"namespace": "default", "user_id": "alice", "agent_id": "assistant"}
KEY = "test-api-key-at-least-16-characters"


@pytest.fixture
def api(tmp_path):
    settings = Settings(db_path=str(tmp_path / "memory.db"), _env_file=None)
    with TestClient(create_app(settings)) as client:
        yield client


def test_http_lifecycle_and_openapi(api):
    assert api.get("/healthz").status_code == 200
    assert api.get("/readyz").json()["status"] == "ready"
    schema = api.get("/openapi.json").json()
    assert "/v1/ingest" in schema["paths"]
    memory = api.post(
        "/v1/memories", json={"scope": SCOPE, "content": "喜欢 Python", "kind": "semantic"}
    )
    assert memory.status_code == 201, memory.text
    memory_id = memory.json()["id"]
    assert api.get(f"/v1/memories/{memory_id}", params=SCOPE).json()["content"] == "喜欢 Python"
    assert len(api.get("/v1/memories", params=SCOPE).json()) == 1
    hits = api.post("/v1/recall", json={"scope": SCOPE, "query": "Python"}).json()
    assert hits["hits"][0]["memory"]["id"] == memory_id
    result = api.post(f"/v1/memories/{memory_id}/rehearse", json=SCOPE)
    assert result.json()["access_count"] == 1
    context = api.post("/v1/context", json={"scope": SCOPE, "query": "Python"}).json()
    assert context["included_ids"] == [memory_id]
    assert context["chars_used"] == len(context["context"])
    assert api.get("/v1/stats", params=SCOPE).status_code == 200
    assert api.delete(f"/v1/memories/{memory_id}", params=SCOPE).json()["deleted_ids"] == [
        memory_id
    ]
    assert api.get(f"/v1/memories/{memory_id}", params=SCOPE).status_code == 404


@pytest.mark.parametrize("field", ["namespace", "user_id", "agent_id"])
def test_scope_isolation_across_all_read_and_delete_paths(api, field):
    record = api.post("/v1/memories", json={"scope": SCOPE, "content": "secret preference"}).json()
    foreign = {**SCOPE, field: "someone-else"}
    assert api.get(f"/v1/memories/{record['id']}", params=foreign).status_code == 404
    assert api.get("/v1/memories", params=foreign).json() == []
    assert api.post("/v1/recall", json={"scope": foreign, "query": "secret"}).json()["hits"] == []
    assert api.delete(f"/v1/memories/{record['id']}", params=foreign).status_code == 404
    assert api.post(f"/v1/memories/{record['id']}/rehearse", json=foreign).status_code == 404
    result = api.post("/v1/consolidate", json={"scope": foreign, "source_ids": [record["id"]]})
    assert result.status_code == 422
    assert api.get(f"/v1/memories/{record['id']}", params=SCOPE).status_code == 200


def test_auth_key_namespace_binding(tmp_path):
    settings = Settings(
        db_path=str(tmp_path / "auth.db"), api_keys={KEY: "default"}, _env_file=None
    )
    with TestClient(create_app(settings)) as api:
        assert api.get("/healthz").status_code == 200
        for headers in ({}, {"X-API-Key": "wrong"}, {"Authorization": f"Bearer {KEY}"}):
            assert api.get("/v1/memories", params=SCOPE, headers=headers).status_code == 401
        api.headers["X-API-Key"] = KEY
        assert api.get("/v1/memories", params=SCOPE).status_code == 200
        foreign = {**SCOPE, "namespace": "unauthorized"}
        assert api.get("/v1/memories", params=foreign).status_code == 403
        assert api.get("/v1/stats", params=foreign).status_code == 403
        record = api.post("/v1/memories", json={"scope": SCOPE, "content": "test"}).json()
        assert api.get(f"/v1/memories/{record['id']}", params=foreign).status_code == 403
        assert api.delete(f"/v1/memories/{record['id']}", params=foreign).status_code == 403
        assert api.post(f"/v1/memories/{record['id']}/rehearse", json=foreign).status_code == 403
        bodies = {
            "/v1/memories": {"scope": foreign, "content": "test"},
            "/v1/recall": {"scope": foreign, "query": "test"},
            "/v1/context": {"scope": foreign, "query": "test"},
            "/v1/ingest": {
                "scope": foreign,
                "event_id": "id",
                "session_id": "s",
                "messages": [{"role": "user", "content": "test"}],
            },
            "/v1/consolidate": {"scope": foreign, "source_ids": ["fake"]},
            "/v1/maintenance": {"scope": foreign},
        }
        for path, body in bodies.items():
            assert api.post(path, json=body).status_code == 403, path


def test_http_ingestion_conflict_and_no_key(api):
    body = {
        "scope": SCOPE,
        "event_id": "event-1",
        "session_id": "s",
        "messages": [{"role": "user", "content": "记住我喜欢咖啡"}],
    }
    first = api.post("/v1/ingest", json=body)
    assert first.status_code == 200, first.text
    assert first.json()["replayed"] is False
    assert api.post("/v1/ingest", json=body).json()["replayed"] is True
    assert api.post("/v1/ingest", json={**body, "session_id": "other"}).status_code == 409
    assert (
        api.post("/v1/ingest", json={**body, "event_id": "event-2", "extract": True}).status_code
        == 503
    )
    assert len(api.get("/v1/memories", params=SCOPE).json()) == 1


def test_http_source_deletion_cascades_and_tombstones_event(api):
    body = {
        "scope": SCOPE,
        "event_id": "event-1",
        "session_id": "s",
        "messages": [{"role": "user", "content": "偏好中文"}],
    }
    episode = api.post("/v1/ingest", json=body).json()["memories"][0]
    summary = api.post("/v1/consolidate", json={"scope": SCOPE, "source_ids": [episode["id"]]})
    assert summary.status_code == 200, summary.text
    assert summary.json()["method"] == "extractive"
    derived = summary.json()["memories"][0]
    result = api.delete(f"/v1/memories/{episode['id']}", params=SCOPE).json()
    assert set(result["deleted_ids"]) == {episode["id"], derived["id"]}
    assert api.post("/v1/ingest", json=body).status_code == 409


@pytest.mark.parametrize(
    "payload",
    [
        {"content": " "},
        {"content": "x", "kind": "working"},
        {"content": "x", "kind": "episodic", "fact_key": "bad"},
        {"content": "x", "importance": 1.1},
        {"content": "x", "unknown": True},
        {"content": "x", "metadata": {"large": "x" * 16001}},
        {"content": "x", "source_ids": ["nonexistent"]},
    ],
)
def test_invalid_write_is_rejected_without_side_effect(api, payload):
    response = api.post("/v1/memories", json={"scope": SCOPE, **payload})
    assert response.status_code == 422, response.text
    assert api.get("/v1/memories", params=SCOPE).json() == []


def test_invalid_scope_and_pagination_return_422(api):
    assert api.get("/v1/memories", params={**SCOPE, "user_id": " "}).status_code == 422
    assert api.get("/v1/memories", params={**SCOPE, "limit": 0}).status_code == 422
    assert api.get("/v1/memories", params={**SCOPE, "offset": -1}).status_code == 422


def test_upstream_error_is_redacted():
    class BrokenProvider:
        def extract(self, messages):
            raise ProviderError("upstream secret abc123")

    store = Store(":memory:")
    settings = Settings(_env_file=None)
    service = MemoryService(store, settings, llm=BrokenProvider())
    with TestClient(create_app(settings, service)) as api:
        result = api.post(
            "/v1/ingest",
            json={
                "scope": SCOPE,
                "event_id": "e",
                "session_id": "s",
                "extract": True,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert result.status_code == 502
        assert "abc123" not in result.text
        assert store.list(Scope(**SCOPE)) == []
    store.close()


def test_sdk_runs_complete_workflow_against_http_app(api):
    def forward(request: httpx.Request):
        return api.request(
            request.method,
            request.url.path,
            params=request.url.params,
            headers=request.headers,
            content=request.content,
        )

    with MemoryClient(
        "http://testserver", Scope(**SCOPE), transport=httpx.MockTransport(forward)
    ) as sdk:
        memory = sdk.remember("I prefer Python", kind="semantic", fact_key="language")
        assert sdk.get(memory.id).id == memory.id
        assert sdk.list()[0].id == memory.id
        assert sdk.recall("Python").hits[0].memory.id == memory.id
        assert sdk.context("Python").included_ids == [memory.id]
        assert sdk.rehearse(memory.id).access_count == 1
        event = sdk.ingest("e", "s", [{"role": "user", "content": "Today I learned Python"}])
        assert len(sdk.consolidate([event.memories[0].id])) == 1
        assert sdk.maintain()["dry_run"] is True
        assert isinstance(sdk.stats(), dict)
        assert sdk.forget(memory.id) == [memory.id]
        with pytest.raises(httpx.HTTPStatusError):
            sdk.get(memory.id)


def test_disk_persistence_across_application_restart(tmp_path):
    settings = Settings(db_path=str(tmp_path / "persistent.db"), _env_file=None)
    with TestClient(create_app(settings)) as first:
        result = first.post("/v1/memories", json={"scope": SCOPE, "content": "remember me"}).json()
    with TestClient(create_app(settings)) as second:
        assert second.get(f"/v1/memories/{result['id']}", params=SCOPE).status_code == 200
