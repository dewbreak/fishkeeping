import json

import httpx
import pytest
from fastapi.testclient import TestClient

from fishkeeping.api import create_app
from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.models import ExtractedFact, JobInfo, MemoryInput, Scope
from fishkeeping.providers import ProviderError
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store

SCOPE = Scope(namespace="team", user_id="alice", agent_id="helper")
KEY = "test-auth-key-at-least-16-characters"


class Provider:
    def __init__(self):
        self.calls = 0
        self.error = None

    def extract(self, messages):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return [ExtractedFact(content="用户喜欢乌龙茶")]


class Embeddings:
    model_key = "fake-multilingual-v1"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]


def body(**kwargs):
    return {
        "scope": SCOPE.model_dump(),
        "event_id": "one",
        "session_id": "chat",
        "messages": [{"role": "user", "content": "我喜欢乌龙茶"}],
        **kwargs,
    }


@pytest.fixture
def api():
    settings = Settings(_env_file=None, deepseek_api_key=None, api_keys={KEY: "team"})
    store = Store(":memory:")
    provider, embeddings = Provider(), Embeddings()
    service = MemoryService(store, settings, llm=provider, embeddings=embeddings)
    app = create_app(settings, service)
    try:
        with TestClient(app, headers={"X-API-Key": KEY}) as client:
            yield client, app, service, provider, embeddings
    finally:
        store.close()


def test_durable_http_lifecycle_worker_replay_and_source_provenance(api):
    client, app, _, provider, _ = api
    response = client.post("/v1/events", json=body())
    assert response.status_code == 202, response.text
    receipt = response.json()
    assert receipt["job"]["state"] == "pending"
    assert receipt["job"]["attempts"] == 0
    assert provider.calls == 0
    result = app.state.queue.process_one(provider)
    assert result["state"] == "succeeded"
    current = client.get("/v1/events/one", params=SCOPE.model_dump())
    assert current.status_code == 200, current.text
    assert JobInfo.model_validate(current.json()["job"]) == JobInfo.model_validate(result)
    assert current.json()["episode"]["id"] == receipt["episode"]["id"]
    sources = client.get(
        f"/v1/memories/{result['result_ids'][0]}/sources", params=SCOPE.model_dump()
    ).json()
    assert sources["items"] == [receipt["episode"]["id"]]
    replay = client.post("/v1/events", json=body()).json()
    assert replay["replayed"] is True
    assert JobInfo.model_validate(replay["job"]) == JobInfo.model_validate(result)
    assert provider.calls == 1
    assert "lease_token" not in current.json()["job"]
    assert client.post("/v1/events", json=body(session_id="changed")).status_code == 409
    deleted = client.delete(f"/v1/memories/{receipt['episode']['id']}", params=SCOPE.model_dump())
    assert set(deleted.json()["deleted_ids"]) == {receipt["episode"]["id"], *result["result_ids"]}
    assert client.get("/v1/events/one", params=SCOPE.model_dump()).status_code == 409
    assert client.post("/v1/events", json=body()).status_code == 409
    assert client.post("/v1/events/one/retry", json=SCOPE.model_dump()).status_code == 409


def test_http_failure_keeps_event_and_explicit_retry_can_finish(api):
    client, app, _, provider, _ = api
    client.post("/v1/events", json=body())
    provider.error = ProviderError("private provider response", code="timeout")
    assert app.state.queue.process_one(provider)["state"] == "failed"
    failed = client.get("/v1/events/one", params=SCOPE.model_dump())
    assert failed.json()["episode"]["content"] == "[user] 我喜欢乌龙茶"
    assert "private provider" not in failed.text
    retried = client.post("/v1/events/one/retry", json=SCOPE.model_dump())
    assert retried.status_code == 200, retried.text
    assert retried.json()["attempts"] == 1
    assert retried.json()["state"] == "pending"
    provider.error = None
    assert app.state.queue.process_one(provider)["state"] == "succeeded"
    assert provider.calls == 2


@pytest.mark.parametrize("field", ["user_id", "agent_id"])
def test_durable_receipt_and_sources_are_scoped(api, field):
    client, _, _, _, _ = api
    receipt = client.post("/v1/events", json=body()).json()
    foreign = {**SCOPE.model_dump(), field: "different"}
    assert client.get("/v1/events/one", params=foreign).status_code == 404
    assert client.post("/v1/events/one/retry", json=foreign).status_code == 409
    assert (
        client.get(f"/v1/memories/{receipt['episode']['id']}/sources", params=foreign).status_code
        == 404
    )


def test_all_new_routes_require_valid_namespace_key_before_work(api):
    client, app, service, provider, embeddings = api
    foreign = {**SCOPE.model_dump(), "namespace": "not-authorized"}
    operations = [
        ("POST", "/v1/events", {"json": body(scope=foreign)}),
        ("GET", "/v1/events/one", {"params": foreign}),
        ("POST", "/v1/events/one/retry", {"json": foreign}),
        ("GET", "/v1/memories/missing/sources", {"params": foreign}),
        ("POST", "/v1/index", {"json": {"scope": foreign}}),
    ]
    for method, path, kwargs in operations:
        assert client.request(method, path, **kwargs).status_code == 403
        response = client.request(method, path, headers={"X-API-Key": "wrong"}, **kwargs)
        assert response.status_code == 401
    assert service.store.stats(SCOPE)["total"] == 0
    assert app.state.queue.budget_status()["attempts"] == 0
    assert provider.calls == 0
    assert embeddings.calls == []


def test_paginated_sources_keep_complete_provenance_beyond_preview(api):
    client, _, service, _, _ = api
    sources = service.store.put_many(
        [MemoryInput(scope=SCOPE, content=f"Evidence {number}") for number in range(105)]
    )[0]
    derived = None
    for start in (0, 100):
        derived = service.store.put(
            MemoryInput(
                scope=SCOPE,
                content="shared conclusion",
                kind="semantic",
                source_ids=[memory.id for memory in sources[start : start + 100]],
            )
        )
    preview = client.get(f"/v1/memories/{derived.id}", params=SCOPE.model_dump()).json()
    assert preview["source_count"] == 105
    assert len(preview["source_ids"]) == 100
    first = client.get(
        f"/v1/memories/{derived.id}/sources", params={**SCOPE.model_dump(), "limit": 100}
    ).json()
    second = client.get(
        f"/v1/memories/{derived.id}/sources",
        params={**SCOPE.model_dump(), "limit": 100, "offset": 100},
    ).json()
    assert first["has_more"] is True and second["has_more"] is False
    assert first["total"] == second["total"] == 105
    assert set(first["items"] + second["items"]) == {source.id for source in sources}
    invalid = client.get(
        f"/v1/memories/{derived.id}/sources", params={**SCOPE.model_dump(), "limit": 1001}
    )
    assert invalid.status_code == 422


def test_explicit_index_is_bounded_and_separate_from_recall(api):
    client, _, service, _, embeddings = api
    service.store.put_many(
        [
            MemoryInput(scope=SCOPE, content="I prefer tea"),
            MemoryInput(scope=SCOPE, content="我喜欢乌龙茶"),
        ]
    )
    other = SCOPE.model_copy(update={"user_id": "other"})
    service.store.put(MemoryInput(scope=other, content="private"))
    first = client.post("/v1/index", json={"scope": SCOPE.model_dump(), "limit": 1})
    assert first.status_code == 200, first.text
    assert first.json()["processed"] == 1
    assert first.json()["indexed"] == 1
    assert first.json()["total"] == 2
    assert first.json()["complete"] is False
    assert sum(map(len, embeddings.calls)) == 1
    second = client.post("/v1/index", json={"scope": SCOPE.model_dump(), "limit": 1})
    assert second.json()["complete"] is True
    assert service.store.embedding_coverage(other, embeddings.model_key)["indexed"] == 0
    embeddings.calls.clear()
    recalled = client.post("/v1/recall", json={"scope": SCOPE.model_dump(), "query": "tea"})
    assert recalled.status_code == 200
    assert recalled.json()["retrieval_mode"] == "hybrid"
    assert embeddings.calls == [["tea"]]
    assert (
        client.post("/v1/index", json={"scope": SCOPE.model_dump(), "limit": 0}).status_code == 422
    )


def test_sdk_new_routes_work_through_real_api_and_preserve_scope(api):
    client, app, _, provider, _ = api
    observed = []

    def transport(request):
        observed.append((request.method, request.url.path, request.headers.get("x-api-key")))
        response = client.request(
            request.method,
            request.url.path,
            params=dict(request.url.params),
            content=request.content,
            headers=dict(request.headers),
        )
        return httpx.Response(response.status_code, json=response.json())

    with MemoryClient(
        "http://memory.test", SCOPE, api_key=KEY, transport=httpx.MockTransport(transport)
    ) as sdk:
        event = sdk.enqueue("one", "chat", [{"role": "user", "content": "我喜欢乌龙茶"}])
        assert event.job.state == "pending"
        provider.error = ProviderError(code="timeout")
        app.state.queue.process_one(provider)
        assert sdk.event("one").job.state == "failed"
        assert sdk.retry_event("one").state == "pending"
        provider.error = None
        app.state.queue.process_one(provider)
        result = sdk.event("one")
        assert result.job.state == "succeeded"
        page = sdk.sources(result.job.result_ids[0], limit=1)
        assert page.items == [event.episode.id]
        assert sdk.index(limit=1).processed == 1
        assert sdk.enqueue("one", "chat", [{"role": "user", "content": "我喜欢乌龙茶"}]).replayed
        with pytest.raises(httpx.HTTPStatusError) as error:
            sdk.event("missing")
        assert error.value.response.status_code == 404
    assert all(key == KEY for _, _, key in observed)
    assert "private provider" not in json.dumps(observed)
