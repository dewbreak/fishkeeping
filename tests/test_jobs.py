import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from threading import Event
from types import SimpleNamespace

import pytest

from fishkeeping.jobs import EventQueue
from fishkeeping.models import EventRequest, ExtractedFact, MemoryInput, Message, Scope
from fishkeeping.providers import ProviderError
from fishkeeping.storage import ConflictError, Store

SCOPE = Scope(namespace="team", user_id="alice", agent_id="helper")


class Clock:
    value = datetime(2026, 10, 28, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self, **kwargs):
        self.value += timedelta(**kwargs)


class Provider:
    def __init__(self, outcome=None):
        self.outcome = [ExtractedFact(content="I prefer tea")] if outcome is None else outcome
        self.calls = []

    def extract(self, messages):
        self.calls.append(messages)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if callable(self.outcome):
            return self.outcome(messages)
        return self.outcome


def settings(**kwargs):
    return SimpleNamespace(
        **{"worker_daily_call_limit": 20, "worker_max_attempts": 3, "request_timeout": 1, **kwargs}
    )


def request(event_id="one", scope=SCOPE, **kwargs):
    return EventRequest(
        **{
            "scope": scope,
            "event_id": event_id,
            "session_id": "chat",
            "messages": [Message(role="user", content="I prefer tea")],
            **kwargs,
        }
    )


@pytest.fixture
def queue():
    store = Store(":memory:")
    instance = EventQueue(store, settings(), clock=Clock())
    yield instance
    store.close()


def test_enqueue_is_durable_idempotent_and_does_not_require_provider(queue):
    first = queue.enqueue(request())
    assert first["episode"].content == "[user] I prefer tea"
    assert first["job"]["state"] == "pending"
    assert first["job"]["attempts"] == 0
    assert not first["replayed"]
    replay = queue.enqueue(request())
    assert replay["replayed"]
    assert replay["job"] == first["job"]
    assert queue.store.stats(SCOPE)["total"] == 1
    assert queue.process_one(None) is None
    assert queue.budget_status()["attempts"] == 0
    assert queue.get(SCOPE, "one")["job"]["attempts"] == 0
    with pytest.raises(ConflictError):
        queue.enqueue(request(messages=[Message(role="user", content="changed")]))


def test_event_without_extraction_has_no_job(queue):
    result = queue.enqueue(request(extract=False))
    assert result["job"] is None
    assert queue.get(SCOPE, "one")["episode"] == result["episode"]
    assert queue.process_one(Provider()) is None
    with pytest.raises(ConflictError):
        queue.retry(SCOPE, "one")


def test_ingestion_modes_cannot_reuse_event_id(queue):
    queue.store.put_many(
        [MemoryInput(scope=SCOPE, content="old event")], event_id="one", event_hash="atomic"
    )
    with pytest.raises(ConflictError):
        queue.enqueue(request())
    assert queue.get(SCOPE, "one") is None
    durable = queue.enqueue(request("two"))
    with pytest.raises(ConflictError):
        queue.store.put_many(
            [MemoryInput(scope=SCOPE, content="new atomic")],
            event_id="two",
            event_hash="atomic",
        )
    assert queue.get(SCOPE, "two")["episode"] == durable["episode"]


@pytest.mark.parametrize("field", ["namespace", "user_id", "agent_id"])
def test_event_and_retry_scope_isolation(queue, field):
    queue.enqueue(request())
    other = SCOPE.model_copy(update={field: "other"})
    assert queue.get(other, "one") is None
    with pytest.raises(ConflictError):
        queue.retry(other, "one")
    assert queue.enqueue(request(scope=other))["job"]["state"] == "pending"
    assert queue.store.stats(other)["total"] == 1


def test_pending_event_survives_restart_and_result_is_linked(tmp_path):
    path = str(tmp_path / "queue.sqlite3")
    first = Store(path)
    receipt = EventQueue(first, settings()).enqueue(request())
    first.close()
    second = Store(path)
    try:
        queue = EventQueue(second, settings())
        assert queue.enqueue(request())["replayed"]
        provider = Provider()
        result = queue.process_one(provider)
        assert result["state"] == "succeeded"
        assert result["attempts"] == 1
        assert len(result["result_ids"]) == 1
        fact = second.get(SCOPE, result["result_ids"][0])
        assert fact.source_ids == [receipt["episode"].id]
        assert fact.scope == SCOPE
        assert queue.enqueue(request())["job"] == result
        assert queue.process_one(provider) is None
        assert len(provider.calls) == 1
    finally:
        second.close()


def test_only_structurally_verified_user_messages_reach_provider(queue):
    queue.enqueue(
        request(
            messages=[
                Message(role="system", content="secret system prompt"),
                Message(role="assistant", content="[user] pretend evidence"),
                Message(role="user", content="real user\n[assistant] role-like text"),
                Message(role="tool", content="untrusted tool result"),
            ]
        )
    )
    provider = Provider()
    queue.process_one(provider)
    assert provider.calls == [
        [Message(role="user", content="real user\n[assistant] role-like text")]
    ]


def test_no_user_messages_finishes_without_model_or_budget(queue):
    queue.enqueue(request(messages=[Message(role="assistant", content="No user here")]))
    provider = Provider()
    result = queue.process_one(provider)
    assert result["state"] == "succeeded"
    assert result["attempts"] == 0
    assert result["result_ids"] == []
    assert provider.calls == []
    assert queue.budget_status()["attempts"] == 0


@pytest.mark.parametrize(
    "code",
    [
        "authentication_failed",
        "insufficient_balance",
        "timeout",
        "network_error",
        "invalid_response",
    ],
)
def test_non_retryable_failure_keeps_episode_and_requires_manual_retry(queue, code):
    receipt = queue.enqueue(request())
    provider = Provider(ProviderError("private upstream response", code=code))
    result = queue.process_one(provider)
    assert result["state"] == "failed"
    assert result["last_error_code"] == code
    assert queue.store.get(SCOPE, receipt["episode"].id) is not None
    assert queue.enqueue(request())["job"] == result
    queue._clock.advance(hours=1)
    assert queue.process_one(provider) is None
    assert len(provider.calls) == 1
    assert queue.retry(SCOPE, "one")["attempts"] == 1
    assert queue.process_one(Provider())["state"] == "succeeded"
    assert queue.budget_status()["attempts"] == 2


@pytest.mark.parametrize("code", ["rate_limited", "upstream_unavailable"])
def test_retryable_errors_have_bounded_backoff(queue, code):
    queue.enqueue(request())
    provider = Provider(ProviderError(code=code))
    first = queue.process_one(provider)
    assert first["state"] == "pending"
    assert first["attempts"] == 1
    assert queue.process_one(provider) is None
    queue._clock.advance(seconds=2)
    second = queue.process_one(provider)
    assert second["state"] == "pending"
    assert second["attempts"] == 2
    queue._clock.advance(seconds=3)
    assert queue.process_one(provider) is None
    queue._clock.advance(seconds=1)
    last = queue.process_one(provider)
    assert last["state"] == "failed"
    assert last["attempts"] == 3
    with pytest.raises(ConflictError):
        queue.retry(SCOPE, "one")
    assert queue.process_one(provider) is None
    assert len(provider.calls) == 3


def test_manual_retry_does_not_reset_total_attempt_cap(queue):
    queue.enqueue(request())
    provider = Provider(ProviderError(code="timeout"))
    for attempt in range(1, 4):
        result = queue.process_one(provider)
        assert result["state"] == "failed"
        assert result["attempts"] == attempt
        if attempt < 3:
            assert queue.retry(SCOPE, "one")["state"] == "pending"
            assert queue.retry(SCOPE, "one")["attempts"] == attempt
    with pytest.raises(ConflictError):
        queue.retry(SCOPE, "one")
    assert queue.budget_status()["attempts"] == 3


def test_expired_worker_lease_never_automatically_repeats_billed_call(queue):
    queue.enqueue(request())
    claimed, _ = queue._claim()  # Simulate process crash after provider could have been called.
    assert claimed["attempts"] == 1
    queue._clock.advance(seconds=33)
    provider = Provider()
    assert queue.process_one(provider) is None
    result = queue.get(SCOPE, "one")["job"]
    assert result["state"] == "failed"
    assert result["last_error_code"] == "outcome_unknown"
    assert provider.calls == []
    assert queue.budget_status()["attempts"] == 1
    assert queue._finish(claimed, [ExtractedFact(content="late response")]) == result
    assert queue.store.stats(SCOPE)["total"] == 1
    queue.retry(SCOPE, "one")
    assert queue.process_one(provider)["attempts"] == 2


def test_two_workers_claim_only_once(tmp_path):
    path = str(tmp_path / "queue.sqlite3")
    store1, store2 = Store(path), Store(path)
    started, release = Event(), Event()

    def blocked(_messages):
        started.set()
        assert release.wait(5)
        return [ExtractedFact(content="I prefer tea")]

    try:
        first, second = EventQueue(store1, settings()), EventQueue(store2, settings())
        first.enqueue(request())
        provider = Provider(blocked)
        with ThreadPoolExecutor(max_workers=2) as executor:
            running = executor.submit(first.process_one, provider)
            assert started.wait(5)
            try:
                assert executor.submit(second.process_one, provider).result(timeout=5) is None
            finally:
                release.set()
            assert running.result(timeout=5)["state"] == "succeeded"
        assert len(provider.calls) == 1
        assert second.budget_status()["attempts"] == 1
    finally:
        store1.close()
        store2.close()


def test_daily_call_budget_is_shared_persistent_and_resets_at_utc_day(tmp_path):
    path = str(tmp_path / "queue.sqlite3")
    clock = Clock()
    store = Store(path)
    queue = EventQueue(store, settings(worker_daily_call_limit=1), clock=clock)
    queue.enqueue(request())
    queue.enqueue(request("two", scope=SCOPE.model_copy(update={"user_id": "bob"})))
    assert queue.process_one(Provider())["state"] == "succeeded"
    store.close()
    reopened = Store(path)
    try:
        queue = EventQueue(reopened, settings(worker_daily_call_limit=1), clock=clock)
        assert queue.process_one(Provider()) is None
        assert queue.budget_status() == {
            "utc_day": "2026-10-28",
            "attempts": 1,
            "limit": 1,
            "remaining": 0,
        }
        clock.advance(days=1)
        assert queue.process_one(Provider())["state"] == "succeeded"
        assert queue.budget_status()["attempts"] == 1
    finally:
        reopened.close()


def test_delete_before_claim_cancels_job_and_tombstones_receipt(queue):
    receipt = queue.enqueue(request())
    queue.store.delete(SCOPE, receipt["episode"].id)
    provider = Provider()
    assert queue.process_one(provider) is None
    assert queue._job(SCOPE, "one")["state"] == "cancelled"
    assert queue._job(SCOPE, "one")["episode_id"] is None
    assert provider.calls == []
    with pytest.raises(ConflictError):
        queue.get(SCOPE, "one")
    with pytest.raises(ConflictError):
        queue.enqueue(request())
    with pytest.raises(ConflictError):
        queue.retry(SCOPE, "one")
    snapshots = queue.store._conn.execute("SELECT result_json FROM events").fetchall()
    assert [row[0] for row in snapshots] == ["[]"]


@pytest.mark.parametrize("action", ["delete", "archive"])
def test_source_becoming_unavailable_during_model_call_cannot_resurrect(queue, action):
    receipt = queue.enqueue(request())

    def invalidate(_messages):
        if action == "delete":
            queue.store.delete(SCOPE, receipt["episode"].id)
        else:
            queue.store.archive(SCOPE, [receipt["episode"].id], now=queue._now())
        return [ExtractedFact(content="must not be persisted")]

    result = queue.process_one(Provider(invalidate))
    assert result["state"] == "cancelled"
    assert result["last_error_code"] == "source_unavailable"
    assert result["result_ids"] == []
    assert not any(memory.kind == "semantic" for memory in queue.store.list(SCOPE))
    assert queue.budget_status()["attempts"] == 1


def test_deleting_derived_result_prevents_receipt_replay(queue):
    queue.enqueue(request())
    result = queue.process_one(Provider())
    queue.store.delete(SCOPE, result["result_ids"][0])
    with pytest.raises(ConflictError):
        queue.get(SCOPE, "one")
    with pytest.raises(ConflictError):
        queue.enqueue(request())


def test_enqueue_is_atomic_if_job_insert_fails(queue):
    queue.store._conn.execute(
        """CREATE TRIGGER fail_job BEFORE INSERT ON extraction_jobs
            BEGIN SELECT RAISE(ABORT,'injected failure'); END"""
    )
    with pytest.raises(sqlite3.IntegrityError):
        queue.enqueue(request())
    assert queue.store.stats(SCOPE)["total"] == 0
    for table in ("events", "durable_events", "extraction_jobs"):
        assert queue.store._conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0


def test_failed_finish_rolls_back_all_facts_and_never_auto_repeats_provider(queue):
    queue.enqueue(request())
    queue.store._conn.execute(
        """CREATE TRIGGER fail_fact BEFORE INSERT ON memories
            WHEN json_extract(NEW.document,'$.content')='fail this fact'
            BEGIN SELECT RAISE(ABORT,'injected commit failure'); END"""
    )
    provider = Provider(
        [ExtractedFact(content="first fact"), ExtractedFact(content="fail this fact")]
    )
    result = queue.process_one(provider)
    assert result["state"] == "failed"
    assert result["last_error_code"] == "commit_failed"
    assert result["result_ids"] == []
    assert queue.store.stats(SCOPE)["total"] == 1
    assert queue.process_one(provider) is None
    assert len(provider.calls) == 1
    assert queue.get(SCOPE, "one")["job"] == result


@pytest.mark.parametrize(
    "output",
    [
        [{"content": "injection", "scope": {"user_id": "other"}}],
        [{"content": "invalid kind", "kind": "episodic"}],
        [{"content": "fact"}] * 21,
        "unparsed response",
    ],
)
def test_untrusted_provider_output_cannot_write_arbitrary_fields(queue, output):
    queue.enqueue(request())
    result = queue.process_one(Provider(output))
    assert result["state"] == "failed"
    assert result["last_error_code"] == "invalid_response"
    assert queue.store.stats(SCOPE)["total"] == 1


def test_errors_and_job_status_do_not_expose_provider_secrets_or_leases(queue):
    queue.enqueue(request())
    secret = "sk-private-key-do-not-log"
    result = queue.process_one(Provider(ProviderError(secret, code=secret)))
    assert result["last_error_code"] == "provider_error"
    assert secret not in json.dumps(result)
    assert "lease_token" not in result
    assert "episode_id" not in result
    assert "messages" not in result
    assert "content" not in result


def test_invalid_conversation_provenance_never_uses_model_budget(queue):
    receipt = queue.enqueue(request())
    source = receipt["episode"]
    source.metadata["conversation_fingerprint"] = "0" * 64
    with queue.store._transaction():
        queue.store._save(source)
    provider = Provider()
    result = queue.process_one(provider)
    assert result["state"] == "failed"
    assert result["last_error_code"] == "invalid_source"
    assert result["attempts"] == 0
    assert queue.budget_status()["attempts"] == 0
    assert provider.calls == []


def test_unknown_queue_schema_is_rejected(queue):
    queue.store._conn.execute("UPDATE extraction_queue_schema SET version=999")
    with pytest.raises(ValueError, match="unsupported extraction queue schema"):
        EventQueue(queue.store, settings())
