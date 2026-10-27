import json
import sys

import pytest

from fishkeeping import cli
from fishkeeping.config import Settings
from fishkeeping.jobs import EventQueue
from fishkeeping.models import EventRequest, ExtractedFact, MemoryInput, Message, Scope
from fishkeeping.storage import Store

SCOPE = Scope(namespace="team", user_id="alice", agent_id="helper")


class Provider:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def extract(self, messages):
        self.calls += 1
        return [ExtractedFact(content="safe extracted preference")]

    def close(self):
        self.closed = True


class Embeddings:
    model_key = "offline-fake-v1"

    def __init__(self):
        self.calls = []
        self.closed = False

    def embed(self, texts):
        self.calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def close(self):
        self.closed = True


def config(tmp_path, **kwargs):
    return Settings(
        _env_file=None,
        db_path=str(tmp_path / "cli.sqlite3"),
        deepseek_api_key="fake-key-never-used-on-network",
        **kwargs,
    )


def prepare(settings, count=3):
    store = Store(settings.db_path)
    try:
        queue = EventQueue(store, settings)
        for number in range(count):
            queue.enqueue(
                EventRequest(
                    scope=SCOPE,
                    event_id=f"event-{number}",
                    session_id="chat",
                    messages=[Message(role="user", content=f"private user content {number}")],
                )
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "once,max_jobs,daily_limit,expected",
    [(True, 10, 20, 1), (False, 2, 20, 2), (False, 10, 1, 1), (False, 10, 0, 0)],
)
def test_worker_honors_once_job_cap_daily_budget_and_redacts_output(
    tmp_path, monkeypatch, capsys, once, max_jobs, daily_limit, expected
):
    settings = config(tmp_path, worker_daily_call_limit=daily_limit)
    prepare(settings)
    provider = Provider()
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda _settings: provider)
    cli.run_worker(settings, once=once, max_jobs=max_jobs, poll_interval=0.1)
    assert provider.calls == expected
    assert provider.closed
    output = capsys.readouterr().out
    assert "private user content" not in output
    assert "safe extracted preference" not in output
    assert "fake-key" not in output
    summary = json.loads(output.splitlines()[-1])
    assert summary["processed"] == expected
    assert summary["budget"]["attempts"] == expected
    store = Store(settings.db_path)
    try:
        assert EventQueue(store, settings).budget_status()["attempts"] == expected
        rows = store._conn.execute("SELECT state FROM extraction_jobs").fetchall()
        assert sum(row[0] == "succeeded" for row in rows) == expected
        assert sum(row[0] == "pending" for row in rows) == 3 - expected
    finally:
        store.close()


def test_once_empty_queue_exits_without_sleep_or_model_request(tmp_path, monkeypatch, capsys):
    settings = config(tmp_path)
    provider = Provider()
    monkeypatch.setattr(cli, "DeepSeekProvider", lambda _settings: provider)

    def unexpected_sleep(_delay):
        pytest.fail("--once must not poll an empty queue")

    monkeypatch.setattr(cli.time, "sleep", unexpected_sleep)
    cli.run_worker(settings, once=True, max_jobs=20, poll_interval=2)
    assert provider.calls == 0
    assert provider.closed
    assert json.loads(capsys.readouterr().out)["processed"] == 0


def test_worker_requires_key_without_consuming_pending_jobs(tmp_path, monkeypatch):
    settings = config(tmp_path)
    prepare(settings, count=1)
    settings.deepseek_api_key = None

    def unexpected_provider(_settings):
        pytest.fail("Worker cannot create a provider without a key")

    monkeypatch.setattr(cli, "DeepSeekProvider", unexpected_provider)
    with pytest.raises(SystemExit, match="FISHKEEPING_DEEPSEEK_API_KEY"):
        cli.run_worker(settings, once=True, max_jobs=1, poll_interval=2)
    store = Store(settings.db_path)
    try:
        queue = EventQueue(store, settings)
        assert queue.get(SCOPE, "event-0")["job"]["state"] == "pending"
        assert queue.budget_status()["attempts"] == 0
    finally:
        store.close()


def test_cli_index_obeys_explicit_batch_and_run_limit_without_network(
    tmp_path, monkeypatch, capsys
):
    settings = config(tmp_path)
    store = Store(settings.db_path)
    for number in range(7):
        store.put(MemoryInput(scope=SCOPE, content=f"Document {number}"))
    store.close()
    embeddings = Embeddings()
    monkeypatch.setattr(cli, "create_embeddings", lambda _settings: embeddings)
    cli.run_index(settings, SCOPE, batch_size=2, max_batches=2)
    assert sum(map(len, embeddings.calls)) == 4
    assert embeddings.closed
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(output) == 2
    assert output[-1]["indexed"] == 4
    assert output[-1]["complete"] is False
    assert all("Document" not in json.dumps(result) for result in output)
    embeddings.calls.clear()
    cli.run_index(settings, SCOPE, batch_size=2, max_batches=20)
    assert sum(map(len, embeddings.calls)) == 3
    result = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(result) == 2
    assert result[-1]["complete"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ["worker", "--max-jobs", "0"],
        ["index", "--user-id", "alice", "--max-batches", "-1"],
        ["index", "--user-id", "alice", "--batch-size", "0"],
    ],
)
def test_cli_rejects_unbounded_or_empty_run_counts_before_loading_env(monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["fishkeeping", *arguments])

    def unexpected_settings():
        pytest.fail("Invalid CLI arguments must be rejected before reading configuration")

    monkeypatch.setattr(cli, "Settings", unexpected_settings)
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
