import importlib.util
import json
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def task_evaluation():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_tasks.py"
    spec = importlib.util.spec_from_file_location("task_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_fixture_frozen_before_comparison(task_evaluation):
    dataset = json.loads(task_evaluation.DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    assert task_evaluation.fixture_hash(dataset) == task_evaluation.FROZEN_SHA256
    assert len(dataset["scenarios"]) == 8
    assert sum(s["op"] == "query" for x in dataset["scenarios"] for s in x["steps"]) == 25


def test_tasks_no_network_lifecycle_and_known_factual_limitation(task_evaluation, monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("offline task evaluation attempted network")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setenv("FISHKEEPING_DEEPSEEK_API_KEY", "not-a-real-secret")
    monkeypatch.setenv("FISHKEEPING_EMBEDDING_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("FISHKEEPING_EMBEDDING_MODEL", "must-not-be-called")
    dataset = json.loads(task_evaluation.DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    first = task_evaluation.evaluate(dataset)
    second = task_evaluation.evaluate(dataset)
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["safety_forbidden_hits"] == 0
    assert first["metrics"]["known_incorrect_hits"] == 1
    assert first["metrics"]["llm_final_answer_accuracy"] is None
    assert first["cost"]["paid_api_requests"] == 0
    assert first["cost"]["llm_final_answers_evaluated"] == 0
    cases = {case["id"]: case for case in first["cases"]}
    assert not cases["wrong_fact_exposed"]["answer_exact"]
    assert cases["wrong_fact_corrected"]["answer_exact"]
    assert cases["uncertainty_abstain"]["answer"]["status"] == "INSUFFICIENT_EVIDENCE"
    assert cases["deletion_after"]["returned"] == []
    assert cases["preference_after"]["returned"] == ["hot_tea"]
    deleted = next(op["deleted"] for op in first["operations"] if op["delete"] == "garden_source")
    assert deleted == ["garden_fact", "garden_source", "garden_summary"]


def test_quote_proxy_uses_actual_content_and_fixed_confidence(task_evaluation):
    hits = [
        SimpleNamespace(memory=SimpleNamespace(content="possibly wrong", confidence=0.4)),
        SimpleNamespace(
            memory=SimpleNamespace(content="wrong with high confidence", confidence=0.95)
        ),
    ]
    assert task_evaluation.quote_answer(hits) == {
        "status": "EVIDENCE",
        "quotes": ["wrong with high confidence"],
    }
    assert task_evaluation.quote_answer([])["status"] == "INSUFFICIENT_EVIDENCE"


def test_metrics_count_false_positives_and_topk(task_evaluation):
    def case(expected, returned, exact=False):
        return {
            "expected": expected,
            "returned": returned,
            "safety_forbidden_hits": [],
            "known_incorrect_hits": [],
            "answer_exact": exact,
        }

    metrics = task_evaluation.calculate_metrics(
        [case(["a", "b"], ["noise", "a", "b"]), case([], ["unrelated"]), case([], [], True)], 2
    )
    assert metrics["recall_at_k"] == 0.5
    assert metrics["mrr_at_k"] == 0.5
    assert metrics["precision_at_k_nonempty_macro"] == 0.25
    assert metrics["false_positive_hits"] == 2
    assert metrics["false_discovery_fraction"] == 2 / 3
    assert metrics["strict_empty_passed"] == 1
    assert metrics["quote_answer_exact_accuracy"] == 1 / 3


def test_unknown_task_labels_and_backward_time_rejected(task_evaluation):
    dataset = {
        "version": 1,
        "clock": "2026-10-28T00:00:00+00:00",
        "scenarios": [
            {
                "id": "test",
                "steps": [{"op": "query", "id": "q", "expected": ["missing"]}],
            }
        ],
    }
    with pytest.raises(ValueError, match="known record keys"):
        task_evaluation.evaluate(dataset)
    dataset["scenarios"][0]["steps"] = [
        {"op": "put", "key": "a", "content": "hello", "at_seconds": 20},
        {"op": "query", "id": "q", "expected": ["a"], "at_seconds": 10},
    ]
    with pytest.raises(ValueError, match="backwards"):
        task_evaluation.evaluate(dataset)
