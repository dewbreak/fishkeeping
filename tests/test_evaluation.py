import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def evaluation():
    path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("offline_evaluate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_use_positive_queries_and_preserve_empty_failures(evaluation):
    cases = [
        {"expected": ["a", "b"], "returned": ["noise", "a"], "forbidden_hits": []},
        {"expected": ["c"], "returned": [], "forbidden_hits": []},
        {"expected": [], "returned": [], "forbidden_hits": []},
        {"expected": [], "returned": ["private"], "forbidden_hits": ["private"]},
    ]
    result = evaluation.calculate_metrics(cases, 2)
    assert result["recall_at_k"] == 0.25
    assert result["mrr_at_k"] == 0.25
    assert result["strict_empty_cases"] == 2
    assert result["strict_empty_passed"] == 1
    assert result["forbidden_hits"] == 1
    assert evaluation.calculate_metrics([], 3)["recall_at_k"] is None


def test_offline_fixture_has_no_network_and_is_repeatable(evaluation, monkeypatch):
    def reject_network(*args, **kwargs):
        pytest.fail("offline evaluation attempted network access")

    monkeypatch.setattr(httpx.Client, "send", reject_network)
    monkeypatch.setenv("FISHKEEPING_DEEPSEEK_API_KEY", "not-a-real-secret")
    monkeypatch.setenv("FISHKEEPING_EMBEDDING_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("FISHKEEPING_EMBEDDING_MODEL", "must-not-be-used")
    dataset = json.loads(evaluation.DEFAULT_FIXTURE.read_text(encoding="utf-8"))
    first = evaluation.evaluate(dataset)
    assert first == evaluation.evaluate(dataset)
    assert first["model_calls"] == 0
    assert first["record_count"] >= 20
    assert first["case_count"] >= 12
    assert first["strict_checks_passed"] is True
    assert all(case["retrieval_mode"] == "lexical" for case in first["cases"])
    # Keep hard semantic cases visible; a lexical-only system should not claim perfect coverage.
    assert any(case["missing"] for case in first["cases"] if case["category"] == "semantic-gap")


def test_invalid_fixture_labels_are_rejected(evaluation):
    dataset = {
        "version": 1,
        "clock": "2026-10-28T00:00:00+00:00",
        "scope": {"user_id": "test"},
        "records": [{"key": "known", "content": "hello"}],
        "cases": [{"id": "bad", "query": "hello", "expected": ["missing"]}],
    }
    with pytest.raises(ValueError, match="known record keys"):
        evaluation.evaluate(dataset)


def test_metric_top_k_is_respected(evaluation):
    result = evaluation.calculate_metrics(
        [{"expected": ["target"], "returned": ["noise", "target"]}], 1
    )
    assert result["recall_at_k"] == 0
    assert result["mrr_at_k"] == 0
