import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("task_answers", SCRIPTS / "live_task_answers.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_distinct_scenarios_and_no_labels_in_prompt():
    dataset = json.loads(
        (Path(__file__).parent / "fixtures/task_memory_cases.json").read_text(encoding="utf-8")
    )
    scenarios = [
        s["id"]
        for s in dataset["scenarios"]
        for step in s["steps"]
        if step.get("id") in module.LABELS
    ]
    assert len(scenarios) == len(set(scenarios)) == 8
    for mode in ("lexical", "local"):
        payload, _ = module.prepare(mode)
        tasks = json.loads(payload["messages"][1]["content"])
        assert len(tasks) == 8
        assert all(set(task) == {"id", "question", "evidence"} for task in tasks)
        assert "expense_wrong" not in json.dumps(payload)
        assert "wrong_fact_exposed" not in json.dumps(payload)


def test_score_and_incomplete_output():
    answers = [
        {
            "id": module.TASK_IDS[key],
            "answer": " ".join(group[0] for group in value["terms"]) or "INSUFFICIENT_EVIDENCE",
            "uncertain": value["uncertain"],
        }
        for key, value in module.LABELS.items()
    ]
    body = {
        "choices": [
            {"finish_reason": "stop", "message": {"content": json.dumps({"answers": answers})}}
        ]
    }
    assert all(result["correct"] for result in module.score(body))
    body["choices"][0]["finish_reason"] = "length"
    with pytest.raises(ValueError):
        module.score(body)
