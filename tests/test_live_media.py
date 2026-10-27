"""End-to-end real-media experiment bookkeeping with tiny local inputs and fake HTTP."""

import importlib.util
import json
import socket
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def harness(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "live_media_under_test", scripts / "live_media.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.load_manifest()
    digest = module.manifest_hash(manifest)
    assert module.FROZEN_MANIFEST_SHA256 == digest
    prepared = {"fixture_sha256": digest, "requests": []}
    for case in manifest["cases"]:
        for variant in ("baseline", "optimized"):
            # Identical paired payloads simulate a native fallback that must be charged once.
            payload = {
                "model": "deepseek-flash",
                "messages": [
                    {"role": "system", "content": "Answer using supplied evidence. Return JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": case["question"]},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "data:image/png;base64,eA==",
                                    "detail": "low",
                                },
                            },
                        ],
                    },
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
                "thinking": {"type": "disabled"},
                "max_tokens": 1024,
            }
            phase = f"{case['id']}:{variant}"
            path = tmp_path / (phase.replace(":", "-") + ".json")
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            estimate = module.estimate_request(
                httpx.Request("POST", "https://api.deepseek.com/chat/completions", json=payload),
                max_images=12,
            )
            prepared["requests"].append(
                {
                    "phase": phase,
                    "path": str(path),
                    "request_sha256": estimate["request_sha256"],
                    "estimate": estimate,
                    "valid_ids": ["image0"],
                }
            )
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")
    return module, prepared_path, tmp_path / "report.json", tmp_path / "budget.json"


def fake_network(module, monkeypatch, *, include_usage=True):
    calls = []
    original_transport = module.BudgetTransport
    original_settings = module.Settings

    def respond(request):
        calls.append(request)
        if request.method == "GET":
            assert str(request.url) == "https://api.deepseek.com/models"
            return httpx.Response(200, json={"data": [{"id": "deepseek-flash"}]})
        body = {
            "model": "deepseek-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "answer": "PRIVATE_LOCAL_ANSWER",
                                "uncertain": True,
                                "evidence_ids": ["image0"],
                            }
                        )
                    },
                }
            ],
        }
        if include_usage:
            body["usage"] = {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        module,
        "BudgetTransport",
        lambda ledger, **kwargs: original_transport(ledger, httpx.MockTransport(respond), **kwargs),
    )
    monkeypatch.setattr(
        module,
        "Settings",
        lambda **kwargs: original_settings(
            _env_file=None, deepseek_api_key="test-only-key", **kwargs
        ),
    )
    return calls


def rewrite_prepared_request(module, prepared_path, index, mutate):
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    item = prepared["requests"][index]
    path = Path(item["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    estimate = module.estimate_request(
        httpx.Request("POST", "https://api.deepseek.com/chat/completions", json=payload),
        max_images=12,
    )
    item.update(estimate=estimate, request_sha256=estimate["request_sha256"])
    prepared_path.write_text(json.dumps(prepared), encoding="utf-8")


def test_preview_has_no_network_settings_secret_or_paid_reservation(harness, monkeypatch):
    module, prepared, report_path, ledger = harness

    def forbidden(*args, **kwargs):
        pytest.fail("Offline preview attempted network, model setup or credential loading")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(module, "BudgetTransport", forbidden)
    monkeypatch.setattr(module, "Settings", forbidden)
    result = module.run(prepared, report_path, ledger_path=ledger)
    assert result["status"] == "preview"
    assert result["planned_variants"] == 12
    assert result["unique_requests"] == 6
    assert 0 < result["planned_reserved_cost_usd"] <= 0.05
    assert not ledger.exists()
    assert not (prepared.parent / "answers.json").exists()
    assert "base64" not in report_path.read_text(encoding="utf-8")


def test_frozen_pair_count_and_question_are_checked(harness):
    module, prepared, _, _ = harness
    _, plan = module.load_plan(prepared)
    assert len(plan) == 12
    rewrite_prepared_request(
        module,
        prepared,
        0,
        lambda payload: payload["messages"][1]["content"][0].update(text="unfrozen question"),
    )
    with pytest.raises(ValueError, match="Question"):
        module.load_plan(prepared)


def test_changed_prepared_bytes_are_rejected_without_silent_repricing(harness):
    module, prepared, _, _ = harness
    data = json.loads(prepared.read_text(encoding="utf-8"))
    path = Path(data["requests"][0]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["messages"][0]["content"] += " extra"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Prepared request changed"):
        module.load_plan(prepared)


def test_prepared_manifest_hash_must_match_frozen_labels(harness):
    module, prepared, _, _ = harness
    data = json.loads(prepared.read_text(encoding="utf-8"))
    data["fixture_sha256"] = "0" * 64
    prepared.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="Frozen fixture"):
        module.load_plan(prepared)


def test_payload_key_reordering_is_rejected_before_any_paid_request(harness):
    module, prepared, _, _ = harness

    def reverse_keys(payload):
        reordered = dict(reversed(list(payload.items())))
        payload.clear()
        payload.update(reordered)

    rewrite_prepared_request(module, prepared, 0, reverse_keys)
    with pytest.raises(ValueError, match="byte order"):
        module.load_plan(prepared)


def test_prepared_extra_message_cannot_be_silently_dropped_by_adapter(harness):
    module, prepared, _, _ = harness
    rewrite_prepared_request(
        module,
        prepared,
        0,
        lambda payload: payload["messages"].append({"role": "system", "content": "extra"}),
    )
    with pytest.raises((ValueError, module.BudgetError)):
        module.load_plan(prepared)


def test_deduplicated_variants_share_six_reservations_and_restart_reuses(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    calls = fake_network(module, monkeypatch)
    result = module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert result["status"] == "complete"
    assert len([call for call in calls if call.method == "POST"]) == 6
    assert len(result["ledger"]["entries"]) == 6
    assert len(result["results"]) == 12
    assert len([item for item in result["results"] if item["reused_from"]]) == 6
    assert all(item["manual_review_required"] for item in result["results"])
    assert "PRIVATE_LOCAL_ANSWER" not in report_path.read_text(encoding="utf-8")
    assert "test-only-key" not in report_path.read_text(encoding="utf-8")
    restarted = module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert restarted["status"] == "complete"
    assert len([call for call in calls if call.method == "POST"]) == 6
    assert len(restarted["ledger"]["entries"]) == 6


def test_lost_answer_cache_does_not_retry_previously_reserved_calls(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    calls = fake_network(module, monkeypatch)
    module.run(prepared, report_path, ledger_path=ledger, live=True)
    (prepared.parent / "answers.json").unlink()
    result = module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert result["status"] == "incomplete_no_retry"
    assert len(result["skipped_reserved_phases"]) == 12
    assert len([call for call in calls if call.method == "POST"]) == 6
    assert len(result["ledger"]["entries"]) == 6
    assert not result["results"]


def test_reordered_duplicates_cannot_retry_after_answer_cache_loss(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    calls = fake_network(module, monkeypatch)
    module.run(prepared, report_path, ledger_path=ledger, live=True)
    (prepared.parent / "answers.json").unlink()
    data = json.loads(prepared.read_text(encoding="utf-8"))
    data["requests"] = list(reversed(data["requests"]))
    prepared.write_text(json.dumps(data), encoding="utf-8")
    try:
        result = module.run(prepared, report_path, ledger_path=ledger, live=True)
    except (ValueError, module.BudgetError):
        pass  # Rejecting a reordered plan is also safe; it must happen before another POST.
    else:
        assert result["status"] == "incomplete_no_retry"
    assert len([call for call in calls if call.method == "POST"]) == 6
    assert len(json.loads(ledger.read_text())["entries"]) == 6


def test_new_request_cannot_reuse_old_phase_answer_or_reservation(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    calls = fake_network(module, monkeypatch)
    module.run(prepared, report_path, ledger_path=ledger, live=True)
    before = len(calls)
    rewrite_prepared_request(
        module,
        prepared,
        0,
        lambda payload: payload["messages"][1]["content"][1]["image_url"].update(detail="original"),
    )
    with pytest.raises(module.BudgetError):
        module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert len(calls) == before


def test_cache_without_ledger_reservation_is_rejected_before_network(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    calls = fake_network(module, monkeypatch)
    data = json.loads(prepared.read_text(encoding="utf-8"))
    first = data["requests"][0]
    (prepared.parent / "answers.json").write_text(
        json.dumps(
            {
                first["phase"]: {
                    "request_sha256": first["request_sha256"],
                    "answer": {},
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.BudgetError):
        module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert not calls


def test_missing_usage_is_unknown_cost_not_zero_cost(harness, monkeypatch):
    module, prepared, report_path, ledger = harness
    fake_network(module, monkeypatch, include_usage=False)
    result = module.run(prepared, report_path, ledger_path=ledger, live=True)
    assert result["status"] == "complete"
    assert result.get("usage_based_peak_cost_estimate_usd") is None
    assert len(result["ledger"]["entries"]) == 6
    assert sum(item["reserved_nano_usd"] for item in result["ledger"]["entries"]) > 0
