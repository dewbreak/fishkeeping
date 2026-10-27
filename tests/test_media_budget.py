"""Regression coverage for the opt-in 12-call/12-image experiment; no paid calls."""

import base64
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def budget_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "live_images.py"
    spec = importlib.util.spec_from_file_location("media_budget_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def video_payload(image_count=12, *, text="Frozen question", image_bytes=b"fixture-pixels"):
    blocks = [{"type": "text", "text": text}, {"type": "text", "text": "No audio evidence"}]
    for index in range(image_count):
        blocks.extend(
            [
                {"type": "text", "text": f"frame{index} timestamp={index}.00 seconds"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode(),
                        "detail": "low",
                    },
                },
            ]
        )
    return {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": blocks},
        ],
        "thinking": {"type": "disabled"},
        "stream": False,
        "response_format": {"type": "json_object"},
        "max_tokens": 1024,
    }


def request(payload=None):
    return httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
        json=video_payload() if payload is None else payload,
    )


def response_with_usage(req):
    return httpx.Response(
        200,
        json={
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 20,
                "total_tokens": 220,
            }
        },
    )


@pytest.mark.parametrize("cap", [0, 13, True, False, 12.0, "12", None])
def test_extended_caps_require_bounded_integer(budget_module, tmp_path, cap):
    with pytest.raises(budget_module.BudgetError, match="call cap"):
        budget_module.LiveBudget(tmp_path / "budget.json", max_calls=cap)
    with pytest.raises(budget_module.BudgetError, match="image cap"):
        budget_module.estimate_request(request(), max_images=cap)


def test_default_limits_remain_eight_calls_and_three_images(budget_module, tmp_path):
    assert budget_module.LiveBudget(tmp_path / "budget.json").snapshot()["max_calls"] == 8
    assert budget_module.estimate_request(request(video_payload(3)))["image_count"] == 3
    with pytest.raises(budget_module.BudgetError):
        budget_module.estimate_request(request(video_payload(4)))
    with pytest.raises(budget_module.BudgetError):
        budget_module.estimate_request(request())


def test_eight_call_ledger_cannot_be_reopened_as_twelve(budget_module, tmp_path):
    path = tmp_path / "budget.json"
    estimate = budget_module.estimate_request(request(video_payload(1)))
    original = budget_module.LiveBudget(path)
    original.reserve("existing-experiment", estimate)
    original_bytes = path.read_bytes()
    network_calls = []
    meter = budget_module.BudgetTransport(
        budget_module.LiveBudget(path, max_calls=12),
        httpx.MockTransport(lambda req: network_calls.append(req)),
        max_images=12,
    )
    meter.phase = "new-experiment"
    with pytest.raises(budget_module.BudgetError, match="refusing to reset"):
        meter.handle_request(request())
    assert not network_calls
    assert path.read_bytes() == original_bytes
    assert len(original.snapshot()["entries"]) == 1


def test_explicit_image_cap_permits_timestamp_blocks_but_remains_bounded(budget_module):
    twelve = video_payload()
    assert len(twelve["messages"][1]["content"]) == 26
    estimate = budget_module.estimate_request(request(twelve), max_images=12)
    assert estimate["image_count"] == 12
    assert estimate["image_token_upper_bound"] == 12 * 1024
    eight = video_payload(8)
    assert len(eight["messages"][1]["content"]) > 17
    assert budget_module.estimate_request(request(eight), max_images=8)["image_count"] == 8
    with pytest.raises(budget_module.BudgetError):
        budget_module.estimate_request(request(video_payload(13)), max_images=12)
    twelve["messages"][1]["content"].append({"type": "text", "text": "extra"})
    with pytest.raises(budget_module.BudgetError):
        budget_module.estimate_request(request(twelve), max_images=12)


def test_image_count_is_global_across_multiple_messages(budget_module):
    payload = video_payload(7)
    payload["messages"].append(video_payload(6)["messages"][1])
    assert all(len(item["content"]) <= 26 for item in payload["messages"][1:])
    with pytest.raises(budget_module.BudgetError):
        budget_module.estimate_request(request(payload), max_images=12)


def test_twelve_images_do_not_charge_base64_as_text(budget_module):
    small = budget_module.estimate_request(request(), max_images=12)
    large = budget_module.estimate_request(
        request(video_payload(image_bytes=b"fixture-pixels" * 10000)), max_images=12
    )
    assert small["text_utf8_bytes_upper_bound"] == large["text_utf8_bytes_upper_bound"]
    assert small["input_token_upper_bound"] == large["input_token_upper_bound"]
    assert small["request_sha256"] != large["request_sha256"]
    assert small["reserved_nano_usd"] == (
        small["input_token_upper_bound"] * budget_module.INPUT_NANO_USD_PER_TOKEN
        + 1024 * budget_module.OUTPUT_NANO_USD_PER_TOKEN
    )


def test_transport_requires_explicit_image_opt_in_before_reservation(budget_module, tmp_path):
    ledger = budget_module.LiveBudget(tmp_path / "budget.json", max_calls=12)
    calls = []

    def respond(req):
        calls.append(req)
        return response_with_usage(req)

    default_meter = budget_module.BudgetTransport(ledger, httpx.MockTransport(respond))
    default_meter.phase = "video:default"
    with pytest.raises(budget_module.BudgetError):
        default_meter.handle_request(request())
    assert not calls and not ledger.snapshot()["entries"]
    opted_in = budget_module.BudgetTransport(ledger, httpx.MockTransport(respond), max_images=12)
    opted_in.phase = "video:explicit"
    opted_in.handle_request(request())
    assert len(calls) == 1
    assert ledger.snapshot()["entries"][0]["image_count"] == 12


def test_twelve_call_lifetime_includes_http_failure_timeout_and_crash(budget_module, tmp_path):
    path = tmp_path / "budget.json"
    calls = []
    small_request = request(video_payload(1))

    def respond(req):
        calls.append(req)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": "temporary failure"})
        if len(calls) == 2:
            raise httpx.ReadTimeout("transport failed", request=req)
        return response_with_usage(req)

    # A reservation persisted just before a process crash consumes one of the twelve slots.
    budget_module.LiveBudget(path, max_calls=12).reserve(
        "crashed", budget_module.estimate_request(small_request)
    )
    for index in range(11):
        ledger = budget_module.LiveBudget(path, max_calls=12)
        meter = budget_module.BudgetTransport(ledger, httpx.MockTransport(respond))
        meter.phase = f"resumed:{index}"
        if index == 1:
            with pytest.raises(httpx.ReadTimeout):
                meter.handle_request(small_request)
        else:
            meter.handle_request(small_request)
    assert len(calls) == 11
    entries = ledger.snapshot()["entries"]
    assert len(entries) == 12
    assert entries[0]["state"] == "reserved"
    assert entries[1]["http_status"] == 503
    assert entries[2]["state"] == "transport_error"
    assert all(entry["reserved_nano_usd"] > 0 for entry in entries)
    meter.phase = "thirteenth"
    with pytest.raises(budget_module.BudgetError, match="exhausted"):
        meter.handle_request(small_request)
    assert len(calls) == 11


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"usage": {}},
        {
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }
        },
    ],
)
def test_missing_or_zero_usage_does_not_refund_reservation(budget_module, tmp_path, body):
    path = tmp_path / "budget.json"
    meter = budget_module.BudgetTransport(
        budget_module.LiveBudget(path, max_calls=12),
        httpx.MockTransport(lambda req: httpx.Response(200, json=body)),
        max_images=12,
    )
    meter.phase = "attempted"
    estimate = budget_module.estimate_request(request(), max_images=12)
    meter.handle_request(request())
    entry = budget_module.LiveBudget(path, max_calls=12).snapshot()["entries"][0]
    assert entry["reserved_nano_usd"] == estimate["reserved_nano_usd"]
    with pytest.raises(budget_module.BudgetError):
        meter.handle_request(request())


def test_extended_call_cap_does_not_increase_five_cent_cost_budget(budget_module, tmp_path):
    ledger = budget_module.LiveBudget(tmp_path / "budget.json", max_calls=12)
    assert ledger.snapshot()["max_cost_nano_usd"] == 50_000_000  # USD 0.05, not USD 0.50.
    expensive_request = request(video_payload(1, text="x" * 100000))
    estimate = budget_module.estimate_request(expensive_request)
    assert 0.025 < estimate["reserved_cost_usd"] < 0.05
    plan = [{"phase": f"planned:{index}", **estimate} for index in range(2)]
    with pytest.raises(budget_module.BudgetError, match="Planned experiment"):
        ledger.check_plan(plan)
    assert not ledger.snapshot()["entries"]
    meter = budget_module.BudgetTransport(ledger, httpx.MockTransport(response_with_usage))
    meter.phase = "first"
    meter.handle_request(expensive_request)
    meter.phase = "second"
    with pytest.raises(budget_module.BudgetError, match="exhausted"):
        meter.handle_request(expensive_request)
    assert len(ledger.snapshot()["entries"]) == 1


def test_overrun_stops_entire_remaining_twelve_call_plan(budget_module, tmp_path):
    ledger = budget_module.LiveBudget(tmp_path / "budget.json", max_calls=12)
    meter = budget_module.BudgetTransport(
        ledger,
        httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                json={
                    "usage": {
                        "prompt_tokens": 100000,
                        "completion_tokens": 1025,
                    }
                },
            )
        ),
        max_images=12,
    )
    meter.phase = "overrun"
    meter.handle_request(request())
    assert ledger.snapshot()["entries"][0]["bound_exceeded"] is True
    plan = [{"phase": "next", **budget_module.estimate_request(request(), max_images=12)}]
    with pytest.raises(budget_module.BudgetError):
        ledger.check_plan(plan)
    meter.phase = "next"
    with pytest.raises(budget_module.BudgetError):
        meter.handle_request(request())


def test_reserved_cost_is_immutable_when_recording_evaluation(budget_module, tmp_path):
    ledger = budget_module.LiveBudget(tmp_path / "budget.json", max_calls=12)
    estimate = budget_module.estimate_request(request(), max_images=12)
    identifier = ledger.reserve("baseline", estimate)
    with pytest.raises(budget_module.BudgetError, match="immutable"):
        ledger.update(identifier, {"reserved_nano_usd": 0})
    saved = json.loads(ledger.path.read_text())
    assert saved["entries"][0]["reserved_nano_usd"] == estimate["reserved_nano_usd"]
