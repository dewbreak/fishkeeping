import base64
import importlib.util
import json
from pathlib import Path

import httpx
import pytest


@pytest.fixture
def live_images():
    path = Path(__file__).resolve().parents[1] / "scripts/live_images.py"
    spec = importlib.util.spec_from_file_location("live_images", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def payload(text="What is shown?", image_bytes=None):
    blocks = [{"type": "text", "text": text}]
    if image_bytes is not None:
        blocks.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image_bytes).decode(),
                    "detail": "original",
                },
            }
        )
    return {
        "model": "deepseek-flash",
        "messages": [
            {"role": "system", "content": "Return JSON."},
            {"role": "user", "content": blocks},
        ],
        "max_tokens": 1024,
        "thinking": {"type": "disabled"},
        "stream": False,
        "response_format": {"type": "json_object"},
    }


def request(body=None, url="https://api.deepseek.com/chat/completions"):
    return httpx.Request("POST", url, json=body or payload())


def success(request):
    return httpx.Response(
        200,
        json={
            "usage": {"prompt_tokens": 200, "completion_tokens": 20, "total_tokens": 220},
            "choices": [{"message": {"content": "PRIVATE RAW RESPONSE"}}],
        },
    )


def test_reservation_persisted_before_request_and_restart_preserves_limit(live_images, tmp_path):
    path = tmp_path / "budget.json"
    calls = []

    def inspect(req):
        saved = json.loads(path.read_text())
        assert saved["entries"][-1]["state"] == "reserved"
        assert saved["entries"][-1]["reserved_nano_usd"] > 0
        calls.append(req)
        return success(req)

    for index in range(8):
        meter = live_images.BudgetTransport(
            live_images.LiveBudget(path), httpx.MockTransport(inspect)
        )
        meter.phase = f"phase-{index}"
        meter.handle_request(request())
        meter.close()
    meter = live_images.BudgetTransport(live_images.LiveBudget(path), httpx.MockTransport(inspect))
    meter.phase = "ninth"
    with pytest.raises(live_images.BudgetError, match="exhausted"):
        meter.handle_request(request())
    assert len(calls) == 8
    saved = json.loads(path.read_text())
    assert len(saved["entries"]) == 8
    assert all(entry["state"] == "response_received" for entry in saved["entries"])
    assert "PRIVATE RAW RESPONSE" not in path.read_text()


def test_failed_request_consumes_budget_no_retry_or_phase_replay(live_images, tmp_path):
    path = tmp_path / "budget.json"
    calls = []

    def fail(req):
        calls.append(req)
        raise httpx.ReadTimeout("private upstream details", request=req)

    meter = live_images.BudgetTransport(live_images.LiveBudget(path), httpx.MockTransport(fail))
    meter.phase = "failed"
    with pytest.raises(httpx.ReadTimeout):
        meter.handle_request(request())
    resumed = live_images.BudgetTransport(live_images.LiveBudget(path), httpx.MockTransport(fail))
    resumed.phase = "failed"
    with pytest.raises(live_images.BudgetError):
        resumed.handle_request(request())
    assert len(calls) == 1
    entry = resumed.ledger.snapshot()["entries"][0]
    assert entry["state"] == "transport_error"
    assert entry["reserved_cost_usd"] > 0
    assert "private upstream details" not in path.read_text()


def test_unknown_crashed_reservation_stays_consumed(live_images, tmp_path):
    path = tmp_path / "budget.json"
    ledger = live_images.LiveBudget(path)
    ledger.reserve("interrupted", live_images.estimate_request(request()))
    restarted = live_images.LiveBudget(path)
    assert restarted.snapshot()["entries"][0]["state"] == "reserved"
    with pytest.raises(live_images.BudgetError):
        restarted.reserve("interrupted", live_images.estimate_request(request()))


def test_cost_ceiling_independent_of_call_count(live_images, tmp_path):
    path = tmp_path / "budget.json"
    meter = live_images.BudgetTransport(live_images.LiveBudget(path), httpx.MockTransport(success))
    # Roughly $0.031 per request; the second would exceed the cumulative $0.05 bound.
    body = payload(text="x" * 100_000)
    meter.phase = "large-first"
    meter.handle_request(request(body))
    meter.phase = "large-second"
    with pytest.raises(live_images.BudgetError):
        meter.handle_request(request(body))
    entries = meter.ledger.snapshot()["entries"]
    assert len(entries) == 1
    assert entries[0]["reserved_nano_usd"] <= live_images.MAX_COST_NANO_USD


def test_images_use_separate_token_bound_not_base64_text(live_images):
    small = live_images.estimate_request(request(payload(image_bytes=b"image")))
    large = live_images.estimate_request(request(payload(image_bytes=b"image" * 100_000)))
    assert small["text_utf8_bytes_upper_bound"] == large["text_utf8_bytes_upper_bound"]
    assert small["input_token_upper_bound"] == large["input_token_upper_bound"]
    assert small["image_token_upper_bound"] == 1024
    text = live_images.estimate_request(request(payload(text="中文")))
    english = live_images.estimate_request(request(payload(text="ab")))
    assert text["text_utf8_bytes_upper_bound"] - english["text_utf8_bytes_upper_bound"] == 4


@pytest.mark.parametrize(
    "change",
    [
        {"model": "deepseek-pro"},
        {"thinking": {"type": "enabled"}},
        {"max_tokens": 1025},
        {"max_tokens": True},
        {"stream": True},
        {"tools": []},
        {"messages": []},
        {"response_format": {"type": "text"}},
        {"messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": "x"}]}]},
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.png", "detail": "low"},
                        }
                    ],
                }
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,%%%", "detail": "low"},
                        }
                    ],
                }
            ]
        },
    ],
)
def test_invalid_payload_never_reaches_network_or_consumes_budget(live_images, tmp_path, change):
    calls = []
    meter = live_images.BudgetTransport(
        live_images.LiveBudget(tmp_path / "budget.json"),
        httpx.MockTransport(lambda req: calls.append(req)),
    )
    with pytest.raises(live_images.BudgetError):
        meter.handle_request(request({**payload(), **change}))
    assert not calls
    assert not meter.ledger.snapshot()["entries"]


@pytest.mark.parametrize(
    "url",
    [
        "http://api.deepseek.com/chat/completions",
        "https://other.example/chat/completions",
        "https://api.deepseek.com:444/chat/completions",
        "https://api.deepseek.com/chat/completions?q=1",
        "https://username:password@api.deepseek.com/chat/completions",
    ],
)
def test_endpoint_validation(live_images, url):
    with pytest.raises(live_images.BudgetError):
        live_images.estimate_request(request(url=url))


def test_free_models_check_is_only_allowed_get(live_images, tmp_path):
    meter = live_images.BudgetTransport(
        live_images.LiveBudget(tmp_path / "budget.json"),
        httpx.MockTransport(lambda req: httpx.Response(200, json={"data": []})),
    )
    assert (
        meter.handle_request(httpx.Request("GET", "https://api.deepseek.com/models")).status_code
        == 200
    )
    assert not meter.ledger.snapshot()["entries"]
    with pytest.raises(live_images.BudgetError):
        meter.handle_request(httpx.Request("GET", "https://api.deepseek.com/other"))


def test_corrupt_ledger_fails_closed(live_images, tmp_path):
    path = tmp_path / "budget.json"
    path.write_text('{"version":1,"entries": []}')
    with pytest.raises(live_images.BudgetError, match="refusing to reset"):
        live_images.LiveBudget(path).snapshot()


def test_actual_usage_exceeding_bound_blocks_followup(live_images, tmp_path):
    meter = live_images.BudgetTransport(
        live_images.LiveBudget(tmp_path / "budget.json"),
        httpx.MockTransport(
            lambda req: httpx.Response(
                200, json={"usage": {"prompt_tokens": 100_000, "completion_tokens": 2000}}
            )
        ),
    )
    meter.phase = "first"
    meter.handle_request(request())
    meter.phase = "blocked"
    with pytest.raises(live_images.BudgetError):
        meter.handle_request(request())
    assert meter.ledger.snapshot()["entries"][0]["bound_exceeded"]


def test_answer_matching_frozen_groups_and_normalization(live_images):
    assert live_images.match_answer("Temperature: 25 ℃; pH 6.8", [["25 ℃", "25 C"], "6.8"]) == [
        True,
        True,
    ]
    assert live_images.match_answer("unknown", ["25", "6.8"]) == [False, False]


def test_whole_plan_checked_before_reserving_any_call(live_images, tmp_path):
    ledger = live_images.LiveBudget(tmp_path / "budget.json")
    estimate = live_images.estimate_request(request())
    with pytest.raises(live_images.BudgetError, match="Planned experiment"):
        ledger.check_plan([{"phase": f"phase-{i}", **estimate} for i in range(9)])
    assert not ledger.snapshot()["entries"]
    expensive = live_images.estimate_request(request(payload(text="x" * 100_000)))
    with pytest.raises(live_images.BudgetError, match="Planned experiment"):
        ledger.check_plan([{"phase": f"phase-{i}", **expensive} for i in range(2)])
    assert not ledger.snapshot()["entries"]


def test_preview_uses_adapter_without_network_or_ledger(live_images, monkeypatch):
    import socket

    from fishkeeping.vision_models import ImageEvidence

    def fail_network(*args, **kwargs):
        pytest.fail("preview used network")

    monkeypatch.setattr(socket.socket, "connect", fail_network)
    monkeypatch.setattr(
        "fishkeeping.image_chat.native_blocks",
        lambda *args, **kwargs: payload(image_bytes=b"synthetic")["messages"][1]["content"][1:],
    )
    evidence = ImageEvidence(
        image_sha256="a" * 64,
        width=10,
        height=10,
        method="fake-ocr",
        method_version="1",
        question="What?",
        regions=[],
        total_regions=0,
        omitted_regions=0,
        requires_review=True,
        warnings=[],
        packed="Untrusted synthetic evidence: 25 C",
        chars_used=34,
    )
    prepared = [
        ({"id": "document", "question": "What?"}, b"synthetic", evidence),
        ({"id": "scene", "question": "How many?"}, b"synthetic", None),
    ]
    settings = live_images.Settings(_env_file=None, deepseek_api_key="not-a-real-secret")
    plan = live_images.preview_plan(prepared, settings)
    assert [item["phase"] for item in plan] == [
        "document:native_original",
        "document:compact_ocr",
        "scene:native_original",
        "scene:native_low",
    ]
    assert [item["image_count"] for item in plan] == [1, 0, 1, 1]
    assert all(item["reserved_nano_usd"] > 0 for item in plan)
    assert "not-a-real-secret" not in json.dumps(plan)
