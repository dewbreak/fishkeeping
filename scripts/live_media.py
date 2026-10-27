"""Explicit frozen real-media experiment; <=12 calls/$0.05, no automatic paid retries.

Run evaluate_media.py first. Answers stay in the ignored work directory for manual semantic
review; the report contains usage and diagnostic term matches, never credentials or image bytes.
Identical requests reuse an existing successful answer and do not reserve another paid call.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
from evaluate_media import manifest_hash
from live_images import (
    MAX_COST_NANO_USD,
    BudgetError,
    BudgetTransport,
    LiveBudget,
    _atomic_json,
    estimate_request,
    match_answer,
)
from media_assets import load_manifest

from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat

FROZEN_MANIFEST_SHA256 = "255a9affb46eed171c6414da3aa4cb746cf8c9182a1a06590ea272bb6caf7446"
DEFAULT_LEDGER = Path(".tmp/media-v04-budget.json")


def load_plan(prepared_path: Path) -> tuple[dict, list[dict]]:
    manifest = load_manifest()
    digest = manifest_hash(manifest)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if digest != FROZEN_MANIFEST_SHA256 or prepared["fixture_sha256"] != digest:
        raise ValueError("Frozen fixture changed; refusing the live experiment")
    cases = {case["id"]: case for case in manifest["cases"]}
    expected = {f"{case}:{variant}" for case in cases for variant in ("baseline", "optimized")}
    requests = prepared["requests"]
    if len(requests) != 12 or {item["phase"] for item in requests} != expected:
        raise ValueError("Exactly the frozen six paired cases are required")
    plan = []
    for item in requests:
        payload = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        if (
            len(payload.get("messages", [])) != 2
            or [message.get("role") for message in payload["messages"]] != ["system", "user"]
            or not isinstance(payload["messages"][0].get("content"), str)
            or not isinstance(payload["messages"][1].get("content"), list)
            or payload.get("max_tokens") != 1024
        ):
            raise ValueError("Prepared payload must match the adapter's exact message structure")
        estimate = estimate_request(
            httpx.Request("POST", "https://api.deepseek.com/chat/completions", json=payload),
            max_images=12,
        )
        if estimate != item["estimate"] or estimate["request_sha256"] != item["request_sha256"]:
            raise ValueError("Prepared request changed; run offline preparation again")
        question = payload["messages"][1]["content"][0]["text"]
        if question != cases[item["phase"].split(":")[0]]["question"]:
            raise ValueError("Question does not match the frozen fixture")
        rebuilt = {
            "model": payload["model"],
            "messages": payload["messages"],
            "response_format": payload["response_format"],
            "stream": payload["stream"],
            "thinking": payload["thinking"],
            "max_tokens": payload["max_tokens"],
        }
        actual_estimate = estimate_request(
            httpx.Request("POST", "https://api.deepseek.com/chat/completions", json=rebuilt),
            max_images=12,
        )
        if actual_estimate != estimate:
            raise ValueError("Prepared request byte order differs from the adapter")
        valid_ids = item["valid_ids"]
        if not valid_ids or not all(isinstance(value, str) for value in valid_ids):
            raise ValueError("Prepared evidence IDs are required")
        plan.append({**item, "payload": payload, **estimate})
    return manifest, plan


def unique_plan(plan: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in plan:
        if item["request_sha256"] not in seen:
            seen.add(item["request_sha256"])
            result.append(item)
    return result


def summarize(plan, answers, cases):
    results = []
    for item in plan:
        saved = answers.get(item["phase"])
        if not saved:
            continue
        case = cases[item["phase"].split(":")[0]]
        answer = saved["answer"]
        matched = match_answer(answer["answer"], case["expected_any_groups"])
        results.append(
            {
                "phase": item["phase"],
                "request_sha256": item["request_sha256"],
                "reused_from": saved.get("reused_from"),
                "term_groups_matched": sum(matched),
                "term_groups_total": len(matched),
                "term_matches": matched,
                "uncertain": answer["uncertain"],
                "evidence_ids": answer["evidence_ids"],
                "usage": answer["usage"],
                "manual_review_required": True,
            }
        )
    return results


def run(prepared_path: Path, report_path: Path, *, ledger_path=DEFAULT_LEDGER, live=False):
    manifest, plan = load_plan(prepared_path)
    unique = unique_plan(plan)
    ledger = LiveBudget(ledger_path, max_calls=12)
    current = {item["phase"]: item for item in plan}
    reserved_entries = ledger.snapshot()["entries"]
    for entry in reserved_entries:
        if (
            entry["phase"] not in current
            or entry["request_sha256"] != current[entry["phase"]]["request_sha256"]
        ):
            raise BudgetError("Reserved request differs from the current prepared plan")
    reserved_digests = {entry["request_sha256"] for entry in reserved_entries}
    ledger.check_plan([item for item in unique if item["request_sha256"] not in reserved_digests])
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "preview",
        "fixture_sha256": FROZEN_MANIFEST_SHA256,
        "planned_variants": len(plan),
        "unique_requests": len(unique),
        "max_calls": 12,
        "max_reserved_cost_usd": MAX_COST_NANO_USD / 1e9,
        "planned_reserved_cost_usd": sum(p["reserved_nano_usd"] for p in unique) / 1e9,
        "pricing_snapshot_date": "2026-10-28",
        "price_basis": "Peak cache-miss $0.30/M input and $1.20/M output; not a billing receipt",
        "score_note": "Term matches are diagnostics, not correctness. Manually review semantics, "
        "negation, table associations/units, and temporal order against frozen references.",
        "video_comparison": "Full-video uniform12/original versus explicit-question-window "
        "adaptive6/low; a combined-policy comparison, not an isolated sampler ablation.",
        "skipped_reserved_phases": [],
    }
    _atomic_json(report_path, report)
    if not live:
        return report
    settings = Settings(deepseek_model="deepseek-flash", deepseek_max_tokens=1024)
    if settings.deepseek_base_url != "https://api.deepseek.com" or not settings.deepseek_api_key:
        raise ValueError("Official endpoint and configured key required")
    answers_path = prepared_path.parent / "answers.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else {}
    # Saved results are bound to immutable ledger reservations, never a reason to erase a ledger.
    entries = ledger.snapshot()["entries"]
    by_phase = {entry["phase"]: entry for entry in entries}
    for phase, saved in answers.items():
        original = saved.get("reused_from") or phase
        entry = by_phase.get(original)
        if (
            phase not in current
            or original not in current
            or entry is None
            or entry["request_sha256"] != saved["request_sha256"]
            or current[phase]["request_sha256"] != saved["request_sha256"]
            or current[original]["request_sha256"] != saved["request_sha256"]
        ):
            raise BudgetError("Answer cache has no matching budget reservation")
    meter = BudgetTransport(ledger, max_images=12)
    cases = {case["id"]: case for case in manifest["cases"]}
    try:
        with httpx.Client(
            transport=meter, follow_redirects=False, timeout=settings.request_timeout
        ) as client:
            response = client.get(
                "https://api.deepseek.com/models",
                headers={"Authorization": "Bearer " + settings.deepseek_api_key.get_secret_value()},
            )
            response.raise_for_status()
            if len(response.content) > 100_000 or not any(
                item.get("id") == "deepseek-flash" for item in response.json()["data"]
            ):
                raise ValueError("Flash unavailable")
            chat = DeepSeekImageChat(settings, client)
            first_by_digest = {entry["request_sha256"]: entry["phase"] for entry in entries}
            for item in plan:
                phase, digest = item["phase"], item["request_sha256"]
                original = first_by_digest.setdefault(digest, phase)
                if phase in answers:
                    continue
                if original != phase:
                    if original in answers:
                        answers[phase] = {**answers[original], "reused_from": original}
                        _atomic_json(answers_path, answers)
                    else:
                        report["skipped_reserved_phases"].append(phase)
                    continue
                if phase in by_phase:
                    report["skipped_reserved_phases"].append(phase)
                    continue  # Reserved/unknown failures are never automatically retried.
                meter.phase = phase
                payload = item["payload"]
                blocks = payload["messages"][1]["content"]
                answer = chat._answer(
                    blocks[0]["text"],
                    blocks[1:],
                    set(item["valid_ids"]),
                    instruction=payload["messages"][0]["content"],
                )
                actual = next(
                    entry
                    for entry in ledger.snapshot()["entries"]
                    if entry["id"] == meter.last_reservation_id
                )
                if actual["request_sha256"] != digest:
                    raise BudgetError("Sent request differs from the prepared request")
                ledger.update(meter.last_reservation_id, {"answer_validation": "accepted"})
                answers[phase] = {"request_sha256": digest, "answer": answer.model_dump()}
                _atomic_json(answers_path, answers)
                report["results"] = summarize(plan, answers, cases)
                report["ledger"] = ledger.snapshot()
                _atomic_json(report_path, report)
        report["status"] = "complete" if len(answers) == len(plan) else "incomplete_no_retry"
    except Exception as exc:
        report["status"] = "failed_no_retry"
        report["error_type"] = type(exc).__name__
        report["error_code"] = getattr(exc, "code", None)
        report["failed_phase"] = meter.phase
        raise
    finally:
        report["results"] = summarize(plan, answers, cases)
        report["ledger"] = ledger.snapshot()
        entries = report["ledger"]["entries"]
        report["attempted_paid_calls"] = len(entries)
        report["unknown_usage_calls"] = sum(
            "usage_based_peak_cost_estimate_usd" not in entry for entry in entries
        )
        report["known_usage_peak_cost_subtotal_usd"] = sum(
            entry.get("usage_based_peak_cost_estimate_usd", 0) for entry in entries
        )
        report["usage_based_peak_cost_estimate_usd"] = (
            None if report["unknown_usage_calls"] else report["known_usage_peak_cost_subtotal_usd"]
        )
        report["reserved_cost_usd"] = sum(entry["reserved_nano_usd"] for entry in entries) / 1e9
        _atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=Path(".tmp/media-v04/prepared.json"))
    parser.add_argument("--report", type=Path, default=Path(".tmp/media-v04-live.json"))
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    result = run(args.prepared, args.report, ledger_path=args.ledger, live=args.live)
    print(
        json.dumps(
            {key: result[key] for key in ("status", "unique_requests", "planned_reserved_cost_usd")}
        )
    )


if __name__ == "__main__":
    main()
