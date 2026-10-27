"""Explicit six-call synthetic image comparison with a persistent eight-call/$0.05 ceiling.

No response text or credentials are logged. Failed/unknown requests consume reservations;
restarting skips previously reserved phases and never automatically retries a paid request.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import copy
import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx

from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat
from fishkeeping.models import Scope
from fishkeeping.vision import ImageProcessor
from fishkeeping.vision_models import ImageRequest

MAX_CALLS = 8
MAX_OUTPUT_TOKENS = 1024
MAX_COST_NANO_USD = 50_000_000
INPUT_NANO_USD_PER_TOKEN = 300  # Conservative peak cache-miss price: $0.30 / 1M.
OUTPUT_NANO_USD_PER_TOKEN = 1200  # Peak price: $1.20 / 1M.
MAX_IMAGE_TOKENS = 1024
DEFAULT_LEDGER = Path(".tmp/vision-live-budget.json")
FROZEN_FIXTURE_SHA256 = "f1a41e44fa9aeb4ba149c6bf02f4695db516fd100beaa7fbf18691100c5fd7c1"
PHASE_RE = re.compile(r"^[a-zA-Z0-9_.:-]{1,128}$")
USAGE_KEYS = {
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
}


class BudgetError(httpx.TransportError):
    pass


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as file:
            json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _ledger_lock(path: Path):
    """OS locks are released on process exit; stale lock files do not reset the budget."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a+b") as file:
        file.seek(0, os.SEEK_END)
        if not file.tell():
            file.write(b"0")
            file.flush()
        file.seek(0)
        locked = False
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
            yield
        except OSError:
            raise BudgetError("Live budget ledger unavailable or locked") from None
        finally:
            if locked:
                file.seek(0)
                if os.name == "nt":
                    msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(file.fileno(), fcntl.LOCK_UN)


def _unique_json(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate JSON key")
        output[key] = value
    return output


def _reject_constant(value):
    raise ValueError("nonfinite JSON")


def estimate_request(request: httpx.Request, *, max_images: int = 3) -> dict:
    """Validate the complete paid request and bound input without counting Base64 as text."""
    if type(max_images) is not int or not 1 <= max_images <= 12:
        raise BudgetError("Invalid experiment image cap")
    url = request.url
    if (
        request.method != "POST"
        or url.scheme != "https"
        or url.host != "api.deepseek.com"
        or url.port not in {None, 443}
        or url.path != "/chat/completions"
        or url.query
        or url.fragment
        or url.userinfo
    ):
        raise BudgetError("Live request endpoint is not allowed")
    try:
        if len(request.content) > 36_000_000:
            raise ValueError("request too large")
        payload = json.loads(
            request.content, object_pairs_hook=_unique_json, parse_constant=_reject_constant
        )
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {"model", "messages", "thinking", "stream", "response_format", "max_tokens"}
            or payload["model"] != "deepseek-flash"
            or payload["thinking"] != {"type": "disabled"}
            or payload["stream"] is not False
            or payload["response_format"] != {"type": "json_object"}
            or type(payload["max_tokens"]) is not int
            or not 1 <= payload["max_tokens"] <= MAX_OUTPUT_TOKENS
            or not isinstance(payload["messages"], list)
            or not 1 <= len(payload["messages"]) <= 8
        ):
            raise ValueError("invalid paid request")
        sanitized = copy.deepcopy(payload)
        image_count = 0
        for message in sanitized["messages"]:
            if set(message) != {"role", "content"} or message["role"] not in {"system", "user"}:
                raise ValueError("unsupported message")
            content = message["content"]
            if isinstance(content, str):
                continue
            if (
                not isinstance(content, list)
                or not content
                or len(content) > max(8, 2 * max_images + 2)
            ):
                raise ValueError("unsupported content")
            for block in content:
                if block.get("type") == "text":
                    if set(block) != {"type", "text"} or not isinstance(block["text"], str):
                        raise ValueError("invalid text block")
                elif block.get("type") == "image_url":
                    if set(block) != {"type", "image_url"}:
                        raise ValueError("invalid image block")
                    image = block["image_url"]
                    if set(image) != {"url", "detail"} or image["detail"] not in {
                        "low",
                        "original",
                    }:
                        raise ValueError("invalid image detail")
                    if not isinstance(image["url"], str):
                        raise ValueError("invalid data URL")
                    prefix, separator, data = image["url"].partition(",")
                    if separator != "," or prefix not in {
                        "data:image/png;base64",
                        "data:image/jpeg;base64",
                    }:
                        raise ValueError("only embedded PNG/JPEG evidence is allowed")
                    raw = base64.b64decode(data, validate=True)
                    if not raw or len(raw) > 8 * 1024 * 1024:
                        raise ValueError("image too large")
                    image["url"] = ""  # The image gets its own token bound, not Base64 text tokens.
                    image_count += 1
                else:
                    raise ValueError("unsupported content block")
        if image_count > max_images:
            raise ValueError("too many images")
        text_bytes = len(
            json.dumps(sanitized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        framing_tokens = 256 + 32 * len(payload["messages"])
        input_bound = text_bytes + framing_tokens + image_count * MAX_IMAGE_TOKENS
        reserved = (
            input_bound * INPUT_NANO_USD_PER_TOKEN
            + payload["max_tokens"] * OUTPUT_NANO_USD_PER_TOKEN
        )
        return {
            "request_sha256": hashlib.sha256(request.content).hexdigest(),
            "text_utf8_bytes_upper_bound": text_bytes,
            "framing_token_allowance": framing_tokens,
            "image_count": image_count,
            "image_token_upper_bound": image_count * MAX_IMAGE_TOKENS,
            "input_token_upper_bound": input_bound,
            "output_token_upper_bound": payload["max_tokens"],
            "reserved_nano_usd": reserved,
            "reserved_cost_usd": reserved / 1_000_000_000,
        }
    except (ValueError, TypeError, KeyError, AttributeError, binascii.Error):
        raise BudgetError("Live request payload failed validation") from None


class LiveBudget:
    def __init__(self, path: Path = DEFAULT_LEDGER, *, max_calls: int = MAX_CALLS):
        if type(max_calls) is not int or not 1 <= max_calls <= 12:
            raise BudgetError("Invalid experiment call cap")
        self.path = Path(path)
        self.max_calls = max_calls

    def _load(self) -> dict:
        if not self.path.exists():
            return {
                "version": 1,
                "max_calls": self.max_calls,
                "max_cost_nano_usd": MAX_COST_NANO_USD,
                "entries": [],
            }
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
            entries = data["entries"]
            if (
                data["version"] != 1
                or data["max_calls"] != self.max_calls
                or data["max_cost_nano_usd"] != MAX_COST_NANO_USD
                or not isinstance(entries, list)
                or len(entries) > self.max_calls
                or len({entry["phase"] for entry in entries}) != len(entries)
                or any(
                    type(entry["reserved_nano_usd"]) is not int
                    or not 0 < entry["reserved_nano_usd"] <= MAX_COST_NANO_USD
                    for entry in entries
                )
            ):
                raise ValueError("invalid ledger")
            return data
        except (ValueError, TypeError, KeyError, OSError):
            raise BudgetError(
                "Existing live budget ledger is invalid; refusing to reset it"
            ) from None

    def snapshot(self) -> dict:
        with _ledger_lock(self.path):
            return self._load()

    def check_plan(self, plan: list[dict]) -> None:
        with _ledger_lock(self.path):
            entries = self._load()["entries"]
            used = {entry["phase"] for entry in entries}
            pending = [item for item in plan if item["phase"] not in used]
            if (
                len(entries) + len(pending) > self.max_calls
                or sum(item["reserved_nano_usd"] for item in [*entries, *pending])
                > MAX_COST_NANO_USD
                or any(entry.get("bound_exceeded") for entry in entries)
            ):
                raise BudgetError("Planned experiment exceeds remaining persistent budget")

    def reserve(self, phase: str, estimate: dict) -> str:
        if not isinstance(phase, str) or not PHASE_RE.fullmatch(phase):
            raise BudgetError("A bounded experiment phase is required")
        with _ledger_lock(self.path):
            data = self._load()
            entries = data["entries"]
            if (
                len(entries) >= self.max_calls
                or any(entry["phase"] == phase or entry.get("bound_exceeded") for entry in entries)
                or sum(entry["reserved_nano_usd"] for entry in entries)
                + estimate["reserved_nano_usd"]
                > MAX_COST_NANO_USD
            ):
                raise BudgetError("Live phase already attempted or cumulative budget exhausted")
            identifier = uuid.uuid4().hex
            entries.append(
                {
                    "id": identifier,
                    "phase": phase,
                    "state": "reserved",
                    "reserved_at": datetime.now(UTC).isoformat(),
                    **estimate,
                }
            )
            _atomic_json(self.path, data)
            return identifier

    def update(self, identifier: str, fields: dict) -> None:
        with _ledger_lock(self.path):
            data = self._load()
            entry = next((item for item in data["entries"] if item["id"] == identifier), None)
            if entry is None:
                raise BudgetError("Live reservation disappeared")
            if set(fields) & {"id", "phase", "reserved_nano_usd", "request_sha256"}:
                raise BudgetError("Reservation identity and cost are immutable")
            entry.update(fields)
            _atomic_json(self.path, data)


class BudgetTransport(httpx.BaseTransport):
    def __init__(
        self, ledger: LiveBudget, inner: httpx.BaseTransport | None = None, *, max_images: int = 3
    ):
        self.ledger = ledger
        self.inner = inner or httpx.HTTPTransport(retries=0)
        self.phase = "unset"
        self.last_reservation_id = None
        self.max_images = max_images

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            if str(request.url) != "https://api.deepseek.com/models":
                raise BudgetError("Only the official free model-list GET is allowed")
            return self.inner.handle_request(request)
        estimate = estimate_request(request, max_images=self.max_images)
        identifier = self.ledger.reserve(self.phase, estimate)
        self.last_reservation_id = identifier
        start = time.perf_counter()
        fields = {"state": "unknown"}
        try:
            response = self.inner.handle_request(request)
            response.read()
            fields.update({"state": "response_received", "http_status": response.status_code})
            if response.is_success and len(response.content) <= 100_000:
                try:
                    body = response.json()
                    raw_usage = body.get("usage", {})
                    usage = {
                        key: value
                        for key, value in raw_usage.items()
                        if key in USAGE_KEYS and type(value) is int and value >= 0
                    }
                    fields["usage"] = usage
                    if {"prompt_tokens", "completion_tokens"} <= usage.keys():
                        cost = (
                            usage["prompt_tokens"] * INPUT_NANO_USD_PER_TOKEN
                            + usage["completion_tokens"] * OUTPUT_NANO_USD_PER_TOKEN
                        )
                        fields["usage_based_peak_cost_estimate_usd"] = cost / 1_000_000_000
                        fields["bound_exceeded"] = (
                            usage["prompt_tokens"] > estimate["input_token_upper_bound"]
                            or usage["completion_tokens"] > estimate["output_token_upper_bound"]
                        )
                except (ValueError, TypeError, AttributeError):
                    fields["usage_unavailable"] = True
            return response
        except Exception as exc:
            fields.update({"state": "transport_error", "error_type": type(exc).__name__})
            raise
        finally:
            fields["latency_seconds"] = round(time.perf_counter() - start, 4)
            self.ledger.update(identifier, fields)

    def close(self) -> None:
        self.inner.close()


def match_answer(answer: str, terms: list) -> list[bool]:
    def normalize(text):
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    normalized = normalize(answer)
    return [
        any(
            normalize(term) in normalized
            for term in (group if isinstance(group, list) else [group])
        )
        for group in terms
    ]


def _invoke(chat, case, image, evidence, variant):
    if variant == "compact_ocr":
        return chat.answer_evidence(evidence, allow_review=True)
    return chat.answer_image(image, case["question"], detail=variant.removeprefix("native_"))


def preview_plan(prepared: list, settings: Settings) -> list[dict]:
    """Use the real adapter to construct requests, intercepted before any network or reservation."""
    plan = []

    class Captured(RuntimeError):
        pass

    def capture(request):
        plan.append({"phase": phase, **estimate_request(request)})
        raise Captured

    with httpx.Client(transport=httpx.MockTransport(capture)) as client:
        chat = DeepSeekImageChat(settings, client)
        for case, image, evidence in prepared:
            for variant in ["native_original", "native_low" if evidence is None else "compact_ocr"]:
                phase = case["id"] + ":" + variant
                try:
                    _invoke(chat, case, image, evidence, variant)
                except Captured:
                    pass
    return plan


def run(report_path: Path, *, ledger_path: Path = DEFAULT_LEDGER, live: bool = False) -> dict:
    # Imported lazily: ordinary unit tests never generate images or load the OCR model.
    from evaluate_images import generate_case, load_dataset

    ledger = LiveBudget(ledger_path)
    report = {
        "started_at": datetime.now(UTC).isoformat(),
        "status": "failed",
        "synthetic_data_only": True,
        "pricing_snapshot_date": "2026-10-28",
        "price_basis": "Peak cache-miss $0.30/M input and $1.20/M output; conservative estimate",
        "match_method": "All frozen casefolded substring term groups and uncertain=false",
        "max_calls": MAX_CALLS,
        "max_reserved_cost_usd": MAX_COST_NANO_USD / 1_000_000_000,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "skipped_previously_reserved_phases": [],
    }
    meter = None
    try:
        settings = Settings(
            **(
                {}
                if live
                else {
                    "_env_file": None,
                    "deepseek_base_url": "https://api.deepseek.com",
                    "deepseek_api_key": "offline-preview-placeholder",
                }
            ),
            deepseek_model="deepseek-flash",
            deepseek_thinking="disabled",
            deepseek_max_tokens=MAX_OUTPUT_TOKENS,
        )
        if (
            settings.deepseek_base_url != "https://api.deepseek.com"
            or not settings.deepseek_api_key
        ):
            raise ValueError("Official DeepSeek endpoint and configured key required")
        dataset = load_dataset()
        report["fixture_sha256_canonical"] = hashlib.sha256(
            json.dumps(dataset, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if report["fixture_sha256_canonical"] != FROZEN_FIXTURE_SHA256:
            raise ValueError("Frozen image labels changed; refusing live comparison")
        wanted = {"long_screenshot", "water_table", "scene_shapes"}
        cases = [case for case in dataset["cases"] if case["id"] in wanted]
        if len(cases) != 3 or {case["id"] for case in cases} != wanted:
            raise ValueError("Expected exactly two document cases and one scene case")
        prepared = []
        processor = ImageProcessor()
        for case in cases:
            path, metadata = generate_case(case, Path(".tmp/image-evaluation"))
            image = Path(path).read_bytes()
            terms = case["expected_terms"]
            if not terms or not all(
                isinstance(group, list)
                and group
                and all(isinstance(term, str) and term for term in group)
                for group in terms
            ):
                raise ValueError("Frozen nonempty answer terms required")
            evidence = None
            if case["id"] != "scene_shapes":
                evidence = processor.analyze(
                    ImageRequest(
                        scope=Scope(namespace="image-live", user_id="synthetic-user"),
                        image_base64=base64.b64encode(image).decode("ascii"),
                        question=case["question"],
                        max_chars=3000,
                    )
                )
            prepared.append((case, image, evidence))
        report["plan"] = preview_plan(prepared, settings)
        report["planned_reserved_cost_usd"] = (
            sum(item["reserved_nano_usd"] for item in report["plan"]) / 1_000_000_000
        )
        ledger.check_plan(report["plan"])
        _atomic_json(report_path, report)
        if not live:
            report["status"] = "preview"
            return report
        meter = BudgetTransport(ledger)
        with httpx.Client(
            transport=meter, follow_redirects=False, timeout=settings.request_timeout
        ) as client:
            response = client.get(
                "https://api.deepseek.com/models",
                headers={
                    "Authorization": "Bearer " + settings.deepseek_api_key.get_secret_value(),
                },
            )
            response.raise_for_status()
            if len(response.content) > 100_000:
                raise ValueError("Model listing too large")
            report["flash_available"] = any(
                entry.get("id") == "deepseek-flash" for entry in response.json()["data"]
            )
            if not report["flash_available"]:
                raise ValueError("Flash unavailable for this account")
            chat = DeepSeekImageChat(settings, client)
            used = {entry["phase"] for entry in ledger.snapshot()["entries"]}
            for case, image, evidence in prepared:
                variants = ["native_original", "native_low" if evidence is None else "compact_ocr"]
                for variant in variants:
                    phase = case["id"] + ":" + variant
                    if phase in used:
                        report["skipped_previously_reserved_phases"].append(phase)
                        continue
                    meter.phase = phase
                    meter.last_reservation_id = None
                    try:
                        answer = _invoke(chat, case, image, evidence, variant)
                        matched = match_answer(answer.answer, case["expected_terms"])
                        ledger.update(
                            meter.last_reservation_id,
                            {
                                "answer_term_groups_matched": sum(matched),
                                "answer_term_groups_total": len(matched),
                                "answer_correct": all(matched) and not answer.uncertain,
                                "uncertain": answer.uncertain,
                                "ocr_requires_review": evidence.requires_review
                                if evidence
                                else None,
                                "image_sha256": hashlib.sha256(image).hexdigest(),
                            },
                        )
                    except Exception as exc:
                        if meter.last_reservation_id is not None:
                            ledger.update(
                                meter.last_reservation_id,
                                {
                                    "answer_evaluation_error_type": type(exc).__name__,
                                },
                            )
                        raise  # No automatic retry or fall-through after an unexpected failure.
                    finally:
                        report["ledger"] = ledger.snapshot()
                        _atomic_json(report_path, report)
        evaluated = {
            entry["phase"] for entry in ledger.snapshot()["entries"] if "answer_correct" in entry
        }
        report["status"] = (
            "completed"
            if all(item["phase"] in evaluated for item in report["plan"])
            else "incomplete"
        )
    except Exception as exc:
        report["error_type"] = type(exc).__name__
    finally:
        if meter is not None:
            meter.close()
        try:
            report["ledger"] = ledger.snapshot()
        except BudgetError:
            report["ledger_unavailable"] = True
        report["completed_at"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--live", action="store_true", help="Explicitly enable paid synthetic checks")
    mode.add_argument(
        "--preview", action="store_true", help="Prepare estimates without any network"
    )
    parser.add_argument("--report", type=Path, default=Path(".tmp/images-live.json"))
    args = parser.parse_args()
    report = run(args.report, live=args.live)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(args.report),
                "total_reserved_calls": len(report.get("ledger", {}).get("entries", [])),
            }
        )
    )
    if report["status"] not in {"completed", "preview"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
