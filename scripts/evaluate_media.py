"""Prepare frozen real-media comparisons entirely locally; no network or paid model calls."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path

import httpx
from evaluate_images import peak_rss_bytes
from live_images import estimate_request
from media_assets import load_manifest, prepare_case, sha256_file

from fishkeeping.config import Settings
from fishkeeping.image_chat import DeepSeekImageChat, compact_evidence
from fishkeeping.models import Scope
from fishkeeping.video import VideoProcessor
from fishkeeping.video_models import VideoRequest
from fishkeeping.vision import ImageProcessor
from fishkeeping.vision_models import ImageRequest


def manifest_hash(manifest):
    return hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prepare(output_dir: Path):
    manifest = load_manifest()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "fixture_sha256": manifest_hash(manifest),
        "paid_calls": 0,
        "source": "Public licensed assets; no business data",
        "cases": [],
        "token_note": "Preflight bounds are estimates, not usage. Each image <=1024 tokens.",
    }
    captures = []

    def capture(request):
        captures.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "deepseek-flash",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {"answer": "offline preview", "uncertain": True, "evidence_ids": []}
                            )
                        },
                    }
                ],
            },
        )

    image_processor, video_processor = ImageProcessor(), VideoProcessor()
    payload_paths = []
    with httpx.Client(transport=httpx.MockTransport(capture)) as client:
        chat = DeepSeekImageChat(
            Settings(_env_file=None, deepseek_api_key="offline-preview", deepseek_max_tokens=1024),
            client,
        )
        for case in manifest["cases"]:
            path = prepare_case(case)
            raw = path.read_bytes()
            item = {
                "id": case["id"],
                "kind": case["kind"],
                "input_sha256": sha256_file(path),
                "input_bytes": len(raw),
                "variants": [],
            }
            for variant in ("baseline", "optimized"):
                start, cpu = time.perf_counter(), time.process_time()
                extra = {}
                valid_ids = ["image0"]
                if case["kind"] == "photo":
                    detail = "original" if variant == "baseline" else "low"
                    chat.answer_image(raw, case["question"], detail=detail)
                    route = "native_" + detail
                elif case["kind"] == "table":
                    if variant == "baseline":
                        chat.answer_image(raw, case["question"], detail="original")
                        route = "native_original"
                    else:
                        evidence = image_processor.analyze(
                            ImageRequest(
                                scope=Scope(
                                    namespace="real-media-evaluation", user_id="public-fixtures"
                                ),
                                image_base64=base64.b64encode(raw).decode(),
                                question=case["question"],
                                layout="table",
                                max_chars=10000,
                            )
                        )
                        extra = {
                            "ocr_selected": len(evidence.regions),
                            "ocr_total": evidence.total_regions,
                            "omitted_regions": evidence.omitted_regions,
                            "requires_review": evidence.requires_review,
                            "selection": evidence.selection,
                            "audit_chars": evidence.chars_used,
                            "model_text_chars": len(compact_evidence(evidence)),
                        }
                        (output_dir / f"{case['id']}-ocr.json").write_text(
                            evidence.model_dump_json(indent=2), encoding="utf-8"
                        )
                        if not evidence.regions:
                            # An empty atomic table group is not usable OCR. This replacement makes
                            # one native call, never an OCR call followed by an automatic retry.
                            chat.answer_image(raw, case["question"], detail="original")
                            route = "native_fallback_empty_table"
                        else:
                            chat.answer_evidence(evidence, allow_review=True)
                            valid_ids = [region.id for region in evidence.regions]
                            route = "table_ocr_review_experiment"
                else:
                    request = VideoRequest(
                        scope=Scope(namespace="real-media-evaluation", user_id="public-fixtures"),
                        video_base64=base64.b64encode(raw).decode(),
                        question=case["question"],
                        strategy="uniform" if variant == "baseline" else "adaptive",
                        max_frames=12 if variant == "baseline" else 6,
                        start_seconds=0 if variant == "baseline" else case["start_seconds"],
                        end_seconds=None if variant == "baseline" else case["end_seconds"],
                    )
                    evidence = video_processor.analyze(request)
                    times = [f.timestamp_seconds for f in evidence.selected_frames]
                    intervals = case.get("evidence_intervals", [])
                    coverage = [
                        any(interval["start"] <= t <= interval["end"] for t in times)
                        for interval in intervals
                    ]
                    extra = {
                        "selected_frames": len(times),
                        "timestamps_seconds": times,
                        "decode_frames": evidence.decoded_frames,
                        "candidate_count": evidence.candidate_count,
                        "window_candidate_count": evidence.window_candidate_count,
                        "analysis_window": [
                            evidence.analysis_start_seconds,
                            evidence.analysis_end_seconds,
                        ],
                        "target_intervals_covered": sum(coverage),
                        "target_intervals_total": len(coverage),
                        "interval_coverage": coverage,
                        "max_gap_seconds": evidence.max_gap_seconds,
                        "observed_transitions": evidence.observed_transitions,
                        "represented_transitions": evidence.represented_transitions,
                        "requires_review": evidence.requires_review,
                    }
                    (output_dir / f"{case['id']}-{variant}-evidence.json").write_text(
                        evidence.model_dump_json(indent=2), encoding="utf-8"
                    )
                    chat.answer_video(
                        evidence,
                        detail="original" if variant == "baseline" else "low",
                        allow_review=True,
                    )
                    valid_ids = [frame.id for frame in evidence.selected_frames]
                    route = (
                        "uniform_full_video_original"
                        if variant == "baseline"
                        else "question_window_adaptive_low"
                    )
                payload = captures.pop()
                estimate = estimate_request(
                    httpx.Request(
                        "POST", "https://api.deepseek.com/chat/completions", json=payload
                    ),
                    max_images=12,
                )
                phase = f"{case['id']}:{variant}"
                payload_path = output_dir / (phase.replace(":", "-") + "-request.json")
                payload_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                payload_paths.append(
                    {
                        "phase": phase,
                        "path": str(payload_path.resolve()),
                        "request_sha256": estimate["request_sha256"],
                        "estimate": estimate,
                        "valid_ids": valid_ids,
                    }
                )
                item["variants"].append(
                    {
                        "variant": variant,
                        "route": route,
                        **extra,
                        "preparation_wall_seconds": time.perf_counter() - start,
                        "preparation_parent_cpu_seconds": time.process_time() - cpu,
                        "parent_peak_rss_bytes": peak_rss_bytes(),
                        "request_estimate": estimate,
                    }
                )
            report["cases"].append(item)
    report["planned_calls"] = len(payload_paths)
    report["reserved_upper_cost_usd"] = (
        sum(item["estimate"]["reserved_nano_usd"] for item in payload_paths) / 1e9
    )
    (output_dir / "prepared.json").write_text(
        json.dumps(
            {"fixture_sha256": report["fixture_sha256"], "requests": payload_paths},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/media-v04"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/media-v04-offline.json"))
    args = parser.parse_args()
    result = prepare(args.output_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "planned_calls": result["planned_calls"],
                "reserved_upper_cost_usd": result["reserved_upper_cost_usd"],
                "paid_calls": 0,
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
