"""Compare video sampling locally on cached, hash-verified media; no model/network calls.

Each unique video is decoded once. Selection receives only the explicit question
window and frame budget; frozen evidence intervals are scored after all selections.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

from media_assets import DEFAULT_CACHE, load_manifest, prepare_case

from fishkeeping.models import Scope
from fishkeeping.video import DecodedVideo, OpenCVDecoder, VideoProcessor
from fishkeeping.video_models import VideoRequest

CONDITIONS = (
    ("window_uniform6", "window", "uniform", 6),
    ("window_adaptive6", "window", "adaptive", 6),
    ("full_uniform6", "full", "uniform", 6),
    ("full_adaptive6", "full", "adaptive", 6),
    ("window_uniform12", "window", "uniform", 12),
)


class ReusedDecoder:
    """Return the same decoded candidates only for the verified original bytes."""

    def __init__(self, source_sha256: str, decoded: DecodedVideo):
        self.source_sha256 = source_sha256
        self.decoded = decoded
        self.requests = 0

    def decode(self, raw: bytes) -> DecodedVideo:
        if hashlib.sha256(raw).hexdigest() != self.source_sha256:
            raise ValueError("Reused decoder source hash mismatch")
        self.requests += 1
        return self.decoded


def score_intervals(timestamps: list[float], intervals: list[dict]) -> list[dict]:
    """Strict closed intervals, with no rounding, tolerance, or label expansion."""
    return [
        {
            **interval,
            "matched_timestamps_seconds": [
                timestamp
                for timestamp in timestamps
                if interval["start"] <= timestamp <= interval["end"]
            ],
            "hit": any(
                interval["start"] <= timestamp <= interval["end"] for timestamp in timestamps
            ),
        }
        for interval in intervals
    ]


def run_ablation(cache_dir: Path = DEFAULT_CACHE) -> dict:
    manifest = load_manifest()
    manifest_bytes = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    videos = [case for case in manifest["cases"] if case["kind"] == "video"]
    report = {
        "schema_version": 1,
        "fixture_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "labels_frozen_on": manifest["frozen_on"],
        "paid_calls": 0,
        "network_calls": 0,
        "decode_calls": 0,
        "sources": [],
        "cases": [],
        "interpretation": [
            "This is a deterministic local sampling ablation, not a model accuracy evaluation.",
            "Two questions use one animated trailer; they are not independent video sources.",
            "Only question-specified start/end bounds are selection inputs; frozen evidence "
            "intervals and expected sequence are never passed to the selector.",
            "A strict interval hit means a selected PTS falls in the unchanged closed interval. "
            "It does not prove the event is visible, complete, understood, or correctly ordered.",
            "No boundary tolerance or rounding is applied before interval scoring.",
            "Within a fixed window and six-frame cap, uniform/adaptive isolates strategy; "
            "full/window comparison changes temporal scope and cannot prove full-video recall.",
            "The twelve-frame window condition is a separate budget control; actual frames "
            "may be fewer than the cap when there are fewer candidates.",
            "Transition counts are color-change heuristics, not semantic events; "
            "requires_review=false does not guarantee strict interval coverage.",
            "No audio is decoded; fast events between candidate probes can still be missed.",
        ],
    }
    decoded_sources: dict[str, ReusedDecoder] = {}
    # Produce every selection before consulting fine-grained scoring labels.
    selections = []
    for case in videos:
        source = prepare_case(case, cache_dir, manifest=manifest)
        raw = source.read_bytes()
        source_sha256 = hashlib.sha256(raw).hexdigest()
        if source_sha256 not in decoded_sources:
            decoded = OpenCVDecoder().decode(raw)
            decoded_sources[source_sha256] = ReusedDecoder(source_sha256, decoded)
            asset = manifest["assets"][case["asset"]]
            report["sources"].append(
                {
                    "asset": case["asset"],
                    "sha256": source_sha256,
                    "bytes": len(raw),
                    "url": asset["url"],
                    "license": asset["license"],
                    "attribution": asset.get("attribution", asset["author"]),
                    "decoder_version": decoded.decoder_version,
                    "decoded_frames": decoded.decoded_frames,
                    "duration_seconds": decoded.duration_seconds,
                    "container_duration_seconds": decoded.container_duration_seconds,
                    "candidate_count": len(decoded.frames),
                    "sample_interval_seconds": decoded.sample_interval_seconds,
                }
            )
            report["decode_calls"] += 1
        processor = VideoProcessor(decoded_sources[source_sha256])
        selected_case = {
            "id": case["id"],
            "source_sha256": source_sha256,
            "question": case["question"],
            "question_window": [case["start_seconds"], case["end_seconds"]],
            "conditions": [],
        }
        for name, scope, strategy, frame_cap in CONDITIONS:
            evidence = processor.analyze(
                VideoRequest(
                    scope=Scope(namespace="sampling-ablation", user_id="public-fixtures"),
                    video_base64=base64.b64encode(raw).decode("ascii"),
                    question=case["question"],
                    strategy=strategy,
                    max_frames=frame_cap,
                    start_seconds=case["start_seconds"] if scope == "window" else 0,
                    end_seconds=case["end_seconds"] if scope == "window" else None,
                )
            )
            selected_case["conditions"].append(
                {
                    "condition": name,
                    "strategy": strategy,
                    "max_frames": frame_cap,
                    "selected_frames": len(evidence.selected_frames),
                    "analysis_window": [
                        evidence.analysis_start_seconds,
                        evidence.analysis_end_seconds,
                    ],
                    "method_version": evidence.method_version,
                    "window_candidate_count": evidence.window_candidate_count,
                    "frame_indices": [f.frame_index for f in evidence.selected_frames],
                    "timestamps_seconds": [f.timestamp_seconds for f in evidence.selected_frames],
                    "selection_reasons": [f.selection_reason for f in evidence.selected_frames],
                    "max_gap_seconds": evidence.max_gap_seconds,
                    "observed_transitions": evidence.observed_transitions,
                    "represented_transitions": evidence.represented_transitions,
                    "requires_review": evidence.requires_review,
                }
            )
        selections.append(selected_case)

    for selected_case, case in zip(selections, videos, strict=True):
        for condition in selected_case["conditions"]:
            interval_hits = score_intervals(
                condition["timestamps_seconds"], case["evidence_intervals"]
            )
            condition.update(
                strict_interval_hits=interval_hits,
                strict_intervals_hit=sum(interval["hit"] for interval in interval_hits),
                strict_intervals_total=len(interval_hits),
            )
        report["cases"].append(selected_case)
    report["unique_video_sources"] = len(decoded_sources)
    report["reused_evidence_requests"] = sum(source.requests for source in decoded_sources.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path(".tmp/video-sampling-ablation.json"))
    args = parser.parse_args()
    report = run_ablation(args.cache_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.output),
                "decode_calls": report["decode_calls"],
                "reused_evidence_requests": report["reused_evidence_requests"],
                "paid_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
