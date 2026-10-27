"""Frozen synthetic image diagnostics. --ocr opts in to bundled CPU models; never pays an API."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from fishkeeping.models import Scope
from fishkeeping.vision import ImageProcessor, RapidOCRBackend
from fishkeeping.vision_models import ImageRequest

FIXTURE = Path(__file__).resolve().parents[1] / "tests/fixtures/image_cases.json"


def load_dataset():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def fingerprint(data):
    return hashlib.sha256(
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def font(size, chinese=False):
    paths = (
        ["C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
        if chinese
        else ["C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    )
    for path in paths:
        if Path(path).is_file():
            return ImageFont.truetype(path, size), path
    if chinese:
        raise RuntimeError("Chinese font unavailable; case must be reported as skipped")
    return ImageFont.load_default(size=size), "Pillow default"


def generate_case(case, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    kind = case["kind"]
    height = 1600 if case["id"] == "long_screenshot" else 650
    image = Image.new("RGB", (1400, height), "white")
    draw = ImageDraw.Draw(image)
    face, font_path = font(32, kind == "chinese")
    truth = []
    if kind == "table":
        for row_index, row in enumerate(case["rows"]):
            for column, cell in enumerate(row):
                x, y = 40 + column * 320, 50 + row_index * 95
                draw.rectangle((x - 10, y - 10, x + 300, y + 75), outline="#888888", width=2)
                draw.text((x, y + 12), cell, font=face, fill="black")
                truth.append(cell)
    elif kind == "scene":
        for x in (100, 480, 860):
            draw.ellipse((x, 70, x + 180, 250), fill="red")
        for x in (260, 750):
            draw.rectangle((x, 370, x + 180, 550), fill="blue")
    else:
        for i, line in enumerate(case["lines"]):
            draw.text((40, 40 + i * 62), line, font=face, fill="black")
            truth.append(line)
    if kind == "rotated":
        image = image.rotate(180)
        truth.reverse()
    if kind == "degraded":
        image = image.resize((350, 162)).filter(ImageFilter.GaussianBlur(0.8)).resize(image.size)
    path = output_dir / f"{case['id']}.png"
    image.save(path)
    return path, {
        "font": font_path,
        "truth": truth,
        "size": list(image.size),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def normalize(text):
    return re.sub(r"\s+", "", text).casefold()


def edit_distance(left, right):
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        new = [i]
        for j, b in enumerate(right, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def peak_rss_bytes():
    if os.name != "nt":
        import resource

        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return value if platform.system() == "Darwin" else value * 1024
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            *[
                (name, ctypes.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ],
        ]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    current_process = ctypes.windll.kernel32.GetCurrentProcess
    current_process.restype = wintypes.HANDLE
    memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
    memory_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    memory_info.restype = wintypes.BOOL
    if not memory_info(current_process(), ctypes.byref(counters), counters.cb):
        raise OSError("Could not measure process peak working set")
    return counters.PeakWorkingSetSize


def run(output_dir: Path, use_ocr: bool):
    data = load_dataset()
    report = {
        "dataset_sha256": fingerprint(data),
        "synthetic_only": True,
        "paid_calls": 0,
        "actual_tokens": None,
        "estimated_api_cost_usd": 0,
        "metric_note": (
            "Evidence phrase retention is a proxy, not LLM final-answer accuracy. "
            "Characters are not tokens."
        ),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cases": [],
    }
    manifest = {"dataset_sha256": report["dataset_sha256"], "cases": []}
    backend = RapidOCRBackend() if use_ocr else None
    processor = ImageProcessor(backend) if use_ocr else None
    for case in data["cases"]:
        result = {"id": case["id"], "task": case["task"]}
        try:
            path, meta = generate_case(case, output_dir)
        except RuntimeError as exc:
            result.update(status="skipped", reason=str(exc))
            report["cases"].append(result)
            continue
        result.update(meta, status="generated")
        live = {
            "id": case["id"],
            "task": case["task"],
            "image_path": str(path.resolve()),
            "question": case["question"],
            "expected_terms": case["expected_terms"],
        }
        if use_ocr:
            with Image.open(path) as image:
                start, cpu = time.perf_counter(), time.process_time()
                baseline = (
                    backend.recognize(image.convert("RGB")) if case["task"] != "scene" else []
                )
                result["baseline_wall_seconds"] = time.perf_counter() - start
                result["baseline_cpu_seconds"] = time.process_time() - cpu
            original = "\n".join(r.text for r in baseline)
            req = ImageRequest(
                scope=Scope(namespace="image-evaluation", user_id="synthetic"),
                image_base64=base64.b64encode(path.read_bytes()).decode(),
                question=case["question"],
                task=case["task"],
            )
            start, cpu = time.perf_counter(), time.process_time()
            evidence = processor.analyze(req)
            result["optimized_wall_seconds"] = time.perf_counter() - start
            result["optimized_cpu_seconds"] = time.process_time() - cpu
            selected = "\n".join(r.text for r in evidence.regions)
            truth = normalize("\n".join(meta["truth"]))
            required = case["required_evidence"]
            result.update(
                status="measured",
                baseline_text_chars=len(original),
                optimized_text_chars=len(selected),
                packed_chars=evidence.chars_used,
                baseline_character_error_rate=edit_distance(truth, normalize(original)) / len(truth)
                if truth
                else None,
                required_phrases=len(required),
                baseline_retained=sum(normalize(x) in normalize(original) for x in required),
                optimized_retained=sum(normalize(x) in normalize(selected) for x in required),
                requires_review=evidence.requires_review,
                omitted_regions=evidence.omitted_regions,
                total_regions=evidence.total_regions,
                warnings=evidence.warnings,
                peak_process_rss_bytes=peak_rss_bytes(),
            )
            start = time.perf_counter()
            repeated = processor.analyze(req)
            result["repeat_wall_seconds"] = time.perf_counter() - start
            result["repeat_cache_hit"] = repeated.cache_hit
            evidence_path = output_dir / f"{case['id']}-evidence.json"
            evidence_path.write_text(evidence.model_dump_json(indent=2), encoding="utf-8")
            live["evidence_path"] = str(evidence_path.resolve())
        if case.get("live"):
            manifest["cases"].append(live)
        report["cases"].append(result)
    (output_dir / "live-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/image-evaluation"))
    parser.add_argument("--output", type=Path, default=Path(".tmp/images-offline.json"))
    args = parser.parse_args()
    report = run(args.output_dir, args.ocr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "dataset_sha256": report["dataset_sha256"],
                "cases": len(report["cases"]),
                "paid_calls": 0,
                "report": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
