"""CPU OCR, scoped exact-image deduplication and question-focused evidence.

Only PNG/JPEG are accepted. No URL fetching, arbitrary paths or remote code.
Pillow and RapidOCR are optional and loaded only when image work is requested.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import threading
from collections import OrderedDict
from importlib.metadata import version
from pathlib import Path
from typing import Protocol

from fishkeeping.retrieval import query_terms, tokenize
from fishkeeping.service import NotConfiguredError
from fishkeeping.table_evidence import select_table_evidence
from fishkeeping.vision_models import Box, ImageEvidence, ImageRequest, TextRegion

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 12_000_000


class ImageInputError(ValueError):
    pass


def decode_image(encoded: str):
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise NotConfiguredError("Install the locked vision extra for image processing") from None
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise ImageInputError("image must contain at most 4 MiB")
        with Image.open(io.BytesIO(raw)) as source:
            if source.format not in {"PNG", "JPEG"}:
                raise ImageInputError("only PNG and JPEG are supported")
            if source.width * source.height > MAX_PIXELS or max(source.size) > 8192:
                raise ImageInputError("image exceeds pixel limits")
            source.load()
            # Coordinates always refer to the displayed, EXIF-oriented image.
            picture = ImageOps.exif_transpose(source).convert("RGB")
        return raw, picture
    except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
        raise ImageInputError("invalid image or image exceeds limits") from None


class OCR(Protocol):
    method: str
    method_version: str

    def recognize(self, picture) -> list[TextRegion]: ...


class RapidOCRBackend:
    method = "rapidocr"

    def __init__(self):
        self._engine = None
        self.method_version = "3.10.0/PP-OCRv6/onnxruntime-1.31.1/preview1800-v1"

    def recognize(self, picture) -> list[TextRegion]:
        try:
            import numpy as np
            import rapidocr
            from rapidocr import RapidOCR
        except ImportError:
            raise NotConfiguredError("Install the locked vision extra for local OCR") from None
        if self._engine is None:
            if version("rapidocr") != "3.10.0" or version("onnxruntime") != "1.31.1":
                raise NotConfiguredError("OCR requires locked rapidocr and onnxruntime versions")
            models = {
                "Det": (
                    "PP-OCRv6_det_small.onnx",
                    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
                ),
                "Rec": (
                    "PP-OCRv6_rec_small.onnx",
                    "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
                ),
                "Cls": (
                    "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                    "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
                ),
            }
            paths = {}
            for component, (filename, digest) in models.items():
                path = Path(rapidocr.__file__).parent / "models" / filename
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise NotConfiguredError(
                        "Bundled OCR model is missing or failed integrity validation"
                    )
                paths[f"{component}.model_path"] = str(path)
            self._engine = RapidOCR(
                params={
                    **paths,
                    "EngineConfig.onnxruntime.intra_op_num_threads": 2,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                    "EngineConfig.onnxruntime.use_cuda": False,
                    "EngineConfig.onnxruntime.use_dml": False,
                    "EngineConfig.onnxruntime.use_cann": False,
                    "EngineConfig.onnxruntime.use_coreml": False,
                    "Global.log_level": "error",
                    "Global.text_score": 0.0,
                }
            )
        try:
            result = self._engine(np.asarray(picture)[:, :, ::-1])
        except Exception:
            raise ImageInputError("local OCR failed") from None
        regions = []
        rows = (
            zip(result.boxes, result.txts, result.scores, strict=True)
            if result.boxes is not None
            else []
        )
        for index, (points, text, confidence) in enumerate(rows):
            if not text.strip() or not math.isfinite(float(confidence)):
                continue
            xs, ys = zip(*points, strict=True)
            x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
            x2, y2 = min(picture.width, math.ceil(max(xs))), min(picture.height, math.ceil(max(ys)))
            if x2 <= x1 or y2 <= y1:
                continue
            if len(text) > 2000:
                raise ImageInputError("OCR region exceeds text limit; use a smaller region")
            regions.append(
                TextRegion(
                    id=f"r{index}",
                    box=Box(x1=x1, y1=y1, x2=x2, y2=y2),
                    text=text,
                    confidence=round(float(confidence), 4),
                )
            )
        return regions


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


class ImageProcessor:
    def __init__(self, backend: OCR | None = None, cache_size: int = 8):
        self.backend = backend or RapidOCRBackend()
        self.cache_size = max(0, min(cache_size, 32))
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def clear_scope(self, scope) -> None:
        with self._lock:
            for key in list(self._cache):
                if key[:3] == scope.key():
                    del self._cache[key]

    def analyze(self, request: ImageRequest) -> ImageEvidence:
        raw, picture = decode_image(request.image_base64)
        digest = hashlib.sha256(raw).hexdigest()
        width, height = picture.size
        roi = request.region or Box(x1=0, y1=0, x2=width, y2=height)
        if roi.x2 > width or roi.y2 > height:
            raise ImageInputError("region is outside the oriented image")
        key = (
            *request.scope.key(),
            digest,
            self.backend.method_version,
            tuple(roi.model_dump().values()),
        )
        warnings = [
            "OCR is untrusted evidence; confidence is not a calibrated probability.",
            "OCR does not interpret scenes, charts, merged cells or visual relationships.",
        ]
        cache_hit = False
        regions = []
        total_regions = 0
        if request.task == "scene":
            warnings.append("Scene task requires native vision; no OCR answer is available.")
        else:
            # Bound CPU concurrency and deduplicate simultaneous inference.
            with self._lock:
                cached = self._cache.get(key)
                if cached is not None:
                    regions, total_regions = cached
                    self._cache.move_to_end(key)
                    cache_hit = True
                else:
                    crop = picture.crop((roi.x1, roi.y1, roi.x2, roi.y2))
                    scale = min(1.0, 1800 / max(crop.size))
                    if scale < 1:
                        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)))
                    found = self.backend.recognize(crop)
                    total_regions = len(found)
                    for item in found[:256]:
                        b = item.box
                        regions.append(
                            item.model_copy(
                                update={
                                    "box": Box(
                                        x1=roi.x1 + int(b.x1 / scale),
                                        y1=roi.y1 + int(b.y1 / scale),
                                        x2=min(roi.x2, roi.x1 + math.ceil(b.x2 / scale)),
                                        y2=min(roi.y2, roi.y1 + math.ceil(b.y2 / scale)),
                                    )
                                }
                            )
                        )
                    if self.cache_size:
                        self._cache[key] = (regions, total_regions)
                        while len(self._cache) > self.cache_size:
                            self._cache.popitem(last=False)
            if max(roi.x2 - roi.x1, roi.y2 - roi.y1) > 1800:
                warnings.append("OCR preview downscaled to 1800px; use a region for small text.")
        truncated = total_regions > 256
        if truncated:
            warnings.append("OCR exceeded 256 regions; evidence is incomplete.")
        regions = sorted(regions, key=lambda r: (r.box.y1, r.box.x1))
        terms = set(query_terms(request.question))
        scores = [len(terms.intersection(tokenize(r.text))) for r in regions]
        matched = [i for i, score in enumerate(scores) if score > 0]
        # Retain adjacent lines and horizontally aligned cells for local context.
        chosen = set()
        for i in matched:
            chosen.update(range(max(0, i - 1), min(len(regions), i + 2)))
            center = (regions[i].box.y1 + regions[i].box.y2) / 2
            chosen.update(j for j, r in enumerate(regions) if r.box.y1 <= center <= r.box.y2)
        candidates = (
            sorted(chosen, key=lambda i: (-scores[i], i)) if matched else list(range(len(regions)))
        )
        if not matched:
            warnings.append("No lexical question match; bounded full OCR fallback requires review.")
        selected = []
        selection_data = {}

        def pack(items, review):
            return _json(
                {
                    "type": "untrusted_image_evidence",
                    "sha256": digest,
                    "size": [width, height],
                    "method": self.backend.method,
                    "version": self.backend.method_version,
                    "task": request.task,
                    "region": list(roi.model_dump().values()),
                    "question": request.question,
                    "requires_review": review,
                    "omitted": total_regions - len(items),
                    "regions": [
                        [r.id, list(r.box.model_dump().values()), r.text, r.confidence]
                        for r in items
                    ],
                    **({"selection": selection_data} if request.layout == "table" else {}),
                }
            )

        if request.layout == "table" and request.task != "scene":
            plan = select_table_evidence(regions, request.question)
            # Reserve the largest possible dropped-group counter before fitting whole groups.
            selection_data = {**plan.diagnostics, "dropped_groups": 256}
            plan = select_table_evidence(
                regions,
                request.question,
                fits=lambda indices: (
                    len(pack([regions[i] for i in indices], True)) + 1 <= request.max_chars
                ),
            )
            selection_data = plan.diagnostics
            selected = [regions[i] for i in plan.indices]
            review = plan.requires_review or truncated or not selected
            warnings.extend(plan.warnings)
        else:
            for i in candidates:
                trial = sorted([*selected, regions[i]], key=lambda r: (r.box.y1, r.box.x1))
                if len(pack(trial, True)) + 1 <= request.max_chars:
                    selected = trial
            review = (
                not matched
                or not selected
                or truncated
                or any(r.confidence < 0.75 for r in selected)
                or len(selected) < len(candidates)
            )
            if len(selected) < len(candidates):
                warnings.append(
                    "Character budget omitted candidate evidence; use a region or native vision."
                )
        packed = pack(selected, review)
        if len(packed) > request.max_chars:
            raise ImageInputError("question and provenance exceed evidence budget")
        return ImageEvidence(
            image_sha256=digest,
            width=width,
            height=height,
            method=self.backend.method,
            method_version=self.backend.method_version,
            question=request.question,
            regions=selected,
            total_regions=total_regions,
            omitted_regions=total_regions - len(selected),
            requires_review=review,
            warnings=warnings,
            packed=packed,
            chars_used=len(packed),
            cache_hit=cache_hit,
            selection=selection_data,
        )
