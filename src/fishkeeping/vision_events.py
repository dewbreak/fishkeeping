"""Image receipts reuse durable events while preserving original request identity."""

from __future__ import annotations

import hashlib
import json

from fishkeeping.jobs import EventQueue
from fishkeeping.models import EventRequest
from fishkeeping.storage import ConflictError
from fishkeeping.vision import ImageProcessor, _json, decode_image
from fishkeeping.vision_models import (
    Box,
    ImageEventRequest,
    ImageEventResponse,
    ImageEvidence,
    TextRegion,
)


def enqueue_image(
    request: ImageEventRequest, queue: EventQueue, processor: ImageProcessor
) -> ImageEventResponse:
    raw, _ = decode_image(request.image_base64)
    identity = request.model_dump(mode="json", exclude={"image_base64"})
    # Preserve identities written by v0.3, which had no layout option.
    if request.layout == "text":
        identity.pop("layout")
    identity["image_sha256"] = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(_json(identity).encode()).hexdigest()

    def previous():
        saved = queue.get(request.scope, request.event_id)
        if saved is None:
            return None
        try:
            episode = saved["episode"]
            if not episode.content.startswith("[tool] "):
                raise ValueError("not an image event")
            envelope = json.loads(episode.content[7:])
            if envelope["type"] != "image_event_v1" or envelope["request_sha256"] != digest:
                raise ValueError("different image event")
            data = envelope["evidence"]
            regions = [
                TextRegion(
                    id=r[0],
                    box=Box(x1=r[1][0], y1=r[1][1], x2=r[1][2], y2=r[1][3]),
                    text=r[2],
                    confidence=r[3],
                )
                for r in data["regions"]
            ]
            packed = _json(data)
            evidence = ImageEvidence(
                image_sha256=data["sha256"],
                width=data["size"][0],
                height=data["size"][1],
                method=data["method"],
                method_version=data["version"],
                question=data["question"],
                regions=regions,
                total_regions=len(regions) + data["omitted"],
                omitted_regions=data["omitted"],
                requires_review=data["requires_review"],
                warnings=envelope["warnings"],
                packed=packed,
                chars_used=len(packed),
                selection=data.get("selection", {}),
            )
            saved["replayed"] = True
            return ImageEventResponse(evidence=evidence, event=saved)
        except (ValueError, TypeError, KeyError, IndexError):
            raise ConflictError(
                "event_id belongs to a different image request or event mode"
            ) from None

    saved = previous()
    if saved is not None:
        return saved
    evidence = processor.analyze(request)
    envelope = _json(
        {
            "type": "image_event_v1",
            "request_sha256": digest,
            "evidence": json.loads(evidence.packed),
            "warnings": evidence.warnings,
        }
    )
    try:
        receipt = queue.enqueue(
            EventRequest(
                scope=request.scope,
                event_id=request.event_id,
                session_id=request.session_id,
                messages=[{"role": "tool", "content": envelope}],
                extract=False,
            )
        )
    except ConflictError:
        saved = previous()
        if saved is not None:
            return saved
        raise
    return ImageEventResponse(evidence=evidence, event=receipt)
