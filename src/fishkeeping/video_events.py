"""Persist sampled video provenance as untrusted evidence, without storing media bytes."""

import hashlib
import json

from fishkeeping.jobs import EventQueue
from fishkeeping.models import EventRequest, EventResponse
from fishkeeping.storage import ConflictError
from fishkeeping.video import VideoProcessor, video_bytes
from fishkeeping.video_models import VideoEventRequest
from fishkeeping.vision import _json


def enqueue_video(
    request: VideoEventRequest, queue: EventQueue, processor: VideoProcessor
) -> EventResponse:
    raw = video_bytes(request.video_base64)
    identity = request.model_dump(mode="json", exclude={"video_base64"})
    identity["video_sha256"] = hashlib.sha256(raw).hexdigest()
    digest = hashlib.sha256(_json(identity).encode()).hexdigest()

    def previous():
        saved = queue.get(request.scope, request.event_id)
        if saved is None:
            return None
        try:
            content = saved["episode"].content
            if not content.startswith("[tool] "):
                raise ValueError("not a video event")
            envelope = json.loads(content[7:])
            if envelope["type"] != "video_event_v1" or envelope["request_sha256"] != digest:
                raise ValueError("different request")
            saved["replayed"] = True
            return EventResponse.model_validate(saved)
        except (ValueError, TypeError, KeyError):
            raise ConflictError("event_id belongs to a different video request or mode") from None

    saved = previous()
    if saved is not None:
        return saved
    evidence = processor.analyze(request)
    envelope = _json(
        {
            "type": "video_event_v1",
            "request_sha256": digest,
            "evidence": json.loads(evidence.packed),
        }
    )
    try:
        result = queue.enqueue(
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
    return EventResponse.model_validate(result)
