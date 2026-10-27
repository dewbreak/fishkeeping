"""Offline media API, durable provenance and paid-adapter boundary regressions."""

from __future__ import annotations

import base64
import hashlib
import io
import json

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fishkeeping.api import create_app
from fishkeeping.client import MemoryClient
from fishkeeping.config import Settings
from fishkeeping.conversations import trusted_conversation_content
from fishkeeping.image_chat import DeepSeekImageChat, compact_evidence
from fishkeeping.models import EventRequest, Scope
from fishkeeping.providers import ProviderError
from fishkeeping.video import DecodedFrame, DecodedVideo, VideoProcessor, _fingerprint
from fishkeeping.video_models import VideoRequest
from fishkeeping.vision import ImageInputError, ImageProcessor
from fishkeeping.vision_models import Box, ImageRequest, TextRegion

KEY = "offline-media-integration-key"
SCOPE = Scope(user_id="media-integration", agent_id="helper")
VIDEO = b"\0\0\0\x10ftypisomoffline-video"
QUESTION = "What appears in temporal order?"


def encoded(raw):
    return base64.b64encode(raw).decode("ascii")


def image_bytes(color="white", fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (160, 90), color).save(output, format=fmt)
    return output.getvalue()


class Decoder:
    def __init__(self):
        self.calls = 0
        frames = []
        for i, color in enumerate(["red", "red", "blue", "blue", "red", "red"]):
            image = Image.new("RGB", (160, 90), color)
            frames.append(
                DecodedFrame(i, float(i), image_bytes(color, "JPEG"), 160, 90, _fingerprint(image))
            )
        self.video = DecodedVideo(frames, 6, 5.0, 6.0, 1.0, "offline-fixture-decoder-v1")

    def decode(self, raw):
        assert raw == VIDEO
        self.calls += 1
        return self.video


class OCR:
    method = "offline-fixture-ocr"
    method_version = "v1"

    def __init__(self):
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return [
            TextRegion(
                id="value",
                box=Box(x1=10, y1=15, x2=150, y2=35),
                text="Filter price: -25 USD",
                confidence=0.99,
            )
        ]


@pytest.fixture
def media_api(tmp_path):
    settings = Settings(
        _env_file=None,
        db_path=str(tmp_path / "media.db"),
        api_keys={KEY: "default"},
    )
    app = create_app(settings)
    decoder, ocr = Decoder(), OCR()
    app.state.videos = VideoProcessor(decoder)
    app.state.images = ImageProcessor(ocr)
    with TestClient(app) as api:
        yield app, api, decoder, ocr


def video_request(**kwargs):
    return VideoRequest(scope=SCOPE, video_base64=encoded(VIDEO), question=QUESTION, **kwargs)


def video_evidence():
    return VideoProcessor(Decoder()).analyze(video_request(max_frames=6, strategy="uniform"))


def image_evidence():
    return ImageProcessor(OCR()).analyze(
        ImageRequest(scope=SCOPE, image_base64=encoded(image_bytes()), question="Filter price?")
    )


def model_response(ids):
    return httpx.Response(
        200,
        json={
            "model": "deepseek-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {"answer": "Observed evidence", "uncertain": True, "evidence_ids": ids}
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        },
    )


def chat_settings():
    return Settings(_env_file=None, deepseek_api_key="offline-test-key")


@pytest.mark.parametrize("path", ["/v1/videos/analyze", "/v1/videos/events"])
def test_video_routes_check_auth_scope_and_invalid_inputs_before_decode(media_api, path):
    _, api, decoder, _ = media_api
    body = video_request().model_dump(mode="json")
    if path.endswith("events"):
        body.update(event_id="video-auth", session_id="media")
    assert api.post(path, json=body).status_code == 401
    api.headers["X-API-Key"] = KEY
    foreign = {**body, "scope": {**body["scope"], "namespace": "foreign"}}
    assert api.post(path, json=foreign).status_code == 403
    assert api.post(path, json={**body, "video_base64": "not base64!"}).status_code == 422
    assert decoder.calls == 0


def test_video_busy_returns_retryable_status_without_decoding(media_api):
    app, api, decoder, _ = media_api
    api.headers["X-API-Key"] = KEY
    app.state.videos._decode_lock.acquire()
    try:
        response = api.post("/v1/videos/analyze", json=video_request().model_dump(mode="json"))
    finally:
        app.state.videos._decode_lock.release()
    assert response.status_code == 429 and decoder.calls == 0


def test_video_sdk_event_lifecycle_preserves_provenance_and_never_redecodes_replay(media_api):
    _, api, decoder, _ = media_api
    options = {"max_frames": 4, "strategy": "uniform", "start_seconds": 1, "end_seconds": 4}
    with MemoryClient("http://testserver", SCOPE, api_key=KEY, transport=api._transport) as sdk:
        evidence = sdk.analyze_video(VIDEO, QUESTION, **options)
        assert decoder.calls == 1
        assert evidence.analysis_start_seconds == 1 and evidence.analysis_end_seconds == 4
        assert [f.timestamp_seconds for f in evidence.selected_frames] == [1, 2, 3, 4]
        assert evidence.video_sha256 == hashlib.sha256(VIDEO).hexdigest()
        result = sdk.enqueue_video(VIDEO, QUESTION, "video-one", "media", **options)
        assert decoder.calls == 2
        assert result.job is None and not result.replayed
        assert trusted_conversation_content(result.episode) is None
        assert result.episode.content.startswith("[tool] ")
        envelope = json.loads(result.episode.content[7:])
        assert envelope["type"] == "video_event_v1"
        assert envelope["evidence"] == json.loads(evidence.packed)
        assert "image_base64" not in result.episode.content
        assert "video_base64" not in result.episode.content
        replay = sdk.enqueue_video(VIDEO, QUESTION, "video-one", "media", **options)
        assert replay.replayed and replay.episode.id == result.episode.id
        assert sdk.event("video-one").episode.id == result.episode.id
        assert decoder.calls == 2
        for change in ({"end_seconds": 5}, {"strategy": "adaptive"}, {"max_frames": 3}):
            with pytest.raises(httpx.HTTPStatusError) as conflict:
                sdk.enqueue_video(VIDEO, QUESTION, "video-one", "media", **{**options, **change})
            assert conflict.value.response.status_code == 409
        for raw, question in ((VIDEO + b"changed", QUESTION), (VIDEO, "A different question?")):
            with pytest.raises(httpx.HTTPStatusError) as conflict:
                sdk.enqueue_video(raw, question, "video-one", "media", **options)
            assert conflict.value.response.status_code == 409
        assert decoder.calls == 2
        fact = sdk.remember(
            "Reviewed: red then blue then red",
            kind="semantic",
            source_ids=[result.episode.id],
        )
        assert sdk.sources(fact.id).items == [result.episode.id]
        assert set(sdk.forget(result.episode.id)) == {result.episode.id, fact.id}
        with pytest.raises(httpx.HTTPStatusError) as forgotten:
            sdk.enqueue_video(VIDEO, QUESTION, "video-one", "media", **options)
        assert forgotten.value.response.status_code == 409
        assert decoder.calls == 2


@pytest.mark.parametrize("other_mode", ["image", "conversation"])
def test_video_event_id_cannot_be_reused_by_other_evidence_modes(media_api, other_mode):
    _, api, decoder, ocr = media_api
    with MemoryClient("http://testserver", SCOPE, api_key=KEY, transport=api._transport) as sdk:
        sdk.enqueue_video(VIDEO, QUESTION, "shared-event", "media")
        with pytest.raises(httpx.HTTPStatusError) as conflict:
            if other_mode == "image":
                sdk.enqueue_image(image_bytes(), "Filter price?", "shared-event", "media")
            else:
                sdk.enqueue(
                    "shared-event", "media", [{"role": "user", "content": QUESTION}], extract=False
                )
        assert conflict.value.response.status_code == 409
    assert decoder.calls == 1 and ocr.calls == 0


@pytest.mark.parametrize("field", ["user_id", "agent_id"])
def test_video_event_receipt_and_source_are_scope_isolated(media_api, field):
    _, api, decoder, _ = media_api
    with MemoryClient("http://testserver", SCOPE, api_key=KEY, transport=api._transport) as sdk:
        saved = sdk.enqueue_video(VIDEO, QUESTION, "scoped-video", "media")
    foreign = SCOPE.model_copy(update={field: "another"})
    with MemoryClient("http://testserver", foreign, api_key=KEY, transport=api._transport) as sdk:
        with pytest.raises(httpx.HTTPStatusError) as unavailable:
            sdk.event("scoped-video")
        assert unavailable.value.response.status_code == 404
        with pytest.raises(httpx.HTTPStatusError) as unavailable:
            sdk.get(saved.episode.id)
        assert unavailable.value.response.status_code == 404
    assert decoder.calls == 1


def test_video_event_replays_after_app_restart_without_decoder(tmp_path):
    settings = Settings(_env_file=None, db_path=str(tmp_path / "restart.db"))
    first = create_app(settings)
    first.state.videos = VideoProcessor(Decoder())
    body = {**video_request().model_dump(mode="json"), "event_id": "persist", "session_id": "s"}
    with TestClient(first) as api:
        saved = api.post("/v1/videos/events", json=body)
        assert saved.status_code == 202
    second = create_app(settings)

    class UnavailableDecoder:
        def decode(self, raw):
            pytest.fail("durable replay must not need installed codecs or decode again")

    second.state.videos = VideoProcessor(UnavailableDecoder())
    with TestClient(second) as api:
        replay = api.post("/v1/videos/events", json=body)
        assert replay.status_code == 202
        assert replay.json()["replayed"]
        assert replay.json()["episode"]["id"] == saved.json()["episode"]["id"]


def test_v03_image_event_identity_replays_with_default_layout_and_rejects_table(media_api):
    app, api, _, ocr = media_api
    raw = image_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    # Exact pre-layout v0.3 field order, deliberately independent of the current model dump.
    legacy_identity = {
        "scope": SCOPE.model_dump(mode="json"),
        "question": "Filter price?",
        "max_chars": 3000,
        "task": "document",
        "region": None,
        "event_id": "legacy-image",
        "session_id": "media",
        "image_sha256": digest,
    }
    old_data = {
        "type": "untrusted_image_evidence",
        "sha256": digest,
        "size": [160, 90],
        "method": "old-ocr",
        "version": "v0.3-locked",
        "task": "document",
        "region": [0, 0, 160, 90],
        "question": "Filter price?",
        "requires_review": False,
        "omitted": 0,
        "regions": [["legacy-r0", [10, 15, 150, 35], "Filter price: -25 USD", 0.99]],
    }
    old_json = json.dumps(legacy_identity, ensure_ascii=False, separators=(",", ":"))
    envelope = {
        "type": "image_event_v1",
        "request_sha256": hashlib.sha256(old_json.encode()).hexdigest(),
        "evidence": old_data,
        "warnings": ["Original v0.3 warning"],
    }
    saved = app.state.queue.enqueue(
        EventRequest(
            scope=SCOPE,
            event_id="legacy-image",
            session_id="media",
            messages=[{"role": "tool", "content": json.dumps(envelope, separators=(",", ":"))}],
            extract=False,
        )
    )
    with MemoryClient("http://testserver", SCOPE, api_key=KEY, transport=api._transport) as sdk:
        for kwargs in ({}, {"layout": "text"}):
            result = sdk.enqueue_image(raw, "Filter price?", "legacy-image", "media", **kwargs)
            assert result.event.replayed and result.event.episode.id == saved["episode"].id
            assert result.evidence.method_version == "v0.3-locked"
            assert result.evidence.warnings == ["Original v0.3 warning"]
            assert result.evidence.selection == {}
            assert json.loads(result.evidence.packed) == old_data
            assert "selection" not in json.loads(result.evidence.packed)
        with pytest.raises(httpx.HTTPStatusError) as conflict:
            sdk.enqueue_image(raw, "Filter price?", "legacy-image", "media", layout="table")
        assert conflict.value.response.status_code == 409
    assert ocr.calls == 0


def test_compact_ocr_keeps_literal_text_ids_coordinates_and_full_audit():
    evidence = image_evidence()
    original = evidence.model_dump()
    packed = compact_evidence(evidence)
    data = json.loads(packed)
    assert data["untrusted_ocr"] == [["value", "Filter price: -25 USD", [10, 15, 150, 35]]]
    assert data["requires_review"] is False and data["omitted_regions"] == 0
    assert len(packed) < evidence.chars_used
    assert evidence.model_dump() == original
    assert json.loads(evidence.packed)["sha256"] == evidence.image_sha256
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return model_response(["value"])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        answer = DeepSeekImageChat(chat_settings(), client).answer_evidence(evidence)
    assert answer.evidence_ids == ["value"]
    assert requests[0]["messages"][1]["content"][1]["text"] == packed


@pytest.mark.parametrize("review", [False, True])
def test_empty_ocr_never_spends_model_call_even_when_review_is_allowed(review):
    evidence = image_evidence().model_copy(update={"regions": [], "requires_review": review})

    def unexpected(request):
        pytest.fail("empty OCR must fail before any paid transport call")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ImageInputError):
            DeepSeekImageChat(chat_settings(), client).answer_evidence(evidence, allow_review=True)


def test_video_adapter_sends_ordered_frame_labels_pixels_and_review_metadata():
    evidence = video_evidence()
    packed = json.dumps({**json.loads(evidence.packed), "requires_review": True})
    evidence = evidence.model_copy(
        update={"requires_review": True, "packed": packed, "chars_used": len(packed)}
    )
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        return model_response([evidence.selected_frames[2].id])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        chat = DeepSeekImageChat(chat_settings(), client)
        with pytest.raises(ImageInputError, match="review"):
            chat.answer_video(evidence)
        assert calls == []
        answer = chat.answer_video(evidence, allow_review=True)
    assert answer.evidence_ids == [evidence.selected_frames[2].id]
    assert len(calls) == 1
    payload = calls[0]
    assert payload["thinking"] == {"type": "disabled"} and payload["max_tokens"] <= 1024
    blocks = payload["messages"][1]["content"]
    metadata = json.loads(blocks[1]["text"])
    assert metadata["requires_review"] is True
    assert metadata["audio_available"] is False
    assert metadata["analysis_window_seconds"] == [0, 5]
    for i, frame in enumerate(evidence.selected_frames):
        assert blocks[2 + 2 * i] == {
            "type": "text",
            "text": f"{frame.id} at {frame.timestamp_seconds:.3f}s",
        }
        image = blocks[3 + 2 * i]["image_url"]
        assert image["detail"] == "low"
        assert image["url"].startswith("data:image/png;base64,")
    assert "no audio" in payload["messages"][0]["content"]


@pytest.mark.parametrize(
    "change", ["hash", "dimensions", "reverse", "duplicate_id", "timestamp", "index", "window"]
)
def test_video_adapter_rejects_invalid_frame_provenance_before_transport(change):
    evidence = video_evidence().model_copy(deep=True)
    frames = evidence.selected_frames
    if change == "hash":
        frames[1].image_sha256 = "0" * 64
    elif change == "dimensions":
        frames[1].width += 1
    elif change == "reverse":
        frames.reverse()
    elif change == "duplicate_id":
        frames[1].id = frames[0].id
    elif change == "timestamp":
        frames[1].timestamp_seconds = frames[0].timestamp_seconds
    elif change == "index":
        frames[1].frame_index = frames[0].frame_index
    else:
        evidence.analysis_end_seconds = 4

    def unexpected(request):
        pytest.fail("invalid video provenance must fail before a paid transport call")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        with pytest.raises(ImageInputError):
            DeepSeekImageChat(chat_settings(), client).answer_video(evidence)


def test_video_adapter_rejects_unknown_citation_without_retry():
    calls = []

    def respond(request):
        calls.append(request)
        return model_response(["frame-not-supplied"])

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ProviderError) as error:
            DeepSeekImageChat(chat_settings(), client).answer_video(video_evidence())
    assert error.value.code == "invalid_output" and len(calls) == 1


@pytest.mark.parametrize("kind", ["image", "video"])
@pytest.mark.parametrize(
    "change",
    ["content", "question", "chars_used", "packed_json", "duplicate_json", "nonfinite_json"],
)
def test_evidence_mutation_cannot_silently_change_model_input_beyond_saved_provenance(kind, change):
    evidence = image_evidence() if kind == "image" else video_evidence()
    evidence = evidence.model_copy(deep=True)
    if change == "content":
        if kind == "image":
            evidence.regions[0].text = "Filter price: 25000 USD"
        else:
            # The substituted pixels and new hash agree with each other but not the audit record.
            replacement = image_bytes("green", "JPEG")
            evidence.selected_frames[1].image_base64 = encoded(replacement)
            evidence.selected_frames[1].image_sha256 = hashlib.sha256(replacement).hexdigest()
    elif change == "question":
        evidence.question = "A different question than the saved request?"
    elif change == "chars_used":
        evidence.chars_used += 1
    elif change == "duplicate_json":
        evidence.packed = (
            evidence.packed[:-1] + ',"question":' + json.dumps(evidence.question) + "}"
        )
        evidence.chars_used = len(evidence.packed)
    elif change == "nonfinite_json":
        data = json.loads(evidence.packed)
        data["task" if kind == "image" else "max_gap_seconds"] = float("nan")
        evidence.packed = json.dumps(data)
        evidence.chars_used = len(evidence.packed)
    else:
        evidence.packed = "{}"
        evidence.chars_used = 2

    def unexpected(request):
        pytest.fail("changed evidence must fail before any paid transport call")

    with httpx.Client(transport=httpx.MockTransport(unexpected)) as client:
        chat = DeepSeekImageChat(chat_settings(), client)
        with pytest.raises(ImageInputError, match="packed provenance"):
            if kind == "image":
                chat.answer_evidence(evidence)
            else:
                chat.answer_video(evidence)
