"""Small protocol-level checks for selectable MCP modules and useful tool calls."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json

import pytest
from PIL import Image

mcp = pytest.importorskip("mcp")

from fishkeeping.config import Settings  # noqa: E402
from fishkeeping.models import Scope  # noqa: E402
from fishkeeping.video import (  # noqa: E402
    DecodedFrame,
    DecodedVideo,
    VideoProcessor,
    _fingerprint,
)
from fishkeeping.vision import ImageProcessor  # noqa: E402
from fishkeeping.vision_models import Box, TextRegion  # noqa: E402

SCOPE = Scope(namespace="mcp-tests", user_id="alice", agent_id="helper")
MEMORY_TOOLS = {
    "memory_remember",
    "memory_recall",
    "memory_context",
    "memory_list",
    "memory_get",
    "memory_forget",
    "memory_sources",
}
IMAGE_TOOLS = {"image_ocr", "image_view"}
VIDEO_TOOLS = {"video_sample"}
VIDEO = b"\0\0\0\x10ftypisomoffline-mcp-video"


def settings(tmp_path):
    return Settings(_env_file=None, db_path=str(tmp_path / "mcp.sqlite3"))


def make_server(tmp_path, modules=("memory",)):
    from fishkeeping.mcp_server import create_mcp_server

    return create_mcp_server(settings(tmp_path), SCOPE, modules=modules, media_root=tmp_path)


def pixels(color="white", size=(640, 320), fmt="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format=fmt)
    return output.getvalue()


@pytest.mark.parametrize(
    "modules,expected",
    [
        (("memory",), MEMORY_TOOLS),
        (("image",), IMAGE_TOOLS),
        (("video",), VIDEO_TOOLS),
        (("memory", "image", "video"), MEMORY_TOOLS | IMAGE_TOOLS | VIDEO_TOOLS),
    ],
)
def test_only_selected_modules_are_advertised(tmp_path, modules, expected):
    async def check():
        async with mcp.Client(make_server(tmp_path, modules), raise_exceptions=True) as client:
            result = await client.list_tools()
            assert {tool.name for tool in result.tools} == expected
            assert all(
                "scope" not in tool.input_schema.get("properties", {}) for tool in result.tools
            )

    asyncio.run(check())
    if "memory" not in modules:
        assert not (tmp_path / "mcp.sqlite3").exists()


def test_memory_tools_complete_a_scoped_source_lifecycle(tmp_path):
    async def check():
        async with mcp.Client(make_server(tmp_path), raise_exceptions=True) as client:
            source = (
                await client.call_tool(
                    "memory_remember", {"content": "Filter replacement is needed every four weeks."}
                )
            ).structured_content
            fact = (
                await client.call_tool(
                    "memory_remember",
                    {
                        "content": "Filter replacement interval: four weeks.",
                        "kind": "semantic",
                        "fact_key": "filter.interval",
                        "source_ids": [source["id"]],
                    },
                )
            ).structured_content
            assert fact["scope"] == SCOPE.model_dump()
            recalled = (
                await client.call_tool("memory_recall", {"query": "Filter replacement"})
            ).structured_content
            assert fact["id"] in {hit["memory"]["id"] for hit in recalled["hits"]}
            context = (
                await client.call_tool(
                    "memory_context", {"query": "Filter replacement", "max_chars": 2000}
                )
            ).structured_content
            assert fact["id"] in context["included_ids"]
            assert context["chars_used"] <= 2000
            fetched = (
                await client.call_tool("memory_get", {"memory_id": fact["id"]})
            ).structured_content
            assert fetched["content"] == fact["content"]
            sources = (
                await client.call_tool("memory_sources", {"memory_id": fact["id"]})
            ).structured_content
            assert sources["items"] == [source["id"]]
            deleted = (
                await client.call_tool("memory_forget", {"memory_id": source["id"]})
            ).structured_content
            assert set(deleted["deleted_ids"]) == {source["id"], fact["id"]}
            listed = (await client.call_tool("memory_list")).structured_content
            assert listed["result"] == []

    asyncio.run(check())


def test_image_view_returns_native_image_blocks_at_the_requested_detail(tmp_path):
    raw = pixels()
    (tmp_path / "photo.png").write_bytes(raw)

    async def check():
        async with mcp.Client(make_server(tmp_path, ("image",)), raise_exceptions=True) as client:
            for detail, expected_size in [("low", (512, 256)), ("original", (640, 320))]:
                result = await client.call_tool(
                    "image_view", {"path": "photo.png", "detail": detail}
                )
                blocks = [block for block in result.content if block.type == "image"]
                assert len(blocks) == 1
                with Image.open(io.BytesIO(base64.b64decode(blocks[0].data))) as picture:
                    assert picture.size == expected_size
                metadata = json.dumps(result.structured_content)
                assert hashlib.sha256(raw).hexdigest() in metadata
                assert "image_base64" not in metadata

    asyncio.run(check())


def test_image_ocr_reuses_local_evidence_and_table_options(tmp_path, monkeypatch):
    import fishkeeping.mcp_server as server_module

    class OCR:
        method = "offline-mcp-ocr"
        method_version = "v1"
        calls = 0

        def recognize(self, picture):
            self.calls += 1
            return [
                TextRegion(
                    id="price",
                    text="Filter price: 25 USD",
                    confidence=0.99,
                    box=Box(x1=10, y1=10, x2=300, y2=40),
                )
            ]

    backend = OCR()
    monkeypatch.setattr(server_module, "ImageProcessor", lambda: ImageProcessor(backend))
    (tmp_path / "table.png").write_bytes(pixels())

    async def check():
        async with mcp.Client(make_server(tmp_path, ("image",)), raise_exceptions=True) as client:
            result = await client.call_tool(
                "image_ocr",
                {"path": "table.png", "question": "Filter price?", "layout": "table"},
            )
            evidence = result.structured_content
            assert evidence["question"] == "Filter price?"
            assert evidence["regions"][0][2] == "Filter price: 25 USD"
            assert evidence["selection"]["selection_mode"] == "full_table_fallback"
            assert "packed" not in evidence
            assert len(result.content) == 1 and result.content[0].type == "text"
            assert json.loads(result.content[0].text) == evidence

    asyncio.run(check())
    assert backend.calls == 1


def test_video_sample_returns_timestamped_frames_without_base64_metadata(tmp_path, monkeypatch):
    import fishkeeping.mcp_server as server_module

    class Decoder:
        calls = 0

        def decode(self, raw):
            assert raw == VIDEO
            self.calls += 1
            frames = [
                DecodedFrame(
                    i,
                    float(i),
                    pixels(color, size=(160, 90), fmt="JPEG"),
                    160,
                    90,
                    _fingerprint(Image.new("RGB", (160, 90), color)),
                )
                for i, color in enumerate(["red", "red", "blue", "blue", "red", "red"])
            ]
            return DecodedVideo(frames, 6, 5.0, 6.0, 1.0, "offline-mcp-decoder-v1")

    decoder = Decoder()
    monkeypatch.setattr(server_module, "VideoProcessor", lambda: VideoProcessor(decoder))
    (tmp_path / "clip.mp4").write_bytes(VIDEO)

    async def check():
        async with mcp.Client(make_server(tmp_path, ("video",)), raise_exceptions=True) as client:
            result = await client.call_tool(
                "video_sample",
                {
                    "path": "clip.mp4",
                    "question": "What changes between one and four seconds?",
                    "start_seconds": 1,
                    "end_seconds": 4,
                    "max_frames": 3,
                    "strategy": "uniform",
                },
            )
            metadata = result.structured_content
            assert metadata["analysis_start_seconds"] == 1
            assert metadata["analysis_end_seconds"] == 4
            assert len([block for block in result.content if block.type == "image"]) == 3
            assert "image_base64" not in json.dumps(metadata)
            frames = metadata["frames"]
            times = [frame["timestamp_seconds"] for frame in frames]
            assert times == sorted(times) and all(1 <= value <= 4 for value in times)
            text = "\n".join(block.text for block in result.content if block.type == "text")
            assert all(frame["id"] in text for frame in frames)

    asyncio.run(check())
    assert decoder.calls == 1
