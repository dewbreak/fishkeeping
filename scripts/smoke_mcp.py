"""Real stdio and loopback Streamable HTTP MCP smoke, without model calls."""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import uvicorn
from mcp import Client, StdioServerParameters
from PIL import Image

from fishkeeping.config import Settings
from fishkeeping.mcp_server import create_mcp_server
from fishkeeping.models import Scope


async def workflow(client: Client):
    tools = await client.list_tools()
    names = {tool.name for tool in tools.tools}
    assert "memory_remember" in names and "image_view" in names and "video_sample" not in names
    saved = await client.call_tool(
        "memory_remember", {"content": "Filter maintenance is weekly", "kind": "episodic"}
    )
    identifier = saved.structured_content["id"]
    recalled = await client.call_tool("memory_context", {"query": "Filter maintenance"})
    assert identifier in recalled.structured_content["included_ids"]
    preview = await client.call_tool("image_view", {"path": "panel.png"})
    assert sum(block.type == "image" for block in preview.content) == 1
    assert preview.structured_content["preview_size"] == [512, 256]
    forgotten = await client.call_tool("memory_forget", {"memory_id": identifier})
    assert forgotten.structured_content["deleted_ids"] == [identifier]
    return {"listed_tools": len(names), "memory_cycle": "passed", "native_image": "passed"}


async def main():
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="fishkeeping-mcp-smoke-") as folder:
        root = Path(folder)
        Image.new("RGB", (800, 400), "teal").save(root / "panel.png")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "fishkeeping.cli",
                "mcp",
                "--user-id",
                "stdio-smoke",
                "--modules",
                "memory",
                "image",
                "--media-root",
                str(root),
            ],
            cwd=root,
            env={"FISHKEEPING_DB_PATH": str(root / "stdio.sqlite3")},
        )
        async with Client(parameters, raise_exceptions=True) as client:
            stdio = await workflow(client)

        mcp = create_mcp_server(
            Settings(
                _env_file=None,
                db_path=str(root / "http.sqlite3"),
                embedding_backend="remote",
                embedding_base_url=None,
                embedding_model=None,
            ),
            Scope(user_id="http-smoke"),
            modules=("memory", "image"),
            media_root=root,
        )
        app = mcp.streamable_http_app(stateless_http=True, json_response=True)
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [listener]}, daemon=True
            )
            thread.start()
            try:
                deadline = time.monotonic() + 15
                while not server.started:
                    if not thread.is_alive() or time.monotonic() > deadline:
                        raise RuntimeError("MCP HTTP server did not start")
                    await asyncio.sleep(0.05)
                async with Client(f"http://127.0.0.1:{port}/mcp", raise_exceptions=True) as client:
                    http = await workflow(client)
            finally:
                server.should_exit = True
                await asyncio.to_thread(thread.join, 15)
    print(
        json.dumps(
            {
                "stdio": stdio,
                "streamable_http": http,
                "model_calls": 0,
                "wall_seconds": round(time.perf_counter() - started, 3),
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
