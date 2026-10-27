"""Run after starting fishkeeping serve: python examples/basic.py."""

import os
from uuid import uuid4

from fishkeeping.client import MemoryClient
from fishkeeping.models import Scope

with MemoryClient(
    os.getenv("MEMORY_URL", "http://127.0.0.1:8765"),
    Scope(namespace=os.getenv("MEMORY_NAMESPACE", "default"), user_id="example-user"),
    api_key=os.getenv("MEMORY_API_KEY"),
) as memory:
    memory.remember(
        "我喜欢用 Python 开发，也喜欢用中文交流。", kind="semantic", fact_key="preferences"
    )
    event = memory.ingest(
        event_id=f"demo-{uuid4().hex}",
        session_id="demo-session",
        messages=[{"role": "user", "content": "今天用 FastAPI 完成了记忆服务的首次接入。"}],
    )
    print(memory.context("我喜欢什么开发语言？").context)
    print("Event:", event.memories[0].id)
    print("Stats:", memory.stats())
