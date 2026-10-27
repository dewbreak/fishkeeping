"""Preserve message boundaries independently of role-like text inside messages.

The digest checks internal consistency; it is not a signature or authorization
mechanism. The trusted application still controls each ingestion's actual roles.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import ValidationError

from fishkeeping.models import Memory, Message

_ROLES = {"user", "assistant", "system", "tool"}


def _digest(messages: list[Message]) -> str:
    canonical = json.dumps(
        [message.model_dump() for message in messages],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def encode_conversation(messages: list[Message]) -> tuple[str, dict]:
    """Return the readable transcript plus small, unambiguous message spans."""
    if not isinstance(messages, list) or not 1 <= len(messages) <= 100:
        raise ValueError("Conversation requires between 1 and 100 messages")
    try:
        if any(not isinstance(message, Message) for message in messages):
            raise ValueError("Conversation messages must use the Message schema")
        validated = [
            Message.model_validate(message.model_dump(), strict=True) for message in messages
        ]
    except ValidationError:
        raise ValueError("Conversation contains an invalid message") from None
    if sum(len(message.content) for message in validated) > 12000:
        raise ValueError("Conversation content must not exceed 12000 characters")
    parts = []
    spans = []
    cursor = 0
    for message in validated:
        prefix = f"[{message.role}] "
        start = cursor + len(prefix)
        end = start + len(message.content)
        parts.append(prefix + message.content)
        spans.append({"role": message.role, "start": start, "end": end})
        cursor = end + 1  # One separator between this message and the next.
    return "\n".join(parts), {
        "method": "conversation",
        "message_spans": spans,
        "conversation_fingerprint": _digest(validated),
    }


def trusted_conversation_content(memory: Memory) -> str | None:
    """Use only structurally verified user messages as conversation evidence.

    Return non-conversation records unchanged. A conversation with no user
    messages returns None. Old or inconsistent conversation records fail closed
    instead of guessing roles from strings appearing in their contents.
    """
    if memory.metadata.get("method") != "conversation":
        return memory.content
    spans = memory.metadata.get("message_spans")
    fingerprint = memory.metadata.get("conversation_fingerprint")
    if not isinstance(spans, list) or not 1 <= len(spans) <= 100:
        raise ValueError(
            "Conversation has no valid message spans; re-ingest it or use extracted facts"
        )
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError("Conversation fingerprint is missing or invalid")
    messages = []
    previous_end = 0
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != {"role", "start", "end"}:
            raise ValueError("Conversation contains an invalid message span")
        role, start, end = span["role"], span["start"], span["end"]
        if not isinstance(role, str) or role not in _ROLES:
            raise ValueError("Conversation contains an invalid message role")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(memory.content)
        ):
            raise ValueError("Conversation contains an invalid message range")
        prefix_start = previous_end + 1 if index else 0
        prefix = f"[{role}] "
        if index and memory.content[previous_end:prefix_start] != "\n":
            raise ValueError("Conversation message separator is invalid")
        if start != prefix_start + len(prefix) or memory.content[prefix_start:start] != prefix:
            raise ValueError("Conversation message boundaries do not match the transcript")
        content = memory.content[start:end]
        try:
            message = Message(role=role, content=content)
        except ValidationError:
            raise ValueError("Conversation contains invalid message content") from None
        if message.content != content:
            raise ValueError("Conversation message content was not preserved exactly")
        messages.append(message)
        previous_end = end
    if (
        previous_end != len(memory.content)
        or sum(len(message.content) for message in messages) > 12000
    ):
        raise ValueError("Conversation message spans do not cover the complete transcript")
    if _digest(messages) != fingerprint:
        raise ValueError("Conversation fingerprint does not match its messages")
    user_messages = [message.content for message in messages if message.role == "user"]
    if not user_messages:
        return None
    return json.dumps({"user_messages": user_messages}, ensure_ascii=False, separators=(",", ":"))
