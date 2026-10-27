"""Deterministic lexical/hybrid retrieval and explicit memory-decay heuristics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from datetime import UTC, datetime

from fishkeeping.models import Memory, RecallHit

_WORDS = re.compile(
    r"[\u3400-\u9fff]+|[^\W_\u3400-\u9fff]+(?:[_'-][^\W_\u3400-\u9fff]+)*", re.UNICODE
)
_CJK = re.compile(r"^[\u3400-\u9fff]+$")
_HALF_LIFE_DAYS = {"working": 1 / 24, "episodic": 14, "semantic": 90, "procedural": 180}
_QUERY_STOPWORDS = frozenset(
    "a an the is are was were am be been being do does did have has had "
    "what which who whom whose when where why how i me my we our you your "
    "he she it its they their this that these those of to in on at for from "
    "with and or as by about please tell can could would should".split()
)


def tokenize(text: str) -> list[str]:
    """English/unicode words and adjacent CJK pairs, without a model download."""
    result = []
    normalized = unicodedata.normalize("NFKC", text).casefold()
    for token in _WORDS.findall(normalized):
        if _CJK.fullmatch(token):
            if len(token) == 1:
                result.append(token)
            else:
                result.extend(token[index : index + 2] for index in range(len(token) - 1))
        else:
            result.append(token)
    return result


def query_terms(text: str) -> list[str]:
    """Ignore English grammatical words that alone are not evidence of relevance.

    Keep negation, temporal relations and technical names; do not rewrite stored
    text or change document tokenization, so existing FTS indexes remain valid.
    """
    return [token for token in tokenize(text) if token not in _QUERY_STOPWORDS]


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def retention(memory: Memory, now: datetime | None = None) -> float:
    """Half-life decay; only explicit rehearsal updates the strength anchor."""
    current = _utc(now or datetime.now(UTC))
    if memory.expires_at is not None and _utc(memory.expires_at) <= current:
        return 0.0
    if memory.pinned:
        return 1.0
    anchor = memory.last_accessed_at or memory.created_at
    elapsed_days = max(0.0, (current - _utc(anchor)).total_seconds() / 86400)
    strength = 1 + 0.5 * math.log1p(max(0, memory.access_count))
    half_life = _HALF_LIFE_DAYS[memory.kind] * (0.5 + memory.importance) * strength
    return 2.0 ** (-elapsed_days / half_life)


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    try:
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in left + right):
            return 0.0
        norm_left, norm_right = math.hypot(*left), math.hypot(*right)
        if not norm_left or not norm_right or not math.isfinite(norm_left + norm_right):
            return 0.0
        value = sum((a / norm_left) * (b / norm_right) for a, b in zip(left, right, strict=True))
        return min(1.0, max(0.0, value))
    except (ValueError, TypeError, OverflowError):
        return 0.0


def rank(
    memories: list[Memory],
    query: str,
    limit: int = 8,
    query_vector: list[float] | None = None,
    vectors: dict[str, list[float]] | None = None,
    now: datetime | None = None,
    semantic_threshold: float = 0.35,
) -> list[RecallHit]:
    """Score relevant active records, never mutating memory or reinforcing reads."""
    current = _utc(now or datetime.now(UTC))
    candidates = [
        memory
        for memory in memories
        if memory.status == "active"
        and (memory.expires_at is None or _utc(memory.expires_at) > current)
    ]
    if not candidates or limit <= 0:
        return []
    query_tokens = set(query_terms(query))
    if not query_tokens and query_vector is None:
        return []
    documents = [Counter(tokenize(memory.content)) for memory in candidates]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())
    average_length = sum(sum(document.values()) for document in documents) / len(documents)
    average_length = max(average_length, 1.0)
    hits = []
    for memory, document in zip(candidates, documents, strict=True):
        raw_bm25 = 0.0
        length = sum(document.values())
        for token in query_tokens:
            frequency = document.get(token, 0)
            if not frequency:
                continue
            idf = math.log(
                1
                + (len(documents) - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denominator = frequency + 1.2 * (0.25 + 0.75 * length / average_length)
            raw_bm25 += idf * frequency * 2.2 / denominator
        lexical = raw_bm25 / (1 + raw_bm25)
        semantic = 0.0
        if query_vector is not None and vectors and memory.id in vectors:
            # Threshold avoids treating every positive embedding cosine as relevant.
            cosine = _cosine(query_vector, vectors[memory.id])
            semantic = max(0.0, (cosine - semantic_threshold) / (1 - semantic_threshold))
        relevance = max(lexical, 0.75 * semantic + 0.25 * lexical)
        if relevance <= 0:
            continue
        retained = retention(memory, current)
        age_days = max(0.0, (current - _utc(memory.created_at)).total_seconds() / 86400)
        recency = 1 / (1 + age_days / 7)
        # Relevance gates all secondary signals, preventing unrelated pinned hits.
        score = relevance * (0.65 + 0.15 * memory.importance + 0.10 * retained + 0.10 * recency)
        hits.append(RecallHit(memory=memory, score=score, relevance=relevance, retention=retained))
    hits.sort(key=lambda hit: (-hit.score, -_utc(hit.memory.created_at).timestamp(), hit.memory.id))
    return hits[:limit]
