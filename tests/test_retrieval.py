from datetime import UTC, datetime, timedelta

import pytest

from fishkeeping.models import Memory, Scope
from fishkeeping.retrieval import rank, retention, tokenize

NOW = datetime(2026, 10, 28, tzinfo=UTC)


def memory(content, identifier="test", age=0, **values):
    created = NOW - timedelta(days=age)
    return Memory(
        scope=Scope(user_id="alice"),
        content=content,
        id=identifier,
        created_at=created,
        updated_at=created,
        **values,
    )


def test_tokenize_english_chinese_and_unicode():
    assert tokenize("ＰＹＴＨＯＮ Python") == ["python", "python"]
    assert tokenize("我喜欢乌龙茶") == ["我喜", "喜欢", "欢乌", "乌龙", "龙茶"]
    assert tokenize("茶") == ["茶"]
    assert tokenize("🐟 , !") == []


def test_english_and_chinese_recall():
    records = [
        memory("I prefer Python for scripting", "english"),
        memory("用户喜欢乌龙茶", "chinese"),
        memory("The moon is bright", "unrelated", importance=1, pinned=True),
    ]
    assert [hit.memory.id for hit in rank(records, "python", now=NOW)] == ["english"]
    assert [hit.memory.id for hit in rank(records, "乌龙茶", now=NOW)] == ["chinese"]
    assert rank(records, "cinnamon", now=NOW) == []
    assert rank(records, "!!!", now=NOW) == []


def test_mixed_chinese_english_without_spaces_stays_searchable():
    assert tokenize("Python开发ABC工具") == ["python", "开发", "abc", "工具"]
    assert tokenize("用户使用SQLite数据库") == ["用户", "户使", "使用", "sqlite", "数据", "据库"]
    records = [memory("用户使用Python开发SQLite工具", "mixed")]
    for query in ["Python", "SQLite", "开发", "工具"]:
        assert [hit.memory.id for hit in rank(records, query, now=NOW)] == ["mixed"]


def test_retention_halflife_and_rehearsal():
    record = memory("event", age=14, kind="episodic", importance=0.5)
    assert retention(record, NOW) == pytest.approx(0.5)
    rehearsed = record.model_copy(update={"access_count": 3})
    assert retention(rehearsed, NOW) > retention(record, NOW)
    refreshed = record.model_copy(update={"last_accessed_at": NOW})
    assert retention(refreshed, NOW) == 1
    future = memory("future event", age=-1)
    assert retention(future, NOW) == 1


def test_pinning_does_not_override_expiration():
    pinned = memory("tea", age=1000, pinned=True)
    assert retention(pinned, NOW) == 1
    expired = pinned.model_copy(update={"expires_at": NOW})
    assert retention(expired, NOW) == 0
    assert rank([expired], "tea", now=NOW) == []


def test_ranking_excludes_inactive_and_prefers_recent_related_memory():
    records = [
        memory("user drinks tea", "old", age=60),
        memory("user drinks tea", "new", age=1),
        memory("user drinks tea", "archived", status="archived"),
        memory("user drinks tea", "superseded", status="superseded"),
    ]
    before = [record.model_dump() for record in records]
    hits = rank(records, "tea", now=NOW)
    assert [hit.memory.id for hit in hits] == ["new", "old"]
    assert all(0 < hit.score <= 1 for hit in hits)
    assert [record.model_dump() for record in records] == before


def test_hybrid_retrieves_semantic_match_without_lexical_overlap():
    records = [memory("I enjoy green tea", "match"), memory("bicycle wheels", "miss")]
    hits = rank(
        records,
        "我爱喝茶",
        query_vector=[1, 0],
        vectors={"match": [0.9, 0.1], "miss": [0, 1]},
        now=NOW,
    )
    assert [hit.memory.id for hit in hits] == ["match"]


@pytest.mark.parametrize(
    "vector", [[], [0, 0], [1], [float("nan"), 0], [float("inf"), 0], [-1, 0], [True, 0]]
)
def test_invalid_or_negative_vectors_do_not_create_hits(vector):
    hits = rank(
        [memory("green tea")],
        "coffee",
        query_vector=[1, 0],
        vectors={"test": vector},
        now=NOW,
    )
    assert hits == []


def test_semantic_failure_preserves_lexical_result():
    hits = rank(
        [memory("green tea")],
        "tea",
        query_vector=[1, 0],
        vectors={"test": [0, 0]},
        now=NOW,
    )
    assert len(hits) == 1


def test_limit_and_stable_tie_break():
    records = [memory("tea", "b"), memory("tea", "a")]
    assert rank(records, "tea", limit=1, now=NOW)[0].memory.id == "a"
    assert rank(records, "tea", limit=0, now=NOW) == []
