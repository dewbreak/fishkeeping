"""Behavioral regressions for repeated, already-consolidated knowledge."""

from types import SimpleNamespace

import pytest

from fishkeeping.config import Settings
from fishkeeping.models import ConsolidateRequest, ExtractedFact, IngestRequest, MemoryInput, Scope
from fishkeeping.service import MemoryService
from fishkeeping.storage import InvalidSourceError, Store

SCOPE = Scope(user_id="quality-regression")


def test_conversation_consolidation_cannot_promote_assistant_claims(service):
    event = service.ingest(
        IngestRequest(
            scope=SCOPE,
            event_id="roles",
            session_id="s",
            messages=[
                {"role": "user", "content": "用户喜欢茶"},
                {"role": "assistant", "content": "用户住在火星"},
            ],
        )
    )
    seen = []

    def consolidate(sources):
        seen.extend(sources)
        return [ExtractedFact(content="用户喜欢茶")]

    service.llm = SimpleNamespace(consolidate=consolidate)
    result = service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[event.memories[0].id]))
    assert "喜欢茶" in seen[0].content and "火星" not in seen[0].content
    assert result["memories"][0].source_ids == [event.memories[0].id]
    assert "火星" in service.store.get(SCOPE, event.memories[0].id).content  # Audit data retained.


def test_conversation_with_no_user_evidence_does_not_call_model(service):
    event = service.ingest(
        IngestRequest(
            scope=SCOPE,
            event_id="assistant-only",
            session_id="s",
            messages=[{"role": "assistant", "content": "用户住在火星"}],
        )
    )
    service.llm = SimpleNamespace(consolidate=lambda _: pytest.fail("model called"))
    result = service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[event.memories[0].id]))
    assert result["memories"] == []


def test_unstructured_legacy_conversation_is_not_guessed(service):
    old = service.store.put(
        MemoryInput(
            scope=SCOPE,
            content="[user] first\n[assistant] invented",
            metadata={"method": "conversation"},
        )
    )
    service.llm = SimpleNamespace(consolidate=lambda _: pytest.fail("model called"))
    with pytest.raises(InvalidSourceError, match="provenance"):
        service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[old.id]))


def test_ignored_assistant_record_is_not_linked_as_supporting_evidence(service):
    event = service.ingest(
        IngestRequest(
            scope=SCOPE,
            event_id="assistant-only",
            session_id="s",
            messages=[{"role": "assistant", "content": "unsupported claim"}],
        )
    )
    valid = service.store.put(MemoryInput(scope=SCOPE, content="用户喜欢中文"))
    result = service.consolidate(
        ConsolidateRequest(scope=SCOPE, source_ids=[event.memories[0].id, valid.id])
    )
    assert result["memories"][0].source_ids == [valid.id]


@pytest.fixture
def service():
    store = Store(":memory:")
    settings = Settings(_env_file=None)
    result = MemoryService(store, settings)
    yield result
    store.close()


@pytest.mark.parametrize("kind", ["semantic", "procedural"])
def test_consolidation_can_repeat_a_source_without_creating_a_self_reference(service, kind):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="使用 Python 处理数据。", kind=kind)
    )
    service.llm = SimpleNamespace(
        consolidate=lambda sources: [ExtractedFact(content=original.content, kind=kind)]
    )

    result = service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[original.id]))

    assert result["method"] == "deepseek"
    assert service.store.stats(SCOPE)["total"] == 1
    assert service.store.get(SCOPE, original.id).source_ids == []
    assert all(memory.id not in memory.source_ids for memory in result["memories"])


def test_noop_detection_respects_the_same_unicode_equivalence_as_storage(service):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户喜欢 café。", kind="semantic")
    )
    service.llm = SimpleNamespace(
        consolidate=lambda sources: [ExtractedFact(content="用户喜欢 cafe\u0301。")]
    )

    service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[original.id]))

    assert service.store.stats(SCOPE)["total"] == 1
    assert service.store.get(SCOPE, original.id).source_ids == []


def test_repeating_an_ancestor_is_a_noop_but_other_new_conclusions_are_saved(service):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户偏好 Python。", kind="semantic")
    )
    derived = service.store.put(
        MemoryInput(
            scope=SCOPE,
            content="用户处理数据时偏好 Python，也接受 SQL 示例。",
            kind="semantic",
            source_ids=[original.id],
        )
    )
    novel = "用户也接受 SQL 示例。"
    service.llm = SimpleNamespace(
        consolidate=lambda sources: [
            ExtractedFact(content=original.content),
            ExtractedFact(content=novel),
        ]
    )

    result = service.consolidate(ConsolidateRequest(scope=SCOPE, source_ids=[derived.id]))

    assert result["method"] == "deepseek"
    assert service.store.stats(SCOPE)["total"] == 3
    assert service.store.get(SCOPE, original.id).source_ids == []
    saved_novel = next(memory for memory in service.store.list(SCOPE) if memory.content == novel)
    assert saved_novel.source_ids == [derived.id]


def test_noop_consolidation_still_merges_independent_supporting_sources(service):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户偏好 Python。", kind="semantic")
    )
    independent = service.store.put(
        MemoryInput(scope=SCOPE, content="今天用户再次明确说偏好 Python。", kind="episodic")
    )
    service.llm = SimpleNamespace(
        consolidate=lambda sources: [ExtractedFact(content=original.content)]
    )

    result = service.consolidate(
        ConsolidateRequest(scope=SCOPE, source_ids=[original.id, independent.id])
    )

    assert [memory.id for memory in result["memories"]] == [original.id]
    assert service.store.get(SCOPE, original.id).source_ids == [independent.id]
    assert service.store.stats(SCOPE)["total"] == 2


def test_consolidation_reuse_does_not_weaken_normal_write_cycle_validation(service):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户偏好 Python。", kind="semantic")
    )
    cyclic = MemoryInput(
        scope=SCOPE, content=original.content, kind="semantic", source_ids=[original.id]
    )

    with pytest.raises(InvalidSourceError, match="cycle"):
        service.store.put(cyclic)

    assert service.store.get(SCOPE, original.id).source_ids == []


def test_consolidation_reuse_rolls_back_source_merges_when_another_item_is_invalid(service):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户偏好 Python。", kind="semantic")
    )
    independent = service.store.put(
        MemoryInput(scope=SCOPE, content="第二次独立确认 Python 偏好。")
    )
    before = service.store.get(SCOPE, original.id)
    duplicate = MemoryInput(
        scope=SCOPE,
        content=original.content,
        kind="semantic",
        source_ids=[original.id, independent.id],
    )
    invalid = MemoryInput(
        scope=SCOPE, content="另一个结论", kind="semantic", source_ids=["missing-source"]
    )

    with pytest.raises(InvalidSourceError):
        service.store.put_many(
            [duplicate, invalid], consolidation=True, require_active_sources=True
        )

    assert service.store.get(SCOPE, original.id) == before
    assert service.store.stats(SCOPE)["total"] == 2


@pytest.mark.parametrize("source_state", ["archived", "foreign"])
def test_consolidation_checks_every_requested_source_before_omitting_cycle_edges(
    service, source_state
):
    original = service.store.put(
        MemoryInput(scope=SCOPE, content="用户偏好 Python。", kind="semantic")
    )
    source = service.store.put(
        MemoryInput(
            scope=SCOPE if source_state == "archived" else Scope(user_id="different-user"),
            content="不可用的来源",
        )
    )
    if source_state == "archived":
        service.store.archive(SCOPE, [source.id])
    duplicate = MemoryInput(
        scope=SCOPE,
        content=original.content,
        kind="semantic",
        source_ids=[original.id, source.id],
    )

    with pytest.raises(InvalidSourceError):
        service.store.put_many([duplicate], consolidation=True, require_active_sources=True)

    assert service.store.get(SCOPE, original.id).source_ids == []


@pytest.mark.parametrize(
    "event_args",
    [
        {"event_id": "e", "event_hash": "h"},
        {"event_hash": "h"},
        {"source_event": True},
    ],
)
def test_consolidation_mode_cannot_be_used_as_event_ingestion(service, event_args):
    with pytest.raises(ValueError, match="consolidation"):
        service.store.put_many([], consolidation=True, **event_args)
