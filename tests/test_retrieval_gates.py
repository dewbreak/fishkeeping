import math

from fishkeeping.config import Settings
from fishkeeping.models import MemoryInput, RecallRequest, Scope
from fishkeeping.retrieval import query_terms, rank
from fishkeeping.service import MemoryService
from fishkeeping.storage import Store


def test_english_grammatical_overlap_does_not_retrieve_unrelated_records():
    scope = Scope(user_id="gates")
    store = Store(":memory:")
    try:
        store.put(MemoryInput(scope=scope, content="Backups are stored on an external drive."))
        store.put(MemoryInput(scope=scope, content="Bedtime is eleven at night."))
        service = MemoryService(store, Settings(_env_file=None))
        for query in ["How is a violin tuned?", "How are ceramic tiles manufactured?", "is are"]:
            assert store.lexical_candidates(scope, query) == []
            assert service.recall(RecallRequest(scope=scope, query=query)).hits == []
        assert service.recall(RecallRequest(scope=scope, query="Where are backups stored?")).hits
        assert query_terms("not before Python after Go C++") == [
            "not",
            "before",
            "python",
            "after",
            "go",
            "c",
        ]
    finally:
        store.close()


def test_semantic_threshold_is_a_model_specific_gate():
    scope = Scope(user_id="gates")
    store = Store(":memory:")
    try:
        memory = store.put(MemoryInput(scope=scope, content="记忆内容"))
        vector = [0.45, math.sqrt(1 - 0.45**2)]
        args = dict(query_vector=[1, 0], vectors={memory.id: vector})
        assert rank([memory], "unrelated", semantic_threshold=0.35, **args)
        assert rank([memory], "unrelated", semantic_threshold=0.50, **args) == []
        # Lexically supported records still work even when semantic similarity is weak.
        assert rank([memory], "记忆", semantic_threshold=0.50, **args)
    finally:
        store.close()
