import pytest
from unittest.mock import MagicMock, patch
import os


def make_store(monkeypatch):
    # Ensure env vars
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://x")

    class DummyOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = MagicMock()

    # Patch SimpleConnectionPool and OpenAI during instantiation
    with patch("src.agent.memory.pool.SimpleConnectionPool"):
        with patch("src.agent.memory.OpenAI", DummyOpenAI):
            from src.agent.memory import MemoryStore

            return MemoryStore()


def test_memory_init_missing_env(monkeypatch):
    # No env vars set
    monkeypatch.delenv("MEMORY_DB_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import src.agent.memory as memmod

    with patch("src.agent.memory.pool.SimpleConnectionPool"):
        with pytest.raises(ValueError):
            memmod.MemoryStore()


def test_remember_input_validations(monkeypatch):
    store = make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.remember("   ", "fact", "ctx")

    with pytest.raises(ValueError):
        store.remember("x" * (store.MAX_TEXT_LENGTH + 1), "fact", "ctx")

    with pytest.raises(ValueError):
        store.remember("ok", "notatype", "ctx")

    with pytest.raises(ValueError):
        store.remember("ok", "fact", "ctx", confidence=2.0)

    with pytest.raises(ValueError):
        store.remember("ok", "fact", "   ")


def test_recall_input_validations(monkeypatch):
    store = make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.recall("   ")

    with pytest.raises(ValueError):
        store.recall("q", type="badtype")

    with pytest.raises(ValueError):
        store.recall("q", limit=0)

    with pytest.raises(ValueError):
        store.recall("q", limit=101)


def test_get_embedding_cached(monkeypatch):
    store = make_store(monkeypatch)

    # Patch the underlying _get_embedding to a deterministic value
    store._get_embedding = lambda text: [0.5, 0.1]

    t1 = store._get_embedding_cached("hello")
    t2 = store._get_embedding_cached("hello")
    assert isinstance(t1, tuple)
    assert t1 == t2


def test_forget_list_stats_db_errors(monkeypatch):
    store = make_store(monkeypatch)

    # Patch _get_connection to raise an error to simulate DB down
    def raise_err():
        raise RuntimeError("db down")

    store._get_connection = raise_err

    assert store.forget(1) is False
    assert store.list_contexts() == []
    assert store.stats() is None
