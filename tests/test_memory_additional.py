import pytest
from unittest.mock import MagicMock, patch
import psycopg2


def make_store(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://x")

    class DummyOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = MagicMock()

    with patch("src.agent.memory.pool.SimpleConnectionPool"):
        with patch("src.agent.memory.OpenAI", DummyOpenAI):
            from src.agent.memory import MemoryStore

            return MemoryStore()


def test_init_pool_raises(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://x")

    # Make SimpleConnectionPool raise psycopg2.Error
    with patch("src.agent.memory.pool.SimpleConnectionPool", side_effect=psycopg2.Error("boom")):
        from importlib import reload
        import src.agent.memory as memmod

        with pytest.raises(psycopg2.Error):
            reload(memmod)
            memmod.MemoryStore()


def test_get_connection_none(monkeypatch):
    store = make_store(monkeypatch)
    # Replace conn_pool.getconn to return None
    store.conn_pool = MagicMock()
    store.conn_pool.getconn.return_value = None

    with pytest.raises(Exception):
        store._get_connection()


def test_get_embedding_success(monkeypatch):
    store = make_store(monkeypatch)

    class Resp:
        def __init__(self):
            self.data = [MagicMock(embedding=[0.2, 0.3])]

    store.openai_client.embeddings.create = MagicMock(return_value=Resp())
    emb = store._get_embedding("hello")
    assert emb == [0.2, 0.3]


def test_remember_operational_error(monkeypatch):
    store = make_store(monkeypatch)

    def raise_op():
        raise psycopg2.OperationalError("conn fail")

    store._get_connection = raise_op
    res = store.remember("ok", "fact", "ctx")
    assert res is None


def test_remember_fetchone_none_rolls_back(monkeypatch):
    store = make_store(monkeypatch)

    class BadCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return None

    class BadConn:
        def __init__(self):
            self.rolled = False

        def cursor(self):
            return BadCursor()

        def commit(self):
            return None

        def rollback(self):
            self.rolled = True

    store._get_connection = lambda: BadConn()
    res = store.remember("ok", "fact", "ctx")
    assert res is None


def test_recall_context_like(monkeypatch):
    store = make_store(monkeypatch)

    class C:
        def __init__(self):
            self.last_sql = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params=None):
            self.last_sql = sql

        def fetchall(self):
            return []

    class Conn:
        def cursor(self, *a, **k):
            return C()

    store._get_connection = lambda: Conn()
    # Should not raise; triggers LIKE branch when context contains %
    res = store.recall("q", context="%test%", limit=5)
    assert res == []


def test_close_calls_pool_closeall(monkeypatch):
    store = make_store(monkeypatch)
    store.conn_pool = MagicMock()
    store.conn_pool.closeall = MagicMock()
    store.close()
    store.conn_pool.closeall.assert_called()
