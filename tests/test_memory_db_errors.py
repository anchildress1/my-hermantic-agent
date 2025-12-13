from unittest.mock import MagicMock, patch


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


def test_remember_db_error_returns_none(monkeypatch):
    store = make_store(monkeypatch)

    # Create a fake connection that raises psycopg2.Error on execute
    class BadCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *a, **k):
            import psycopg2

            raise psycopg2.Error("boom")

    class BadConn:
        def cursor(self, *a, **k):
            return BadCursor()

        def commit(self):
            # noop for test (commit side-effect not needed)
            return None

        def rollback(self):
            # noop for test
            return None

    store._get_connection = lambda: BadConn()

    res = store.remember("ok", "fact", "ctx")
    assert res is None


def test_recall_db_error_returns_empty(monkeypatch):
    store = make_store(monkeypatch)

    class BadCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *a, **k):
            import psycopg2

            raise psycopg2.Error("boom")

    class BadConn:
        def cursor(self, *a, **k):
            return BadCursor()

    store._get_connection = lambda: BadConn()

    res = store.recall("query")
    assert res == []


def test_recall_fulltext_path(monkeypatch):
    store = make_store(monkeypatch)

    # Provide a connection whose cursor returns expected rows
    class GoodCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, sql, params):
            # noop for test; just validate SQL path
            return None

        def fetchall(self):
            return []

    class GoodConn:
        def cursor(self, *a, **k):
            return GoodCursor()

    store._get_connection = lambda: GoodConn()

    res = store.recall("query", use_semantic=False)
    assert isinstance(res, list)
