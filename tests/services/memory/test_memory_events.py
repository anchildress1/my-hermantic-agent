from unittest.mock import MagicMock, patch

from src.services.memory.vector_store import MemoryStore


def make_store(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://test")

    class DummyOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = MagicMock()

    with patch("src.services.memory.vector_store.pool.SimpleConnectionPool"):
        with patch("src.services.memory.vector_store.OpenAI", DummyOpenAI):
            return MemoryStore()


def test_list_events_success(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "memory_id": 2,
            "operation": "remember",
            "status": "success",
            "details": {"memory_preview": "x"},
            "created_at": "2026-02-19T00:00:00Z",
        }
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None

    events = store.list_events(limit=10, operation="remember")
    assert len(events) == 1
    assert events[0]["operation"] == "remember"


def test_memory_exists_true(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = (1,)
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None

    exists = store.memory_exists("User prefers Python", "preference", "coding")
    assert exists is True


def test_record_event_does_not_raise_on_missing_table(monkeypatch):
    store = make_store(monkeypatch)

    class BrokenCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            raise RuntimeError("table missing")

    class BrokenConn:
        def cursor(self):
            return BrokenCursor()

        def rollback(self):
            return None

    store._get_connection = lambda: BrokenConn()
    store._return_connection = lambda _: None

    store.record_event(
        operation="remember",
        status=MemoryStore.EVENT_ERROR,
        details={"error": "x"},
    )


def test_revive_exact_memory_success(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = {
        "id": 3,
        "importance": 1.8,
        "access_count": 9,
        "last_accessed": "2026-02-19T00:00:00Z",
    }
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None

    row = store.revive_exact_memory("User prefers Python", "preference", "coding")
    assert row["id"] == 3
    assert row["importance"] == 1.8
    assert store.get_last_error() is None


def test_revive_exact_memory_db_error_sets_last_error(monkeypatch):
    store = make_store(monkeypatch)

    class BrokenCursor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def execute(self, *args, **kwargs):
            raise RuntimeError("deadlock detected")

        def fetchone(self):
            return None

    class BrokenConn:
        def cursor(self, *args, **kwargs):
            return BrokenCursor()

        def rollback(self):
            return None

    store._get_connection = lambda: BrokenConn()
    store._return_connection = lambda _: None

    row = store.revive_exact_memory("User prefers Python", "preference", "coding")
    assert row is None
    assert store.get_last_error()["operation"] == "revive_exact_memory"
