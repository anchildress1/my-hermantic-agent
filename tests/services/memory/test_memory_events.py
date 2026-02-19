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
