from unittest.mock import MagicMock, patch

import pytest

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


def test_list_events_invalid_limit_raises(monkeypatch):
    store = make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.list_events(limit=0)


def test_prune_events_success(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.rowcount = 7
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None

    deleted = store.prune_events(retention_days=14)
    assert deleted == 7
    assert "DELETE FROM hermes.memory_events" in mock_cursor.execute.call_args[0][0]
    assert mock_cursor.execute.call_args[0][1] == (14,)


def test_prune_events_invalid_days_raises(monkeypatch):
    store = make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.prune_events(retention_days=0)


def test_record_event_triggers_prune_when_due(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None
    store._next_event_prune_monotonic = 0.0
    store._prune_events_with_connection = MagicMock(return_value=3)

    store.record_event(
        operation="remember",
        status=MemoryStore.EVENT_SUCCESS,
        details={"memory_preview": "x"},
        memory_id=11,
    )

    store._prune_events_with_connection.assert_called_once_with(
        conn=mock_conn,
        retention_days=store.memory_events_retention_days,
    )


def test_record_event_skips_prune_before_interval(monkeypatch):
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None
    store._next_event_prune_monotonic = float("inf")
    store._prune_events_with_connection = MagicMock(return_value=3)

    store.record_event(
        operation="remember",
        status=MemoryStore.EVENT_SUCCESS,
        details={"memory_preview": "x"},
    )

    store._prune_events_with_connection.assert_not_called()


def test_event_retention_env_overrides_defaults(monkeypatch):
    monkeypatch.setenv("MEMORY_EVENTS_RETENTION_DAYS", "21")
    monkeypatch.setenv("MEMORY_EVENTS_PRUNE_INTERVAL_SECONDS", "15")
    store = make_store(monkeypatch)

    assert store.memory_events_retention_days == 21
    assert store.event_prune_interval_seconds == 15


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


def test_revive_exact_memory_no_match_clears_previous_last_error(monkeypatch):
    store = make_store(monkeypatch)

    # Simulate stale previous error state
    store._set_last_error("remember", "old error", {"memory_preview": "x"})

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchone.return_value = None
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor

    store._get_connection = lambda: mock_conn
    store._return_connection = lambda _: None

    row = store.revive_exact_memory("User prefers Python", "preference", "coding")
    assert row is None
    assert store.get_last_error() is None


def test_revive_exact_memory_invalid_boost_raises(monkeypatch):
    store = make_store(monkeypatch)

    with pytest.raises(ValueError):
        store.revive_exact_memory(
            "User prefers Python",
            "preference",
            "coding",
            importance_boost=0.0,
        )
