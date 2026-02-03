from unittest.mock import MagicMock, patch
from src.services.memory.vector_store import MemoryStore


def make_store(monkeypatch):
    """Helper to create a MemoryStore for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://test")

    class DummyOpenAI:
        def __init__(self, *a, **k):
            self.embeddings = MagicMock()

    with patch("src.services.memory.vector_store.pool.SimpleConnectionPool"):
        with patch("src.services.memory.vector_store.OpenAI", DummyOpenAI):
            return MemoryStore()


def test_list_memories_no_filters(monkeypatch):
    """Test listing memories without filters."""
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"id": 1, "memory_text": "Test 1", "tag": "work", "type": "fact"},
        {"id": 2, "memory_text": "Test 2", "tag": "personal", "type": "task"},
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store._get_connection = lambda: mock_conn
    store._return_connection = lambda c: None

    results = store.list_memories()

    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[1]["id"] == 2
    mock_cursor.execute.assert_called_once()
    # Verify default limit=20, offset=0
    call_args = mock_cursor.execute.call_args[0]
    assert call_args[1] == [20, 0]


def test_list_memories_with_tag_filter(monkeypatch):
    """Test listing memories filtered by tag."""
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"id": 1, "memory_text": "Work task", "tag": "work", "type": "task"},
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store._get_connection = lambda: mock_conn
    store._return_connection = lambda c: None

    results = store.list_memories(tag="work")

    assert len(results) == 1
    assert results[0]["tag"] == "work"
    # Verify tag parameter was passed
    call_args = mock_cursor.execute.call_args[0]
    assert "work" in call_args[1]


def test_list_memories_with_type_filter(monkeypatch):
    """Test listing memories filtered by type."""
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {"id": 1, "memory_text": "Preference", "tag": "coding", "type": "preference"},
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store._get_connection = lambda: mock_conn
    store._return_connection = lambda c: None

    results = store.list_memories(type="preference")

    assert len(results) == 1
    assert results[0]["type"] == "preference"


def test_list_memories_with_pagination(monkeypatch):
    """Test pagination parameters."""
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = []
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store._get_connection = lambda: mock_conn
    store._return_connection = lambda c: None

    store.list_memories(limit=10, offset=5)

    call_args = mock_cursor.execute.call_args[0]
    # Verify limit and offset in params
    assert 10 in call_args[1]
    assert 5 in call_args[1]


def test_list_memories_invalid_limit():
    """Test that invalid limit raises ValueError."""
    from unittest.mock import MagicMock

    store = MagicMock(spec=MemoryStore)
    store.VALID_TYPES = {"preference", "fact", "task", "insight"}

    # Call the actual method
    from src.services.memory.vector_store import MemoryStore as RealStore

    real_store = RealStore.__new__(RealStore)
    real_store.VALID_TYPES = store.VALID_TYPES

    try:
        real_store.list_memories(limit=0)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "limit must be between 1 and 100" in str(e)

    try:
        real_store.list_memories(limit=101)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "limit must be between 1 and 100" in str(e)


def test_list_memories_invalid_offset():
    """Test that negative offset raises ValueError."""
    from src.services.memory.vector_store import MemoryStore as RealStore

    real_store = RealStore.__new__(RealStore)
    real_store.VALID_TYPES = {"preference", "fact", "task", "insight"}

    try:
        real_store.list_memories(offset=-1)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "offset must be non-negative" in str(e)


def test_list_memories_invalid_type():
    """Test that invalid type raises ValueError."""
    from src.services.memory.vector_store import MemoryStore as RealStore

    real_store = RealStore.__new__(RealStore)
    real_store.VALID_TYPES = {"preference", "fact", "task", "insight"}

    try:
        real_store.list_memories(type="invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "type must be one of" in str(e)


def test_list_memories_db_error(monkeypatch):
    """Test that database errors are handled gracefully."""
    store = make_store(monkeypatch)

    def raise_error():
        import psycopg2

        raise psycopg2.Error("DB connection failed")

    store._get_connection = raise_error

    results = store.list_memories()
    assert results == []


def test_list_memories_combined_filters(monkeypatch):
    """Test combining tag and type filters."""
    store = make_store(monkeypatch)

    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = [
        {
            "id": 1,
            "memory_text": "Work preference",
            "tag": "work",
            "type": "preference",
        },
    ]
    mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
    store._get_connection = lambda: mock_conn
    store._return_connection = lambda c: None

    results = store.list_memories(tag="work", type="preference", limit=5)

    assert len(results) == 1
    call_args = mock_cursor.execute.call_args[0]
    # Verify both filters and limit are in params
    assert "work" in call_args[1]
    assert "preference" in call_args[1]
    assert 5 in call_args[1]
