"""Tests for memory store."""

import pytest
from unittest.mock import Mock, MagicMock, patch
from src.agent.memory import MemoryStore


@pytest.fixture
def mock_db_connection():
    """Mock database connection."""
    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor
    conn.__enter__.return_value = conn
    conn.__exit__.return_value = None
    return conn, cursor


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    with patch('src.agent.memory.OpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        # Mock embedding response
        client.embeddings.create.return_value.data = [
            MagicMock(embedding=[0.1] * 1536)
        ]
        yield client


@pytest.fixture
def store(mock_openai):
    """Create a memory store instance with mocked dependencies."""
    with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
        return MemoryStore()


def test_remember(store, mock_db_connection, mock_openai):
    """Test storing a memory."""
    conn, cursor = mock_db_connection
    cursor.fetchone.return_value = [123]
    
    with patch.object(store, '_get_connection', return_value=conn):
        mem_id = store.remember(
            memory_text="Test memory",
            type="fact",
            context="test-context",
            confidence=1.0
        )
    
    assert mem_id == 123
    assert cursor.execute.called
    assert conn.commit.called


def test_recall_semantic(store, mock_db_connection, mock_openai):
    """Test semantic search recall."""
    conn, cursor = mock_db_connection
    cursor.fetchall.return_value = [
        {
            'id': 1,
            'memory_text': 'Test memory',
            'type': 'fact',
            'context': 'test',
            'confidence': 1.0,
            'source_context': None,
            'created_at': '2024-01-01',
            'last_accessed': '2024-01-01',
            'access_count': 1,
            'embedding_model': 'text-embedding-3-small',
            'similarity': 0.95
        }
    ]
    
    with patch.object(store, '_get_connection', return_value=conn):
        results = store.recall("test query", limit=5)
    
    assert len(results) == 1
    assert results[0]['memory_text'] == 'Test memory'
    assert results[0]['similarity'] == 0.95


def test_recall_with_filters(store, mock_db_connection, mock_openai):
    """Test recall with type and context filters."""
    conn, cursor = mock_db_connection
    cursor.fetchall.return_value = []
    
    with patch.object(store, '_get_connection', return_value=conn):
        results = store.recall(
            "test query",
            type="preference",
            context="test-context",
            limit=5
        )
    
    assert isinstance(results, list)
    # Verify SQL was called with filters
    assert cursor.execute.called


def test_forget(store, mock_db_connection):
    """Test deleting a memory."""
    conn, cursor = mock_db_connection
    cursor.rowcount = 1
    
    with patch.object(store, '_get_connection', return_value=conn):
        result = store.forget(123)
    
    assert result is True
    assert cursor.execute.called
    assert conn.commit.called


def test_list_contexts(store, mock_db_connection):
    """Test listing unique contexts."""
    conn, cursor = mock_db_connection
    cursor.fetchall.return_value = [('work',), ('personal',), ('project',)]
    
    with patch.object(store, '_get_connection', return_value=conn):
        contexts = store.list_contexts()
    
    assert contexts == ['work', 'personal', 'project']


def test_stats(store, mock_db_connection):
    """Test memory statistics."""
    conn, cursor = mock_db_connection
    cursor.fetchone.return_value = {
        'total_memories': 42,
        'unique_types': 4,
        'unique_contexts': 3,
        'avg_confidence': 0.95,
        'last_memory_at': '2024-01-01'
    }
    
    with patch.object(store, '_get_connection', return_value=conn):
        stats = store.stats()
    
    assert stats['total_memories'] == 42
    assert stats['unique_types'] == 4
    assert 'avg_confidence' in stats
