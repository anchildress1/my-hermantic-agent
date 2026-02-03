from unittest.mock import MagicMock
from src.tools.memory_tool import create_store_memory_tool
from src.services.memory.vector_store import MemoryStore


def test_store_memory_tool_success():
    mock_store = MagicMock(spec=MemoryStore)
    mock_store.remember.return_value = 123

    tool = create_store_memory_tool(mock_store)

    result = tool(
        memory_text="User likes blue", type="preference", importance=2.0, confidence=0.9
    )

    assert result == "Stored memory #123"
    mock_store.remember.assert_called_once_with(
        memory_text="User likes blue",
        type="preference",
        context="chat",
        importance=2.0,
        confidence=0.9,
    )


def test_store_memory_tool_failure_none_id():
    mock_store = MagicMock(spec=MemoryStore)
    mock_store.remember.return_value = None

    tool = create_store_memory_tool(mock_store)

    result = tool("text", "fact")
    assert result == "Failed to store memory"


def test_store_memory_tool_exception():
    mock_store = MagicMock(spec=MemoryStore)
    mock_store.remember.side_effect = Exception("DB Error")

    tool = create_store_memory_tool(mock_store)

    result = tool("text", "fact")
    assert "Error: DB Error" in result


def test_store_memory_tool_no_store():
    tool = create_store_memory_tool(None)
    result = tool("text", "fact")
    assert result == "Error: Memory store not available"
