import pytest
from unittest.mock import Mock, patch
from src.interfaces.cli.chat import chat_loop
from src.services.memory.vector_store import MemoryStore


@pytest.fixture
def mock_memory_store():
    store = Mock(spec=MemoryStore)
    store.remember.return_value = 42
    store.close = Mock()
    return store


@pytest.fixture
def mock_config():
    return {
        "model": "test-model",
        "system": "You are a test assistant.",
        "parameters": {"num_ctx": 8192},
    }


def test_tool_call_stores_memory(mock_memory_store, mock_config, monkeypatch):
    """Test that tool calls trigger memory storage."""

    def mock_stream(*args, **kwargs):
        yield {
            "message": {
                "content": "I'll remember that.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "store_memory_tool",
                            "arguments": {
                                "memory_text": "User prefers dark mode",
                                "type": "preference",
                                "importance": 1.5,
                                "confidence": 0.9,
                            },
                        }
                    }
                ],
            }
        }

    with (
        patch("src.interfaces.cli.chat.OllamaService") as MockService,
        patch("builtins.input", side_effect=["test message", "/bye"]),
        patch("src.interfaces.cli.chat.save_chat_history"),  # Mock file save
    ):
        service_instance = MockService.return_value
        service_instance.check_connection.return_value = True
        service_instance.chat.side_effect = mock_stream

        chat_loop(
            mock_config, context_file="/tmp/test.json", memory_store=mock_memory_store
        )

        mock_memory_store.remember.assert_called_once_with(
            memory_text="User prefers dark mode",
            type="preference",
            context="chat",
            importance=1.5,
            confidence=0.9,
        )
