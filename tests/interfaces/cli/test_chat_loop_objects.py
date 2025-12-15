import builtins
from unittest.mock import MagicMock
from src.interfaces.cli import chat
from src.services.llm.ollama_service import OllamaService


class MockMessage:
    def __init__(self, content="", thinking="", tool_calls=None):
        self.content = content
        self.thinking = thinking
        self.tool_calls = tool_calls
        self.role = "assistant"


class MockResponse:
    def __init__(self, message):
        self.message = message

    # Simulate missing .get() by NOT implementing it
    # If code uses .get(), it raises AttributeError


class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, function):
        self.function = function


def test_chat_loop_object_response(tmp_path, monkeypatch, capsys):
    """Test chat loop with object-based responses (Ollama 0.4.0+ style)"""
    template = {
        "model": "llama3.2",
        "system": "You are a test",
        "parameters": {"num_ctx": 1024},
    }
    mem_file = tmp_path / "memory.json"
    mem_file.write_text('{"timestamp": "t", "messages": []}')

    # Patch OllamaService
    monkeypatch.setattr(OllamaService, "check_connection", lambda self: True)

    # Mock chat yielding objects instead of dicts
    def mock_chat(*args, **kwargs):
        if kwargs.get("stream"):

            def stream_gen():
                # Stream yield 1: Content only
                msg1 = MockMessage(content="Hello ")
                yield MockResponse(msg1)

                # Stream yield 2: Thinking
                msg2 = MockMessage(content="", thinking="Hmm...")
                yield MockResponse(msg2)

                # Stream yield 3: Content + Tool calls
                tc = MockToolCall(
                    MockFunction(
                        "store_memory_tool", {"text": "something", "type": "fact"}
                    )
                )
                msg3 = MockMessage(content="World", tool_calls=[tc])
                yield MockResponse(msg3)

            return stream_gen()
        else:
            # Non-streaming response
            tc = MockToolCall(
                MockFunction(
                    "store_memory_tool", {"text": "non-stream", "type": "fact"}
                )
            )
            msg = MockMessage(content="Hello Non-Stream", tool_calls=[tc])
            return MockResponse(msg)

    monkeypatch.setattr(OllamaService, "chat", mock_chat)
    monkeypatch.setattr(chat, "save_chat_history", lambda messages, file_path: None)

    # Mock create_store_memory_tool
    mock_tool_func = MagicMock(return_value="Memory saved")
    monkeypatch.setattr(chat, "create_store_memory_tool", lambda store: mock_tool_func)

    # Inputs:
    # 1. "hello" -> triggers streaming response
    # 2. "/stream" -> toggles to non-streaming
    # 3. "hello again" -> triggers non-streaming response
    # 4. "/quit" -> exit
    inputs = iter(["hello", "/stream", "hello again", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    # Dummy memory store
    class DummyStore:
        def close(self):
            pass

    # This should NOT raise AttributeError if fixed
    chat.chat_loop(template, context_file=str(mem_file), memory_store=DummyStore())

    out = capsys.readouterr().out

    # Verify we got the content printed (stream)
    assert "Hello World" in out
    # Verify non-stream content
    assert "Hello Non-Stream" in out
    # Verify tool execution (both stream and non-stream calls trigger it, checking one is enough but logic implies both run)
    assert "Memory saved" in out
    # Verify no error
    assert "❌ Error" not in out
