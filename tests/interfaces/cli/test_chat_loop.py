import builtins

from src.interfaces.cli import chat
from src.services.llm.ollama_service import OllamaService


def test_chat_loop_basic_flow(tmp_path, monkeypatch, capsys):
    # Prepare template and memory file
    template = {
        "model": "llama3.2",
        "system": "You are a test",
        "parameters": {"num_ctx": 1024},
    }
    mem_file = tmp_path / "memory.json"
    mem_file.write_text(
        '{"timestamp": "t", "messages": [{"role": "system", "content": "You are a test"}]}'
    )

    # Dummy memory store to be used by chat_loop
    class DummyStore:
        VALID_TYPES = {"preference", "fact", "task", "insight"}

        def remember(
            self,
            text,
            type,
            context=None,
            tags=None,
            importance=1.0,
            confidence=1.0,
            source=None,
        ):
            return 123

        def recall(
            self, query, type=None, context=None, tag=None, limit=5, use_semantic=True
        ):
            return [
                {
                    "id": 1,
                    "memory_text": "test memory",
                    "type": "fact",
                    "tag": "test",
                    "confidence": 1.0,
                    "source": None,
                    "created_at": "2024-01-01",
                    "last_accessed": "2024-01-01",
                    "access_count": 1,
                    "embedding_model": "text-embedding-3-small",
                    "similarity": 0.9,
                    "importance": 1.0,
                }
            ]

        def forget(self, mem_id):
            return True

        def list_tags(self):
            return ["work", "personal"]

        def list_recent(self, limit=20):
            return [
                {
                    "id": 1,
                    "memory_text": "test memory",
                    "type": "fact",
                    "tag": "test",
                }
            ]

        def list_by_tag(self, tag, limit=20):
            return [
                {
                    "id": 1,
                    "memory_text": "test memory",
                    "type": "fact",
                    "tag": tag,
                }
            ]

        def get_stats(self):
            return {
                "total_memories": 2,
                "memory_types": {"fact": 1, "preference": 1},
                "total_tags": 2,
            }

        def close(self):
            pass

    # Patch OllamaService
    monkeypatch.setattr(OllamaService, "check_connection", lambda self: True)

    class MockMessage:
        def __init__(self, content):
            self.content = content
            self.thinking = None
            self.tool_calls = []
            self.role = "assistant"

    class MockChunk:
        def __init__(self, message):
            self.message = message

    class MockResponse:
        def __init__(self, message):
            self.message = message

    def mock_chat(*args, **kwargs):
        if kwargs.get("stream"):
            yield MockChunk(MockMessage("ok response"))
        else:
            return MockResponse(MockMessage("ok response"))

    monkeypatch.setattr(OllamaService, "chat", mock_chat)

    # Patch save_chat_history to avoid writing to disk repeatedly
    monkeypatch.setattr(
        "src.services.memory.file_storage.save_chat_history",
        lambda messages, file_path: None,
    )

    # Sequence of inputs to drive through many branches
    inputs = iter(
        [
            "/?",
            "/context",
            "/context brief",
            "/trim",
            "/stream",
            "/remember type=fact tag=test remember this",
            "/recall test",
            "/memories",
            "/memories test",
            "/forget 1",
            "/tags",
            "/stats",
            "/clear",
            "/load",
            "/save",
            "/quit",
        ]
    )

    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    # Run chat loop; it should exit cleanly
    chat.chat_loop(template, context_file=str(mem_file), memory_store=DummyStore())

    # Capture output to check some expected substrings
    out = capsys.readouterr().out
    assert "Ollama Chat" in out
    assert "Memory stored with ID" in out


def test_chat_loop_trimming(tmp_path, monkeypatch, capsys):
    # Prepare template and memory file
    template = {
        "model": "llama3.2",
        "system": "You are a test",
        "parameters": {"num_ctx": 100},
    }
    mem_file = tmp_path / "memory.json"

    # Mock services
    monkeypatch.setattr(OllamaService, "check_connection", lambda self: True)

    class MockMessage:
        def __init__(self, content):
            self.content = content
            self.thinking = None
            self.tool_calls = []
            self.role = "assistant"

    class MockChunk:
        def __init__(self, message):
            self.message = message

    def mock_chat(*args, **kwargs):
        yield MockChunk(MockMessage("ok"))

    monkeypatch.setattr(OllamaService, "chat", mock_chat)
    monkeypatch.setattr(
        "src.services.memory.file_storage.save_chat_history",
        lambda messages, file_path: None,
    )

    # Mock token counting to trigger trim condition
    monkeypatch.setattr("src.agent.chat_session.count_message_tokens", lambda m: 200)

    # Mock trim_context to simulate successful trim
    monkeypatch.setattr("src.agent.chat_session.trim_context", lambda m, t: ([], True))

    # Mock inputs
    inputs = iter(["hello", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    chat.chat_loop(template, context_file=str(mem_file))

    out = capsys.readouterr().out
    assert "Auto-trimmed context" in out
