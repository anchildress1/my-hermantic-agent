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
            tag=None,
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

        def stats(self):
            return {
                "total_memories": 2,
                "unique_types": 2,
                "unique_tags": 2,
                "avg_confidence": 0.9,
                "avg_importance": 1.0,
                "last_memory_at": "now",
            }

        def close(self):
            pass

    # Patch OllamaService
    monkeypatch.setattr(OllamaService, "check_connection", lambda self: True)

    # Mock chat response (non-streaming for simplicity in this test, though loop supports stream=True)
    # The chat_loop expects stream=True by default, but we can return a list for iterator or dict for non-stream
    # The updated chat_loop handles streaming by iterating. If we return a list of chunks, it works.

    def mock_chat(*args, **kwargs):
        if kwargs.get("stream"):
            yield {"message": {"content": "ok response"}}
        else:
            return {"message": {"content": "ok response"}}

    monkeypatch.setattr(OllamaService, "chat", mock_chat)

    # Patch save_chat_history to avoid writing to disk repeatedly
    monkeypatch.setattr(chat, "save_chat_history", lambda messages, file_path: None)

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

    def mock_chat(*args, **kwargs):
        yield {"message": {"content": "ok"}}

    monkeypatch.setattr(OllamaService, "chat", mock_chat)
    monkeypatch.setattr(chat, "save_chat_history", lambda messages, file_path: None)

    # Mock token counting to trigger trim condition
    monkeypatch.setattr("src.interfaces.cli.chat.count_message_tokens", lambda m: 200)

    # Mock trim_context to simulate successful trim
    monkeypatch.setattr("src.interfaces.cli.chat.trim_context", lambda m, t: ([], True))

    # Mock inputs
    inputs = iter(["hello", "/quit"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    chat.chat_loop(template, context_file=str(mem_file))

    out = capsys.readouterr().out
    assert "Auto-trimmed context" in out
