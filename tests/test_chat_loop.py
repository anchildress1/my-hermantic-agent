import builtins
from unittest.mock import MagicMock

from src.agent import chat


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

    # Patch ollama and connectivity
    monkeypatch.setattr("src.agent.chat.check_ollama_connection", lambda m: True)

    # Ollama chat should return a simple non-stream response
    fake_resp = {"message": {"content": "ok response"}}
    monkeypatch.setattr("src.agent.chat.ollama", MagicMock())
    oll = __import__("src.agent.chat", fromlist=["ollama"]).ollama
    oll.chat.return_value = fake_resp

    # Patch MemoryStore import used inside chat_loop
    import src.agent.memory as memmod

    monkeypatch.setattr(memmod, "MemoryStore", DummyStore)

    # Patch save_memory to avoid writing to disk repeatedly
    monkeypatch.setattr(chat, "save_memory", lambda messages, memory_file: None)

    # Sequence of inputs to drive through many branches
    inputs = iter(
        [
            "/?",
            "/remember type=fact tag=test remember this",
            "/recall test",
            "/forget 1",
            "/memories",
            "/clear",
            "/load",
            "/save",
            "/bye",
        ]
    )

    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))

    # Run chat loop; it should exit cleanly
    chat.chat_loop(template, memory_file=str(mem_file))

    # Capture output to check some expected substrings
    out = capsys.readouterr().out
    assert "Ollama Chat" in out
    assert "Memory stored with ID" in out
