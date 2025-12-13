import pytest
from unittest.mock import MagicMock, patch

from src.agent import chat


def test_check_ollama_connection_success(monkeypatch):
    monkeypatch.setattr("src.agent.chat.ollama", MagicMock())
    src = __import__("src.agent.chat", fromlist=["ollama"]).ollama
    src.list.return_value = {"models": [{"model": "llama3.2"}]}

    assert chat.check_ollama_connection("llama3.2") is True


def test_check_ollama_connection_failure(monkeypatch, capsys):
    def raise_exc():
        raise RuntimeError("no service")

    monkeypatch.setattr("src.agent.chat.ollama", MagicMock())
    src = __import__("src.agent.chat", fromlist=["ollama"]).ollama
    src.list.side_effect = raise_exc

    assert chat.check_ollama_connection("llama3.2") is False


def test_rate_limit_decorator():
    from src.agent.memory import rate_limit

    @rate_limit(max_calls=1, period=1.0)
    def dummy():
        return True

    assert dummy() is True
    with pytest.raises(Exception):
        dummy()


def test_get_embedding_errors(monkeypatch):
    # Ensure environment variables are set for instantiation
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("MEMORY_DB_URL", "postgresql://x")
    # Patch connection pool to avoid DB init
    with patch("src.agent.memory.pool.SimpleConnectionPool"):
        # Replace the exception classes in memory module with simple Exception subclasses
        import src.agent.memory as memmod

        class SimpleErr(Exception):
            pass

        monkeypatch.setattr(memmod, "RateLimitError", SimpleErr)
        monkeypatch.setattr(memmod, "APITimeoutError", SimpleErr)
        monkeypatch.setattr(memmod, "OpenAIError", SimpleErr)

        class Dummy:
            def __init__(self, *a, **k):
                self.embeddings = MagicMock()

        with patch("src.agent.memory.OpenAI", Dummy):
            from src.agent.memory import MemoryStore

            store = MemoryStore()

            # Rate limit
            store.openai_client.embeddings.create.side_effect = SimpleErr("rl")
            with pytest.raises(Exception):
                store._get_embedding("text")

            # Timeout
            store.openai_client.embeddings.create.side_effect = SimpleErr("to")
            with pytest.raises(Exception):
                store._get_embedding("text")

            # Generic OpenAIError
            store.openai_client.embeddings.create.side_effect = SimpleErr("oe")
            with pytest.raises(Exception):
                store._get_embedding("text")
