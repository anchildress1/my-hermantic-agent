import pytest
from unittest.mock import MagicMock, patch


def test_rate_limit_decorator():
    from src.services.memory.vector_store import rate_limit

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
    with patch("src.services.memory.vector_store.pool.SimpleConnectionPool"):
        # Replace the exception classes in memory module with simple Exception subclasses
        import src.services.memory.vector_store as memmod

        class SimpleErr(Exception):
            pass

        monkeypatch.setattr(memmod, "RateLimitError", SimpleErr)
        monkeypatch.setattr(memmod, "APITimeoutError", SimpleErr)
        monkeypatch.setattr(memmod, "OpenAIError", SimpleErr)

        class Dummy:
            def __init__(self, *a, **k):
                self.embeddings = MagicMock()

        with patch("src.services.memory.vector_store.OpenAI", Dummy):
            from src.services.memory.vector_store import MemoryStore

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
