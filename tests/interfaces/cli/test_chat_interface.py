from src.agent.chat_session import ChatSession


def test_cmd_help_with_memory_store(capsys):
    """Test help command includes memory commands when store is available."""

    class DummyMemoryStore:
        pass

    session = ChatSession(
        config={"model": "test", "system": "", "parameters": {}},
        context_file="test.json",
        memory_store=DummyMemoryStore(),
    )

    session.cmd_help()
    out = capsys.readouterr().out
    assert "Memory Commands" in out


def test_cmd_help_without_memory_store(capsys):
    """Test help command excludes memory commands when store is unavailable."""
    session = ChatSession(
        config={"model": "test", "system": "", "parameters": {}},
        context_file="test.json",
        memory_store=None,
    )

    session.cmd_help()
    out = capsys.readouterr().out
    assert "Memory Commands" not in out
