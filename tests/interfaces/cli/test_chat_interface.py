from src.agent.chat_session import ChatSession
from src.services.llm.ollama_service import OllamaService
from unittest.mock import MagicMock, patch


def test_cmd_help_with_memory_store(capsys):
    """Test help command includes memory commands when store is available."""

    class DummyMemoryStore:
        pass

    session = ChatSession(
        config={"model": "test", "system": "", "parameters": {}},
        context_file="test.json",
        llm_service=MagicMock(spec=OllamaService),
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
        llm_service=MagicMock(spec=OllamaService),
        memory_store=None,
    )

    session.cmd_help()
    out = capsys.readouterr().out
    assert "Memory Commands" not in out


def test_cmd_load_with_files(capsys):
    """Test loading specific files."""
    session = ChatSession(
        config={"model": "test", "system": "sys", "parameters": {}},
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )

    with patch("src.agent.chat_session.load_chat_history") as mock_load:
        # Simulate loading two files
        mock_load.side_effect = [
            [{"role": "user", "content": "1"}],
            [{"role": "assistant", "content": "2"}],
        ]

        session.cmd_load(["f1.json", "f2.json"])

        assert len(session.messages) == 2
        assert session.messages[0]["content"] == "1"
        assert session.messages[1]["content"] == "2"
        out = capsys.readouterr().out
        assert "Context loaded from: f1.json f2.json" in out


def test_cmd_load_no_files_found(capsys):
    """Test loading files where none exist."""
    session = ChatSession(
        config={"model": "test", "system": "sys", "parameters": {}},
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )

    with patch("src.agent.chat_session.load_chat_history") as mock_load:
        mock_load.return_value = []

        session.cmd_load(["bad.json"])

        out = capsys.readouterr().out
        assert "No saved context loaded from: bad.json" in out


def test_cmd_load_default_success(capsys):
    """Test loading default context successfully."""
    session = ChatSession(
        config={"model": "test", "system": "sys", "parameters": {}},
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )

    with patch("src.agent.chat_session.load_chat_history") as mock_load:
        mock_load.return_value = [{"role": "user", "content": "old"}]

        session.cmd_load()

        assert session.messages[0]["role"] == "system"
        assert session.messages[0]["content"] == "sys"
        out = capsys.readouterr().out
        assert "Context loaded from default.json" in out


def test_cmd_load_default_failure(capsys):
    """Test loading default context failure."""
    session = ChatSession(
        config={"model": "test", "system": "sys", "parameters": {}},
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )

    with patch("src.agent.chat_session.load_chat_history") as mock_load:
        mock_load.return_value = []

        session.cmd_load()

        assert len(session.messages) == 1
        assert session.messages[0]["role"] == "system"
        out = capsys.readouterr().out
        assert "No saved context loaded from default.json" in out


def test_cmd_clear(capsys):
    """Test clear command with archive."""
    session = ChatSession(
        config={"model": "test", "system": "sys", "parameters": {}},
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )
    session.messages.append({"role": "user", "content": "hi"})

    with patch("src.agent.chat_session.archive_chat_history") as mock_archive:
        with patch("src.agent.chat_session.save_chat_history") as _:
            mock_archive.return_value = "archive.json"

            session.cmd_clear()

            assert len(session.messages) == 1
            assert session.messages[0]["content"] == "sys"
            out = capsys.readouterr().out
            assert "Previous conversation archived to archive.json" in out
