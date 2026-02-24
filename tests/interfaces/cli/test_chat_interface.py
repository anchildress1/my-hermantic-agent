from src.agent.chat_session import ChatSession
from src.services.llm.ollama_service import OllamaService
from src.core.config import AgentConfig
from unittest.mock import MagicMock, patch

from src.services.memory.auto_writer import AutoMemoryFailure, AutoMemoryResult


def test_cmd_help_with_memory_store(capsys):
    """Test help command includes memory guidance when store is available."""

    class DummyMemoryStore:
        pass

    session = ChatSession(
        config=AgentConfig(model="test", system="", parameters={}),
        context_file="test.json",
        llm_service=MagicMock(spec=OllamaService),
        memory_store=DummyMemoryStore(),
    )

    session.cmd_help()
    out = capsys.readouterr().out
    assert "Memory" in out
    assert "automatic" in out


def test_cmd_help_without_memory_store(capsys):
    """Test help command excludes memory section when store is unavailable."""
    session = ChatSession(
        config=AgentConfig(model="test", system="", parameters={}),
        context_file="test.json",
        llm_service=MagicMock(spec=OllamaService),
        memory_store=None,
    )

    session.cmd_help()
    out = capsys.readouterr().out
    assert "Memory (automatic)" not in out


def test_cmd_load_with_files(capsys):
    """Test loading specific files."""
    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
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
        config=AgentConfig(model="test", system="sys", parameters={}),
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
        config=AgentConfig(model="test", system="sys", parameters={}),
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
        config=AgentConfig(model="test", system="sys", parameters={}),
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
        config=AgentConfig(model="test", system="sys", parameters={}),
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


def test_cmd_audit_with_events(capsys):
    """Test audit command displays events."""
    mock_store = MagicMock()
    mock_store.list_events.return_value = [
        {
            "id": 1,
            "memory_id": 2,
            "operation": "remember",
            "status": "success",
            "details": {"memory_preview": "User prefers Python"},
            "created_at": "2026-02-19T00:00:00Z",
        }
    ]

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
        memory_store=mock_store,
    )

    session.cmd_audit(operation="remember")
    out = capsys.readouterr().out
    assert "Memory Events" in out
    assert "remember | success" in out


def test_cmd_audit_no_events(capsys):
    """Test audit command when no events are returned."""
    mock_store = MagicMock()
    mock_store.list_events.return_value = []

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
        memory_store=mock_store,
    )

    session.cmd_audit()
    out = capsys.readouterr().out
    assert "No memory events found" in out


def test_send_message_surfaces_full_auto_memory_failure_and_saves(capsys):
    """Test full failed memory text is shown and turn is persisted."""
    auto_writer = MagicMock()
    auto_writer.last_result = AutoMemoryResult(
        failures=[
            AutoMemoryFailure(
                memory_text="remember exact string that failed to write",
                type="fact",
                tag="chat",
                error="deadlock detected",
            )
        ]
    )

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
        auto_memory_writer=auto_writer,
    )

    session._handle_response = MagicMock(return_value=("assistant ok", False))

    with patch("src.agent.chat_session.save_chat_history") as mock_save:
        session._send_message("hello")

    out = capsys.readouterr().out
    assert "remember exact string that failed to write" in out
    assert "deadlock detected" in out
    mock_save.assert_called_once()


def test_send_message_sanitizes_auto_memory_failure_output(capsys):
    """Test failure output strips control/ANSI characters for safe terminal display."""
    auto_writer = MagicMock()
    auto_writer.last_result = AutoMemoryResult(
        failures=[
            AutoMemoryFailure(
                memory_text="remember\x1b[31m leaked\npayload\x1b[0m",
                type="fact",
                tag="chat",
                error="db\x1b[32m fail\rnow",
            )
        ]
    )

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
        auto_memory_writer=auto_writer,
    )

    session._handle_response = MagicMock(return_value=("assistant ok", False))

    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("hello")

    out = capsys.readouterr().out
    assert "\x1b[" not in out
    assert "remember leaked payload" in out
    assert "db fail now" in out


def test_cmd_audit_sanitizes_details(capsys):
    """Test audit output sanitizes detail payloads for terminal safety."""
    mock_store = MagicMock()
    mock_store.list_events.return_value = [
        {
            "id": 7,
            "memory_id": 9,
            "operation": "remember",
            "status": "error",
            "details": {"error": "bad\x1b[31m\npayload"},
            "created_at": "2026-02-22T00:00:00Z",
        }
    ]

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
        memory_store=mock_store,
    )

    session.cmd_audit(operation="remember")
    out = capsys.readouterr().out
    assert "\x1b[" not in out
    assert "bad payload" in out


def test_run_keyboard_interrupt_closes_memory_store(capsys):
    """Test run handles KeyboardInterrupt and closes memory store in finally."""
    mock_store = MagicMock()
    mock_llm = MagicMock(spec=OllamaService)
    mock_llm.check_connection.return_value = True

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=mock_llm,
        memory_store=mock_store,
    )

    with patch("builtins.input", side_effect=KeyboardInterrupt):
        with patch("src.agent.chat_session.save_chat_history"):
            session.run()

    out = capsys.readouterr().out
    assert "Saving before exit" in out
    mock_store.close.assert_called_once()


def test_handle_response_supports_dict_stream_chunks():
    """Test streamed responses handle dict-shaped Ollama payloads."""
    mock_llm = MagicMock(spec=OllamaService)
    mock_llm.chat.return_value = iter(
        [
            {"message": {"content": "hello ", "thinking": "thought ", "tool_calls": []}},
            {"message": {"content": "world", "thinking": "done"}},
        ]
    )

    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={}),
        context_file="default.json",
        llm_service=mock_llm,
    )

    response, memory_tool_called = session._handle_response()

    assert response == "hello world"
    assert memory_tool_called is False
    assert session.messages[-1]["content"] == "hello world"
    assert session.messages[-1]["thinking"] == "thought done"


def test_handle_xml_tool_calls_propagates_continuation_tool_flag():
    """Test continuation responses propagate memory-tool usage."""
    session = ChatSession(
        config=AgentConfig(model="test", system="sys", parameters={"use_xml_tools": True}),
        context_file="default.json",
        llm_service=MagicMock(spec=OllamaService),
    )
    session._handle_response = MagicMock(return_value=("continued", True))

    memory_tool_called = session._handle_xml_tool_calls([])

    assert memory_tool_called is True
    session._handle_response.assert_called_once()
