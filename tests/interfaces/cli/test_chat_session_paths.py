"""Focused branch-path tests for chat session control flow."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent.chat_session import ChatSession
from src.core.config import AgentConfig
from src.services.llm.ollama_service import OllamaService


def _make_session(
    *,
    use_xml_tools: bool = False,
    memory_store=None,
    auto_memory_writer=None,
    llm_service=None,
    system_prompt: str = "sys",
) -> ChatSession:
    return ChatSession(
        config=AgentConfig(
            model="test-model",
            system=system_prompt,
            parameters={"use_xml_tools": use_xml_tools, "num_ctx": 128},
        ),
        context_file="ctx.json",
        llm_service=llm_service or MagicMock(spec=OllamaService),
        memory_store=memory_store,
        auto_memory_writer=auto_memory_writer,
    )


@dataclass
class _ToolFunction:
    name: str
    arguments: dict


@dataclass
class _ToolCall:
    function: _ToolFunction


@dataclass
class _ChunkMessage:
    content: str | None
    thinking: str | None
    tool_calls: list


@dataclass
class _Chunk:
    message: _ChunkMessage


def test_cmd_clear_without_archive_path_message(capsys):
    session = _make_session()
    with patch("src.agent.chat_session.archive_chat_history", return_value=None):
        with patch("src.agent.chat_session.save_chat_history"):
            session.cmd_clear()

    out = capsys.readouterr().out
    assert "Previous conversation archived" not in out
    assert "Context cleared and saved" in out


def test_sanitize_cli_text_truncates_and_normalizes():
    value = "  line1\r\nline2\x1b[31m!\x1b[0m " + ("x" * 40)
    sanitized = ChatSession._sanitize_cli_text(value, max_len=20)
    assert "\x1b[" not in sanitized
    assert "\n" not in sanitized
    assert sanitized.endswith("...")


def test_sanitize_details_payload_handles_collections_and_passthrough():
    payload = {
        "k\x1b[31m": ["a\nb", ("z\r", {"inner": "ok\x1b[0m"})],
        "n": 7,
    }
    sanitized = ChatSession._sanitize_details_payload(payload)
    assert sanitized["k"] == ["a b", ("z", {"inner": "ok"})]
    assert sanitized["n"] == 7


def test_cmd_trim_saves_when_trimmed():
    session = _make_session()
    with patch("src.agent.chat_session.trim_context", return_value=(session.messages, True)):
        with patch("src.agent.chat_session.save_chat_history") as mock_save:
            session.cmd_trim()
    mock_save.assert_called_once()


def test_cmd_trim_reports_when_not_trimmed(capsys):
    session = _make_session()
    with patch("src.agent.chat_session.trim_context", return_value=(session.messages, False)):
        with patch("src.agent.chat_session.count_message_tokens", return_value=42):
            session.cmd_trim()
    out = capsys.readouterr().out
    assert "Context is within limits" in out


def test_cmd_audit_without_memory_store(capsys):
    session = _make_session(memory_store=None)
    session.cmd_audit()
    assert "Memory store not available" in capsys.readouterr().out


def test_cmd_audit_handles_memory_store_exception(capsys):
    store = MagicMock()
    store.list_events.side_effect = RuntimeError("audit exploded")
    session = _make_session(memory_store=store)
    session.cmd_audit()
    out = capsys.readouterr().out
    assert "Error: audit exploded" in out


def test_cmd_audit_skips_details_output_for_empty_payload(capsys):
    store = MagicMock()
    store.list_events.return_value = [
        {
            "id": 10,
            "memory_id": None,
            "operation": "forget",
            "status": "success",
            "details": {},
            "created_at": "2026-02-24T00:00:00Z",
        }
    ]
    session = _make_session(memory_store=store)
    session.cmd_audit()
    out = capsys.readouterr().out
    assert "forget | success" in out
    assert "details:" not in out


def test_append_xml_tool_response_wraps_payload():
    session = _make_session(use_xml_tools=True)
    session._append_xml_tool_response("ok")
    assert session.messages[-1] == {
        "role": "user",
        "content": "<tool_response>ok</tool_response>",
    }


def test_execute_xml_tool_returns_false_for_unknown_tool():
    session = _make_session(use_xml_tools=True)
    assert session._execute_xml_tool({}, "missing_tool", {}) is False


def test_execute_xml_tool_handles_exceptions_and_records_error():
    session = _make_session(use_xml_tools=True)

    def boom_tool(**_kwargs):
        raise RuntimeError("boom")

    result = session._execute_xml_tool(
        tool_map={"boom_tool": boom_tool},
        tool_name="boom_tool",
        arguments={"x": 1},
    )

    assert result is False
    assert "Error: boom" in session.messages[-1]["content"]


def test_execute_xml_tool_marks_store_memory_tool_success():
    session = _make_session(use_xml_tools=True)

    def store_memory_tool(**_kwargs):
        return "Stored memory #9"

    result = session._execute_xml_tool(
        tool_map={"store_memory_tool": store_memory_tool},
        tool_name="store_memory_tool",
        arguments={"memory_text": "x"},
    )

    assert result is True
    assert "Stored memory #9" in session.messages[-1]["content"]


def test_run_command_parses_audit_operation():
    session = _make_session(memory_store=MagicMock())
    session.cmd_audit = MagicMock()

    handled = session._run_command("/audit forget")

    assert handled is True
    session.cmd_audit.assert_called_once_with(operation="forget")


def test_init_with_xml_tools_appends_markup_when_memory_enabled():
    with patch("src.agent.chat_session.format_tools_xml", return_value="<tools/>"):
        session = _make_session(
            use_xml_tools=True,
            memory_store=MagicMock(),
            system_prompt="",
        )
    content = session.messages[0]["content"]
    assert "<tools/>" in content
    assert "Memory policy:" in content


def test_handle_user_input_handles_exit_and_empty_cases():
    session = _make_session()
    session.cmd_quit = MagicMock(return_value=True)
    session._send_message = MagicMock()
    assert session._handle_user_input("quit") is True

    session.cmd_quit.reset_mock()
    assert session._handle_user_input("   ".strip()) is False
    session._send_message.assert_not_called()


def test_handle_tool_calls_executes_memory_tool_and_appends_tool_message():
    store = MagicMock()
    session = _make_session(memory_store=store)
    call = _ToolCall(
        function=_ToolFunction(
            name="store_memory_tool",
            arguments={"memory_text": "x", "type": "fact", "tag": "chat"},
        )
    )
    mock_tool = MagicMock(return_value="Stored memory #7")
    with patch("src.agent.chat_session.create_store_memory_tool", return_value=mock_tool):
        called = session._handle_tool_calls([call])

    assert called is True
    mock_tool.assert_called_once()
    assert session.messages[-1]["role"] == "tool"


def test_handle_tool_calls_ignores_non_memory_tool_calls():
    session = _make_session(memory_store=MagicMock())
    call = _ToolCall(function=_ToolFunction(name="different_tool", arguments={}))
    assert session._handle_tool_calls([call]) is False


def test_send_message_skips_auto_writer_for_blank_assistant_text():
    auto_writer = MagicMock()
    session = _make_session(auto_memory_writer=auto_writer)
    session._handle_response = MagicMock(return_value=("   ", False))
    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("hello")
    auto_writer.process_turn.assert_not_called()


def test_send_message_skips_extra_save_when_auto_trim_not_applied():
    session = _make_session()
    session._handle_response = MagicMock(return_value=("assistant", False))

    with patch("src.agent.chat_session.count_message_tokens", side_effect=[999, 20]):
        with patch(
            "src.agent.chat_session.trim_context",
            return_value=(session.messages, False),
        ):
            with patch("src.agent.chat_session.save_chat_history") as mock_save:
                session._send_message("hello")

    # Only the final per-turn persistence should run.
    mock_save.assert_called_once()


def test_send_message_skips_auto_writer_when_memory_tool_already_called():
    auto_writer = MagicMock()
    session = _make_session(auto_memory_writer=auto_writer)
    session._handle_response = MagicMock(return_value=("assistant", True))
    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("hello")
    auto_writer.process_turn.assert_not_called()


def test_send_message_logs_auto_writer_exceptions():
    auto_writer = MagicMock()
    auto_writer.process_turn.side_effect = RuntimeError("writer failed")
    auto_writer.last_result = SimpleNamespace(inserted_ids=[], revived_ids=[], failures=[])

    session = _make_session(auto_memory_writer=auto_writer)
    session._handle_response = MagicMock(return_value=("assistant", False))
    with patch("src.agent.chat_session.save_chat_history"):
        with patch("src.agent.chat_session.logger.error") as mock_error:
            session._send_message("hello")

    mock_error.assert_called()


def test_send_message_prints_refreshed_memory_ids(capsys):
    auto_writer = MagicMock()
    auto_writer.last_result = SimpleNamespace(inserted_ids=[], revived_ids=[12], failures=[])

    session = _make_session(auto_memory_writer=auto_writer)
    session._handle_response = MagicMock(return_value=("assistant", False))
    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("hello")

    out = capsys.readouterr().out
    assert "Auto-memory refreshed: 12" in out


def test_resolve_ollama_tools_respects_xml_mode():
    session = _make_session(use_xml_tools=False, memory_store=MagicMock())
    assert session._resolve_ollama_tools() == session.tools

    session.use_xml_tools = True
    assert session._resolve_ollama_tools() is None


def test_read_payload_value_handles_dict_and_object():
    assert ChatSession._read_payload_value({"x": 1}, "x", 0) == 1
    obj = SimpleNamespace(x=2)
    assert ChatSession._read_payload_value(obj, "x", 0) == 2
    assert ChatSession._read_payload_value(obj, "missing", 9) == 9


def test_stream_assistant_response_collects_tool_calls_from_object_chunks():
    mock_llm = MagicMock(spec=OllamaService)
    tool_call = _ToolCall(function=_ToolFunction(name="store_memory_tool", arguments={}))
    mock_llm.chat.return_value = iter(
        [
            _Chunk(_ChunkMessage(content="hello ", thinking="t1", tool_calls=[])),
            _Chunk(_ChunkMessage(content="world", thinking="t2", tool_calls=[tool_call])),
        ]
    )
    session = _make_session(llm_service=mock_llm)
    response, thinking, tool_calls = session._stream_assistant_response(ollama_tools=None)

    assert response == "hello world"
    assert thinking == "t1t2"
    assert tool_calls == [tool_call]


def test_stream_assistant_response_captures_tool_calls_without_content():
    mock_llm = MagicMock(spec=OllamaService)
    tool_call = _ToolCall(function=_ToolFunction(name="store_memory_tool", arguments={}))
    mock_llm.chat.return_value = iter(
        [
            _Chunk(_ChunkMessage(content="", thinking=None, tool_calls=[tool_call])),
        ]
    )
    session = _make_session(llm_service=mock_llm)
    response, thinking, tool_calls = session._stream_assistant_response(ollama_tools=None)

    assert response == ""
    assert thinking == ""
    assert tool_calls == [tool_call]


def test_build_assistant_message_omits_optional_fields_when_empty():
    payload = ChatSession._build_assistant_message(
        full_response="ok",
        thinking="",
        tool_calls=[],
    )
    assert payload == {"role": "assistant", "content": "ok"}


def test_process_response_tool_calls_combines_native_and_xml_results():
    session = _make_session(use_xml_tools=True)
    session._handle_tool_calls = MagicMock(return_value=False)
    session._handle_xml_tool_calls = MagicMock(return_value=True)
    with patch("src.agent.chat_session.parse_tool_calls", return_value=[{"name": "x"}]):
        called = session._process_response_tool_calls(
            tool_calls=[_ToolCall(function=_ToolFunction(name="x", arguments={}))],
            full_response="<tool_call>{}</tool_call>",
        )
    assert called is True
    session._handle_tool_calls.assert_called_once()
    session._handle_xml_tool_calls.assert_called_once()


def test_handle_xml_tool_calls_skips_invalid_shapes_and_executes_valid_calls():
    session = _make_session(use_xml_tools=True)

    def dummy_tool(**_kwargs):
        return "ok"

    session.tools = [dummy_tool]
    session._execute_xml_tool = MagicMock(return_value=True)
    session._handle_response = MagicMock(return_value=("continued", False))

    called = session._handle_xml_tool_calls(
        [
            {"name": 123, "arguments": {}},
            {"name": "dummy_tool", "arguments": []},
            {"name": "dummy_tool", "arguments": {"x": 1}},
        ]
    )

    assert called is True
    session._execute_xml_tool.assert_called_once()
    session._handle_response.assert_called_once()


def test_handle_xml_tool_calls_keeps_false_when_execute_returns_false():
    session = _make_session(use_xml_tools=True)

    def dummy_tool(**_kwargs):
        return "ok"

    session.tools = [dummy_tool]
    session._execute_xml_tool = MagicMock(return_value=False)
    session._handle_response = MagicMock(return_value=("continued", False))

    called = session._handle_xml_tool_calls(
        [
            {"name": "dummy_tool", "arguments": {"x": 1}},
        ]
    )

    assert called is False
    session._execute_xml_tool.assert_called_once()
    session._handle_response.assert_called_once()


def test_run_exits_early_when_llm_unavailable(capsys):
    llm = MagicMock(spec=OllamaService)
    llm.check_connection.return_value = False
    session = _make_session(llm_service=llm)
    session.run()
    out = capsys.readouterr().out
    assert "LLM service unavailable" in out


def test_run_handles_generic_loop_exception_and_continues(capsys):
    llm = MagicMock(spec=OllamaService)
    llm.check_connection.return_value = True
    session = _make_session(llm_service=llm)

    with patch("builtins.input", side_effect=["first", "second"]):
        with patch.object(
            session,
            "_handle_user_input",
            side_effect=[RuntimeError("loop bug"), True],
        ):
            session.run()

    out = capsys.readouterr().out
    assert "❌ Error: loop bug" in out
