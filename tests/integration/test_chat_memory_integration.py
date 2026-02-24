"""Integration coverage for chat + memory write flows."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

from src.agent.chat_session import ChatSession
from src.core.config import AgentConfig
from src.services.memory.auto_writer import AutoMemoryWriter
from src.services.memory.langmem_extractor import LangMemExtractor


class _FakeMemoryStore:
    EVENT_SUCCESS = "success"
    EVENT_ERROR = "error"

    def __init__(self) -> None:
        self.remember_calls: list[dict] = []
        self.events: list[dict] = []
        self.closed = False

    def memory_exists(self, **_kwargs) -> bool:
        return False

    def revive_exact_memory(self, **_kwargs):  # pragma: no cover - not hit in these flows
        return None

    def remember(self, **kwargs):
        self.remember_calls.append(kwargs)
        return len(self.remember_calls)

    def record_event(self, **kwargs) -> None:
        self.events.append(kwargs)

    def get_last_error(self):
        return None

    def close(self) -> None:
        self.closed = True


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
    thinking: str | None = None
    tool_calls: list | None = None


@dataclass
class _Chunk:
    message: _ChunkMessage


class _FakeLLMService:
    def __init__(self, chunks) -> None:
        self._chunks = chunks

    def chat(self, messages, tools=None, stream=True):  # noqa: ARG002
        return iter(self._chunks)

    def check_connection(self) -> bool:
        return True


def _make_session(llm_service, memory_store=None, auto_memory_writer=None) -> ChatSession:
    return ChatSession(
        config=AgentConfig(
            model="test-model",
            system="sys",
            parameters={"use_xml_tools": False, "num_ctx": 256},
        ),
        context_file="ctx.json",
        llm_service=llm_service,
        memory_store=memory_store,
        auto_memory_writer=auto_memory_writer,
    )


def test_send_message_integration_stores_auto_memory_from_extractor():
    store = _FakeMemoryStore()
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = 2
    extractor.default_tag = "chat"
    extractor._manager = MagicMock()
    extractor._manager.invoke.return_value = [
        {"memory_text": "User likes espresso", "type": "preference", "tag": "chat"}
    ]
    auto_writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    llm = _FakeLLMService(
        chunks=[{"message": {"content": "Noted. I'll remember that."}}]
    )
    session = _make_session(llm_service=llm, auto_memory_writer=auto_writer)

    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("Remember that I like espresso")

    assert len(store.remember_calls) == 1
    assert store.remember_calls[0]["memory_text"] == "User likes espresso"
    assert any(event["status"] == store.EVENT_SUCCESS for event in store.events)
    assert session.messages[-1]["role"] == "assistant"


def test_send_message_integration_skips_auto_writer_after_native_tool_call():
    store = _FakeMemoryStore()
    auto_writer = MagicMock()
    tool_call = _ToolCall(
        function=_ToolFunction(
            name="store_memory_tool",
            arguments={
                "memory_text": "User prefers concise answers",
                "type": "preference",
                "tag": "chat",
                "importance": 2.0,
                "confidence": 0.9,
            },
        )
    )
    llm = _FakeLLMService(
        chunks=[
            _Chunk(
                _ChunkMessage(
                    content="Stored with tool.", tool_calls=[tool_call], thinking=None
                )
            )
        ]
    )
    session = _make_session(
        llm_service=llm,
        memory_store=store,
        auto_memory_writer=auto_writer,
    )

    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("Remember this preference")

    assert len(store.remember_calls) == 1
    auto_writer.process_turn.assert_not_called()
    assert any(msg.get("role") == "tool" for msg in session.messages)


def test_send_message_integration_handles_extractor_exception_without_crashing():
    store = _FakeMemoryStore()
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = 2
    extractor.default_tag = "chat"
    extractor._manager = MagicMock()
    extractor._manager.invoke.side_effect = RuntimeError("extractor unavailable")
    auto_writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    llm = _FakeLLMService(chunks=[{"message": {"content": "ok"}}])
    session = _make_session(llm_service=llm, auto_memory_writer=auto_writer)

    with patch("src.agent.chat_session.save_chat_history"):
        session._send_message("hello there")

    assert store.remember_calls == []
    assert auto_writer.last_result.all_ids == []
