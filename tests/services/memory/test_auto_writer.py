import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

from src.services.memory.auto_writer import AutoMemoryWriter
from src.services.memory.langmem_extractor import LangMemExtractor, MemoryCandidate
from src.services.memory.vector_store import MemoryStore


def test_langmem_extractor_extract_dedup_and_limit():
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = 2
    extractor.default_tag = "chat"

    manager = MagicMock()
    manager.invoke.return_value = [
        {"memory_text": "User prefers Python", "type": "preference", "tag": "coding"},
        {"memory_text": "User prefers Python", "type": "preference", "tag": "coding"},
        {"memory_text": "User has meeting Friday", "type": "task", "tag": "work"},
    ]
    extractor._manager = manager

    results = extractor.extract(
        [{"role": "user", "content": "I prefer Python and I have a meeting Friday."}]
    )

    assert len(results) == 2
    assert results[0].memory_text == "User prefers Python"
    assert results[1].type == "task"


def test_auto_writer_revives_duplicates_and_stores_new():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.side_effect = [True, False]
    store.revive_exact_memory.return_value = {
        "id": 5,
        "importance": 1.5,
        "access_count": 4,
    }
    store.remember.return_value = 7

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = [
        MemoryCandidate(
            memory_text="User prefers Python", type="preference", tag="coding"
        ),
        MemoryCandidate(memory_text="User has standup", type="task", tag="work"),
    ]

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="I prefer Python and have standup tomorrow",
        assistant_message="Noted",
    )

    assert ids == [7, 5]
    assert store.remember.call_count == 1
    store.revive_exact_memory.assert_called_once()
    assert store.record_event.call_count == 2
    assert writer.last_result.inserted_ids == [7]
    assert writer.last_result.revived_ids == [5]


def test_auto_writer_no_candidates_returns_empty():
    store = MagicMock(spec=MemoryStore)
    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = []

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(user_message="hello", assistant_message="hi")

    assert ids == []
    store.remember.assert_not_called()


def test_auto_writer_explicit_remember_boosts_importance():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.return_value = False
    store.remember.return_value = 99

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = [
        MemoryCandidate(
            memory_text="User prefers Python for backend",
            type="preference",
            tag="coding",
            importance=1.0,
            confidence=0.6,
        )
    ]

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="Remember that I prefer Python for backend work",
        assistant_message="Noted.",
    )

    assert ids == [99]
    kwargs = store.remember.call_args.kwargs
    assert kwargs["importance"] >= 2.0
    assert kwargs["confidence"] >= 0.9


def test_auto_writer_non_explicit_remember_phrase_not_boosted():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.return_value = False
    store.remember.return_value = 11

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = [
        MemoryCandidate(
            memory_text="User said they do not remember old code details",
            type="fact",
            tag="chat",
            importance=1.0,
            confidence=0.6,
        )
    ]

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="I don't remember how this old function worked.",
        assistant_message="Let's inspect it together.",
    )

    assert ids == [11]
    kwargs = store.remember.call_args.kwargs
    assert kwargs["importance"] == pytest.approx(1.0)
    assert kwargs["confidence"] == pytest.approx(0.6)


def test_auto_writer_explicit_remember_fallback_when_extractor_empty():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.return_value = False
    store.remember.return_value = 10

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = []

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="Remember that I have a dentist appointment Friday at 2pm",
        assistant_message="Got it.",
    )

    assert ids == [10]
    kwargs = store.remember.call_args.kwargs
    assert "dentist appointment" in kwargs["memory_text"]
    assert kwargs["importance"] >= 2.0


def test_auto_writer_records_error_event_when_store_fails():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.return_value = False
    store.remember.return_value = None
    store.get_last_error.return_value = {"error": "database timeout"}

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = [
        MemoryCandidate(
            memory_text="User wants calendar reminders", type="preference", tag="ops"
        )
    ]

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="Remember that I want calendar reminders.",
        assistant_message="Noted.",
    )

    assert ids == []
    assert store.record_event.call_count >= 1
    statuses = [call.kwargs["status"] for call in store.record_event.call_args_list]
    assert MemoryStore.EVENT_ERROR in statuses
    assert writer.last_result.failures[0].error == "database timeout"


def test_auto_writer_records_failure_when_duplicate_revive_fails():
    store = MagicMock(spec=MemoryStore)
    store.memory_exists.return_value = True
    store.revive_exact_memory.return_value = None
    store.get_last_error.return_value = {"error": "write conflict"}

    extractor = MagicMock(spec=LangMemExtractor)
    extractor.extract.return_value = [
        MemoryCandidate(memory_text="User prefers vim", type="preference", tag="coding")
    ]

    writer = AutoMemoryWriter(memory_store=store, extractor=extractor)
    ids = writer.process_turn(
        user_message="Remember that I prefer vim",
        assistant_message="Noted.",
    )

    assert ids == []
    assert writer.last_result.failures[0].error == "write conflict"


def test_langmem_extractor_init_and_extract_with_fake_sdk(monkeypatch):
    fake_llm = object()
    manager = MagicMock()
    manager.invoke.return_value = {
        "memories": [{"text": "User likes CLI", "type": "preference"}]
    }

    captured = {}

    def fake_init_chat_model(*, model, model_provider, temperature):
        captured["init"] = {
            "model": model,
            "model_provider": model_provider,
            "temperature": temperature,
        }
        return fake_llm

    def fake_create_memory_manager(llm, **kwargs):
        captured["llm"] = llm
        captured["kwargs"] = kwargs
        return manager

    langchain_mod = ModuleType("langchain")
    chat_models_mod = ModuleType("langchain.chat_models")
    chat_models_mod.init_chat_model = fake_init_chat_model
    langchain_mod.chat_models = chat_models_mod

    langmem_mod = ModuleType("langmem")
    langmem_mod.create_memory_manager = fake_create_memory_manager

    monkeypatch.setitem(sys.modules, "langchain", langchain_mod)
    monkeypatch.setitem(sys.modules, "langchain.chat_models", chat_models_mod)
    monkeypatch.setitem(sys.modules, "langmem", langmem_mod)

    extractor = LangMemExtractor(
        model="gpt-4.1-mini",
        model_provider="openai",
        temperature=0.0,
        max_memories_per_turn=2,
        default_tag="chat",
    )
    results = extractor.extract([{"role": "user", "content": "I like CLI workflows."}])

    assert captured["llm"] is fake_llm
    assert captured["init"]["model"] == "gpt-4.1-mini"
    assert captured["kwargs"]["enable_updates"] is False
    assert len(results) == 1
    assert results[0].memory_text == "User likes CLI"
    assert results[0].tag == "chat"


def test_langmem_extractor_handles_object_content_and_invalid_items():
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = 5
    extractor.default_tag = "chat"

    class Item:
        def __init__(self, content):
            self.content = content

    manager = MagicMock()
    manager.invoke.return_value = [
        Item({"content": "User writes tests first", "type": "fact", "tag": "dev"}),
        {"garbage": "no memory text"},
        "User prefers deterministic behavior",
    ]
    extractor._manager = manager

    results = extractor.extract(
        [
            {"role": "user", "content": "I write tests first."},
            {"role": "assistant", "content": "Good call."},
        ]
    )

    assert len(results) == 2
    assert results[0].type == "fact"
    assert results[1].memory_text == "User prefers deterministic behavior"
