"""Edge-case coverage for LangMem extractor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
import builtins

import pytest

from src.services.memory.langmem_extractor import LangMemExtractor, MemoryCandidate


def _make_extractor() -> LangMemExtractor:
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = 5
    extractor.default_tag = "chat"
    extractor._manager = MagicMock()
    return extractor


def test_init_raises_runtime_error_when_langmem_imports_missing(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"langchain.chat_models", "langmem"}:
            raise ImportError("missing test dependency")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="LangMem dependencies missing"):
        LangMemExtractor(model="x", model_provider="openai")


def test_normalize_messages_skips_blank_role_or_content():
    extractor = _make_extractor()
    normalized = extractor._normalize_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "", "content": "missing role"},
            {"role": "assistant", "content": "   "},
            {"role": "system", "content": " go "},
        ]
    )
    assert normalized == [
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "go"},
    ]


def test_coerce_candidate_handles_none_invalid_and_content_aliases():
    extractor = _make_extractor()

    assert extractor._coerce_candidate(None) is None
    assert extractor._coerce_candidate(12345) is None
    assert extractor._coerce_candidate({"content": "User likes tea"}).memory_text == (
        "User likes tea"
    )
    assert extractor._coerce_candidate({"memory_text": ""}) is None

    wrapped = SimpleNamespace(content={"text": "Keep answers concise", "type": "fact"})
    candidate = extractor._coerce_candidate(wrapped)
    assert candidate is not None
    assert candidate.memory_text == "Keep answers concise"


def test_coerce_candidate_returns_memory_candidate_unchanged():
    extractor = _make_extractor()
    original = MemoryCandidate(memory_text="User likes coffee", type="fact", tag="chat")
    assert extractor._coerce_candidate(original) is original


@pytest.mark.parametrize(
    ("raw_result", "expected"),
    [
        (None, []),
        ([{"memory_text": "a"}], [{"memory_text": "a"}]),
        ({"memories": [{"memory_text": "a"}]}, [{"memory_text": "a"}]),
        ({"results": {"memory_text": "a"}}, [{"memory_text": "a"}]),
        ({"memory_text": "a"}, [{"memory_text": "a"}]),
        ("single", ["single"]),
    ],
)
def test_coerce_raw_items_variants(raw_result, expected):
    assert LangMemExtractor._coerce_raw_items(raw_result) == expected


def test_dedupe_candidates_applies_default_tag_and_limit():
    extractor = _make_extractor()
    extractor.max_memories_per_turn = 2
    candidates = extractor._dedupe_candidates(
        [
            {"memory_text": "User likes tea", "type": "preference", "tag": "  "},
            {"memory_text": "user likes tea", "type": "preference", "tag": "chat"},
            {"memory_text": "User works nights", "type": "fact", "tag": "work"},
        ]
    )

    assert len(candidates) == 2
    assert candidates[0].tag == "chat"
    assert candidates[1].memory_text == "User works nights"


def test_extract_returns_empty_for_invalid_messages():
    extractor = _make_extractor()
    assert extractor.extract([{"role": "", "content": ""}]) == []


def test_extract_returns_empty_when_manager_raises():
    extractor = _make_extractor()
    extractor._manager.invoke.side_effect = RuntimeError("manager down")

    result = extractor.extract([{"role": "user", "content": "remember this"}])

    assert result == []
