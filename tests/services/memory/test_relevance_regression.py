"""Fixture-driven relevance regression tests for LangMem extraction behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.services.memory.langmem_extractor import LangMemExtractor, MemoryCandidate


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "relevance_regression.json"


def _load_relevance_cases() -> list[dict[str, Any]]:
    """Load relevance regression fixture cases from disk."""
    with FIXTURE_PATH.open("r", encoding="utf-8") as fixture_file:
        payload: dict[str, Any] = json.load(fixture_file)

    cases = payload.get("cases", [])
    if not isinstance(cases, list):
        raise ValueError("Fixture payload must contain a list under 'cases'")
    return cases


def _count_type(results: list[MemoryCandidate], memory_type: str) -> int:
    """Count extracted memories for a given type."""
    return sum(1 for result in results if result.type == memory_type)


def _within_band(value: float, band: dict[str, float]) -> bool:
    """Return True when value is inside an inclusive min/max tolerance band."""
    return band["min"] <= value <= band["max"]


@pytest.mark.parametrize(
    "case",
    _load_relevance_cases(),
    ids=lambda case: str(case["name"]),
)
def test_relevance_regression_tolerance_bands(case: dict[str, Any]) -> None:
    """Validate extraction behavior against fixture-defined tolerance bands."""
    extractor = LangMemExtractor.__new__(LangMemExtractor)
    extractor.max_memories_per_turn = int(case.get("max_memories_per_turn", 2))
    extractor.default_tag = str(case.get("default_tag", "chat"))

    manager = MagicMock()
    manager.invoke.return_value = case["raw_result"]
    extractor._manager = manager

    results = extractor.extract(messages=case["messages"])
    expectations = case["expectations"]

    count_band = expectations["count_band"]
    assert _within_band(float(len(results)), count_band)

    for memory_type, band in expectations.get("type_bands", {}).items():
        type_count = _count_type(results, memory_type)
        assert _within_band(float(type_count), band)

    lower_texts = [result.memory_text.lower() for result in results]
    for required_text in expectations.get("required_texts", []):
        assert any(required_text.lower() in text for text in lower_texts)

    for forbidden_text in expectations.get("forbidden_texts", []):
        assert not any(forbidden_text.lower() in text for text in lower_texts)

    required_tags = {tag.lower() for tag in expectations.get("required_tags", [])}
    if required_tags:
        observed_tags = {result.tag.lower() for result in results}
        assert required_tags.issubset(observed_tags)

    confidence_band = expectations.get("confidence_band")
    if confidence_band:
        assert all(
            _within_band(result.confidence, confidence_band) for result in results
        )

    importance_band = expectations.get("importance_band")
    if importance_band:
        assert all(
            _within_band(result.importance, importance_band) for result in results
        )
