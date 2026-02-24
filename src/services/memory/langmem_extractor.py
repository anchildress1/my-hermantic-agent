"""LangMem-based relevance-first memory extraction."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class MemoryCandidate(BaseModel):
    """Extracted memory candidate."""

    memory_text: str = Field(min_length=1, max_length=8000)
    type: Literal["preference", "fact", "task", "insight"] = "fact"
    tag: str = Field(default="chat", min_length=1, max_length=100)
    importance: float = Field(default=1.0, ge=0.0, le=3.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class LangMemExtractor:
    """Extracts structured memories from chat turns using LangMem SDK."""

    def __init__(
        self,
        model: str,
        model_provider: str,
        temperature: float = 0.2,
        max_memories_per_turn: int = 2,
        default_tag: str = "chat",
    ) -> None:
        """Initialize extractor.

        Args:
            model: LLM model name for LangChain.
            model_provider: LangChain provider string (e.g., openai, ollama).
            temperature: Generation temperature.
            max_memories_per_turn: Max extracted memories accepted per turn.
            default_tag: Default tag/context when extraction omits one.
        """
        self.max_memories_per_turn = max_memories_per_turn
        self.default_tag = default_tag

        try:
            from langchain.chat_models import init_chat_model
            from langmem import create_memory_manager
        except ImportError as e:
            raise RuntimeError(
                "LangMem dependencies missing. Install langchain/langmem and provider integrations."
            ) from e

        llm = init_chat_model(
            model=model,
            model_provider=model_provider,
            temperature=temperature,
        )

        instructions = (
            "Extract durable, user-specific memories only. "
            "Prefer memories with long-term relevance to stable preferences, goals, and constraints. "
            "Skip transient chatter, jokes, or one-off requests unless they affect future behavior. "
            "Each memory must be atomic and independently useful. "
            "Prefer precision over volume."
        )

        self._manager = create_memory_manager(
            llm,
            schemas=[MemoryCandidate],
            instructions=instructions,
            enable_inserts=True,
            enable_updates=False,
            enable_deletes=False,
        )

    @staticmethod
    def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Normalize chat messages for memory extraction."""
        normalized: List[Dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if not role or not content:
                continue
            normalized.append({"role": role, "content": content})
        return normalized

    @staticmethod
    def _coerce_candidate(raw: Any) -> Optional[MemoryCandidate]:
        """Coerce a LangMem output item into MemoryCandidate."""
        if raw is None:
            return None

        if hasattr(raw, "content"):
            raw = getattr(raw, "content")

        if isinstance(raw, MemoryCandidate):
            return raw

        if isinstance(raw, str):
            return MemoryCandidate(memory_text=raw)

        if isinstance(raw, dict):
            payload = dict(raw)
            if "memory_text" not in payload:
                if "text" in payload:
                    payload["memory_text"] = payload["text"]
                elif "content" in payload and isinstance(payload["content"], str):
                    payload["memory_text"] = payload["content"]
            try:
                return MemoryCandidate.model_validate(payload)
            except Exception:
                return None

        return None

    @staticmethod
    def _coerce_raw_items(raw_result: Any) -> List[Any]:
        """Normalize manager output into a flat list of candidate payloads."""
        if raw_result is None:
            return []

        if isinstance(raw_result, list):
            return raw_result

        if isinstance(raw_result, dict):
            extracted = raw_result.get("memories") or raw_result.get("results")
            if extracted:
                if isinstance(extracted, list):
                    return extracted
                return [extracted]
            return [raw_result]

        return [raw_result]

    def _dedupe_candidates(self, raw_items: List[Any]) -> List[MemoryCandidate]:
        """Coerce, normalize, and deduplicate extracted memory candidates."""
        dedupe: set[tuple[str, str, str]] = set()
        extracted: List[MemoryCandidate] = []
        for item in raw_items:
            candidate = self._coerce_candidate(item)
            if candidate is None:
                continue

            if not candidate.tag.strip():
                candidate.tag = self.default_tag

            key = (
                candidate.memory_text.strip().lower(),
                candidate.type,
                candidate.tag.strip().lower(),
            )
            if key in dedupe:
                continue

            dedupe.add(key)
            extracted.append(candidate)
            if len(extracted) >= self.max_memories_per_turn:
                break

        return extracted

    def extract(self, messages: List[Dict[str, Any]]) -> List[MemoryCandidate]:
        """Extract memory candidates from message list.

        Args:
            messages: Chat messages with role/content.

        Returns:
            Deduplicated memory candidates limited by max_memories_per_turn.
        """
        normalized = self._normalize_messages(messages)
        if not normalized:
            return []

        try:
            raw_result = self._manager.invoke({"messages": normalized, "existing": []})
        except Exception as e:
            logger.error(f"LangMem extraction failed: {e}")
            return []

        raw_items = self._coerce_raw_items(raw_result)
        return self._dedupe_candidates(raw_items)
