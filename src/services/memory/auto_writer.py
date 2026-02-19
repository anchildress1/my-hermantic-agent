"""Automatic memory writing with auditability safeguards."""

from __future__ import annotations

import logging
import re
from typing import List

from src.services.memory.langmem_extractor import LangMemExtractor, MemoryCandidate
from src.services.memory.vector_store import MemoryStore

logger = logging.getLogger(__name__)


class AutoMemoryWriter:
    """Writes extracted memories to persistent storage."""

    EXPLICIT_REMEMBER_PATTERNS = (
        re.compile(r"^\s*(please\s+)?remember\b", re.IGNORECASE),
        re.compile(r"\b(?:can|could|will|would)\s+you\s+remember\b", re.IGNORECASE),
        re.compile(r"\bplease\s+remember\b", re.IGNORECASE),
    )
    EXPLICIT_REMEMBER_IMPORTANCE = 2.0
    EXPLICIT_REMEMBER_CONFIDENCE = 0.9

    def __init__(
        self,
        memory_store: MemoryStore,
        extractor: LangMemExtractor,
        source_char_limit: int = 2000,
    ) -> None:
        """Initialize auto writer.

        Args:
            memory_store: Persistent memory store.
            extractor: LangMem-backed extraction service.
            source_char_limit: Max source payload saved per memory.
        """
        self.memory_store = memory_store
        self.extractor = extractor
        self.source_char_limit = source_char_limit

    @classmethod
    def _is_explicit_remember_intent(cls, user_message: str) -> bool:
        """Return True when the user explicitly asks to remember."""
        return any(
            pattern.search(user_message) for pattern in cls.EXPLICIT_REMEMBER_PATTERNS
        )

    @staticmethod
    def _fallback_memory_text(user_message: str) -> str:
        """Extract a fallback memory sentence from explicit remember text."""
        stripped = re.sub(
            r"^\s*(please\s+)?remember(?:\s+that|\s+to)?\s*[:,-]?\s*",
            "",
            user_message,
            flags=re.IGNORECASE,
        ).strip()
        return stripped or user_message.strip()

    def process_turn(self, user_message: str, assistant_message: str) -> List[int]:
        """Extract and store memories for a chat turn.

        Args:
            user_message: User message text.
            assistant_message: Assistant response text.

        Returns:
            Stored memory ids.
        """
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        explicit_remember = self._is_explicit_remember_intent(user_message)
        candidates = self.extractor.extract(messages)

        if explicit_remember and not candidates:
            candidates = [
                MemoryCandidate(
                    memory_text=self._fallback_memory_text(user_message),
                    type="fact",
                    tag="chat",
                    importance=self.EXPLICIT_REMEMBER_IMPORTANCE,
                    confidence=self.EXPLICIT_REMEMBER_CONFIDENCE,
                )
            ]

        if not candidates:
            return []

        source = (f"user: {user_message}\nassistant: {assistant_message}")[
            : self.source_char_limit
        ]

        stored_ids: List[int] = []
        for candidate in candidates:
            if explicit_remember:
                candidate.importance = max(
                    candidate.importance, self.EXPLICIT_REMEMBER_IMPORTANCE
                )
                candidate.confidence = max(
                    candidate.confidence, self.EXPLICIT_REMEMBER_CONFIDENCE
                )

            if self.memory_store.memory_exists(
                memory_text=candidate.memory_text,
                type=candidate.type,
                context=candidate.tag,
            ):
                logger.info(
                    "Skipping duplicate auto-memory: type=%s tag=%s text=%s",
                    candidate.type,
                    candidate.tag,
                    candidate.memory_text[:120],
                )
                self.memory_store.record_event(
                    operation="auto_remember",
                    status=MemoryStore.EVENT_SUCCESS,
                    details={
                        "memory_text": candidate.memory_text[:200],
                        "type": candidate.type,
                        "tag": candidate.tag,
                        "action": "skipped_duplicate",
                        "explicit_remember": explicit_remember,
                    },
                )
                continue

            memory_id = self.memory_store.remember(
                memory_text=candidate.memory_text,
                type=candidate.type,
                context=candidate.tag,
                importance=candidate.importance,
                confidence=candidate.confidence,
                source=source,
            )
            if memory_id:
                self.memory_store.record_event(
                    operation="auto_remember",
                    status=MemoryStore.EVENT_SUCCESS,
                    memory_id=memory_id,
                    details={
                        "memory_id": memory_id,
                        "type": candidate.type,
                        "tag": candidate.tag,
                        "importance": candidate.importance,
                        "confidence": candidate.confidence,
                        "explicit_remember": explicit_remember,
                    },
                )
                stored_ids.append(memory_id)
            else:
                self.memory_store.record_event(
                    operation="auto_remember",
                    status=MemoryStore.EVENT_ERROR,
                    details={
                        "memory_text": candidate.memory_text[:200],
                        "type": candidate.type,
                        "tag": candidate.tag,
                        "action": "store_failed",
                        "explicit_remember": explicit_remember,
                    },
                )

        return stored_ids
