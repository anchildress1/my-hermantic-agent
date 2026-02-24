"""Automatic memory writing with auditability safeguards."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import List

from src.services.memory.langmem_extractor import LangMemExtractor, MemoryCandidate
from src.services.memory.vector_store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AutoMemoryFailure:
    """Structured failure payload for auto-memory operations."""

    memory_text: str
    type: str
    tag: str
    error: str


@dataclass(slots=True)
class AutoMemoryResult:
    """Structured outcome payload for one chat turn."""

    inserted_ids: List[int] = field(default_factory=list)
    revived_ids: List[int] = field(default_factory=list)
    failures: List[AutoMemoryFailure] = field(default_factory=list)

    @property
    def all_ids(self) -> List[int]:
        """Return all touched memory ids."""
        return [*self.inserted_ids, *self.revived_ids]


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
        self.last_result = AutoMemoryResult()

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

    def _extract_candidates(
        self, messages: List[dict[str, str]], user_message: str, explicit_remember: bool
    ) -> List[MemoryCandidate]:
        """Extract memory candidates and apply explicit-intent fallback."""
        candidates = self.extractor.extract(messages)
        if explicit_remember and not candidates:
            return [
                MemoryCandidate(
                    memory_text=self._fallback_memory_text(user_message),
                    type="fact",
                    tag="chat",
                    importance=self.EXPLICIT_REMEMBER_IMPORTANCE,
                    confidence=self.EXPLICIT_REMEMBER_CONFIDENCE,
                )
            ]
        return candidates

    def _apply_explicit_remember_boost(self, candidate: MemoryCandidate) -> None:
        """Raise candidate priority when user explicitly asks to remember."""
        candidate.importance = max(
            candidate.importance, self.EXPLICIT_REMEMBER_IMPORTANCE
        )
        candidate.confidence = max(
            candidate.confidence, self.EXPLICIT_REMEMBER_CONFIDENCE
        )

    def _record_auto_remember_event(
        self,
        *,
        status: str,
        details: dict[str, object],
        memory_id: int | None = None,
    ) -> None:
        """Record an auto-memory lifecycle event."""
        self.memory_store.record_event(
            operation="auto_remember",
            status=status,
            memory_id=memory_id,
            details=details,
        )

    def _record_failure(
        self,
        *,
        result: AutoMemoryResult,
        candidate: MemoryCandidate,
        action: str,
        explicit_remember: bool,
        default_error: str,
    ) -> None:
        """Capture and emit a failed auto-memory write operation."""
        error_payload = self.memory_store.get_last_error() or {}
        error_text = str(error_payload.get("error", default_error))
        result.failures.append(
            AutoMemoryFailure(
                memory_text=candidate.memory_text,
                type=candidate.type,
                tag=candidate.tag,
                error=error_text,
            )
        )
        self._record_auto_remember_event(
            status=MemoryStore.EVENT_ERROR,
            details={
                "memory_text": candidate.memory_text[:200],
                "type": candidate.type,
                "tag": candidate.tag,
                "action": action,
                "error": error_text,
                "explicit_remember": explicit_remember,
            },
        )

    def _handle_duplicate_candidate(
        self,
        *,
        result: AutoMemoryResult,
        candidate: MemoryCandidate,
        explicit_remember: bool,
    ) -> None:
        """Revive an existing memory when an exact candidate already exists."""
        revived = self.memory_store.revive_exact_memory(
            memory_text=candidate.memory_text,
            type=candidate.type,
            context=candidate.tag,
        )
        if revived:
            memory_id = int(revived["id"])
            self._record_auto_remember_event(
                status=MemoryStore.EVENT_SUCCESS,
                memory_id=memory_id,
                details={
                    "memory_id": memory_id,
                    "type": candidate.type,
                    "tag": candidate.tag,
                    "new_importance": revived.get("importance"),
                    "access_count": revived.get("access_count"),
                    "action": "revived_duplicate",
                    "explicit_remember": explicit_remember,
                },
            )
            result.revived_ids.append(memory_id)
            return

        self._record_failure(
            result=result,
            candidate=candidate,
            action="revive_failed",
            explicit_remember=explicit_remember,
            default_error="Failed to revive existing memory",
        )

    def _handle_new_candidate(
        self,
        *,
        result: AutoMemoryResult,
        candidate: MemoryCandidate,
        source: str,
        explicit_remember: bool,
    ) -> None:
        """Insert a new memory candidate and record outcome."""
        memory_id = self.memory_store.remember(
            memory_text=candidate.memory_text,
            type=candidate.type,
            context=candidate.tag,
            importance=candidate.importance,
            confidence=candidate.confidence,
            source=source,
        )
        if memory_id:
            self._record_auto_remember_event(
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
            result.inserted_ids.append(memory_id)
            return

        self._record_failure(
            result=result,
            candidate=candidate,
            action="store_failed",
            explicit_remember=explicit_remember,
            default_error="Failed to store memory",
        )

    def process_turn(self, user_message: str, assistant_message: str) -> List[int]:
        """Extract and store memories for a chat turn.

        Args:
            user_message: User message text.
            assistant_message: Assistant response text.

        Returns:
            Stored memory ids.
        """
        result = AutoMemoryResult()
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]

        explicit_remember = self._is_explicit_remember_intent(user_message)
        candidates = self._extract_candidates(
            messages=messages,
            user_message=user_message,
            explicit_remember=explicit_remember,
        )

        if not candidates:
            self.last_result = result
            return []

        source = (f"user: {user_message}\nassistant: {assistant_message}")[
            : self.source_char_limit
        ]

        for candidate in candidates:
            if explicit_remember:
                self._apply_explicit_remember_boost(candidate)

            if self.memory_store.memory_exists(
                memory_text=candidate.memory_text,
                type=candidate.type,
                context=candidate.tag,
            ):
                self._handle_duplicate_candidate(
                    result=result,
                    candidate=candidate,
                    explicit_remember=explicit_remember,
                )
                continue

            self._handle_new_candidate(
                result=result,
                candidate=candidate,
                source=source,
                explicit_remember=explicit_remember,
            )

        self.last_result = result
        return result.all_ids
