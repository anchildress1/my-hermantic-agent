"""LLM service contracts used by the application layer."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Protocol


class LLMService(Protocol):
    """Interface for chat-capable language model services."""

    def check_connection(self) -> bool:
        """Verify the model backend is reachable and ready."""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[Any] | None = None,
        stream: bool = True,
    ) -> Generator[Any, None, None] | dict[str, Any]:
        """Execute a chat completion request."""
