"""Command-line chat interface."""

from typing import Dict, Optional

from src.agent.chat_session import ChatSession
from src.services.memory.vector_store import MemoryStore


def chat_loop(
    config: Dict, context_file: str, memory_store: Optional[MemoryStore] = None
) -> None:
    """Run interactive chat loop with Ollama.

    Args:
        config: Chat configuration including model, system prompt, and parameters
        context_file: Path to save/load conversation history
        memory_store: Optional vector memory store for semantic memory operations
    """
    session = ChatSession(config, context_file, memory_store)
    session.run()
