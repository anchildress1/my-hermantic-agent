"""Command-line chat interface."""

import logging
from typing import Optional

from src.agent.chat_session import ChatSession
from src.core.config import AgentConfig
from src.services.llm.base import LLMService
from src.services.memory.auto_writer import AutoMemoryWriter
from src.services.memory.vector_store import MemoryStore

logger = logging.getLogger(__name__)


def chat_loop(
    config: AgentConfig,
    context_file: str,
    llm_service: LLMService,
    memory_store: Optional[MemoryStore] = None,
    auto_memory_writer: Optional[AutoMemoryWriter] = None,
) -> None:
    """Run interactive chat loop with Ollama.

    Args:
        config: Chat configuration object
        context_file: Path to save/load conversation history
        llm_service: Injected LLM service
        memory_store: Optional vector memory store for semantic memory operations
        auto_memory_writer: Optional auto-memory writer
    """
    session = ChatSession(
        config,
        context_file,
        llm_service,
        memory_store,
        auto_memory_writer=auto_memory_writer,
    )
    session.run()
