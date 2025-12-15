import logging
from src.services.memory.vector_store import MemoryStore

logger = logging.getLogger(__name__)


def create_store_memory_tool(memory_store: MemoryStore):
    """Create the store_memory_tool function bound to a memory store."""

    def store_memory_tool(
        memory_text: str,
        type: str,
        importance: float = 1.0,
        confidence: float = 0.0,
    ) -> str:
        """
        Store a semantic memory in the database.

        Args:
            memory_text: Concise description of what to remember
            type: Memory type (preference, fact, task, insight)
            importance: Importance score 0.0-3.0 (0=low, 1=normal, 2=high, 3=critical)
            confidence: Confidence score 0.0-1.0

        Returns:
            Confirmation message with memory ID
        """
        if not memory_store:
            return "Error: Memory store not available"

        try:
            memory_id = memory_store.remember(
                memory_text=memory_text,
                type=type,
                context="chat",  # Default context
                importance=importance,
                confidence=confidence,
            )
            if memory_id:
                logger.info(
                    f"Stored memory {memory_id} via tool: {memory_text[:50]}..."
                )
                return f"Stored memory #{memory_id}"
            return "Failed to store memory"
        except Exception as e:
            logger.error(f"Tool call error: {e}")
            return f"Error: {e}"

    return store_memory_tool
