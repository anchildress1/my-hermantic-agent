import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from src.core.logging import setup_logging
from src.core.config import load_config
from src.services.memory.vector_store import MemoryStore
from src.services.llm.ollama_service import OllamaService
from src.interfaces.cli.chat import chat_loop

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """Validate required environment variables."""
    required = {
        "OPENAI_API_KEY": "OpenAI API key for embeddings",
    }

    optional = {
        "MEMORY_DB_URL": "TimescaleDB connection string for semantic memory",
        "OPENAI_EMBEDDING_MODEL": "OpenAI embedding model",
        "OPENAI_EMBEDDING_DIM": "OpenAI embedding dimensions",
    }

    missing = []
    for var, description in required.items():
        if not os.getenv(var):
            missing.append(f"  ❌ {var}: {description}")

    missing_optional = []
    for var, description in optional.items():
        if not os.getenv(var):
            missing_optional.append(f"  ⚠️  {var}: {description}")

    if missing:
        print("❌ Missing required environment variables:")
        print("\n".join(missing))
        print("\nCopy .env.example to .env and fill in values")
        return False

    if missing_optional:
        print("⚠️  Optional environment variables not set:")
        print("\n".join(missing_optional))
        print("\nSemantic memory features will be unavailable without MEMORY_DB_URL")
        print()

    return True


def main():
    setup_logging()
    logger.info("Starting Ollama Agent")

    if not validate_environment():
        return

    template_path = Path(os.getenv("TEMPLATE_CONFIG", "config/template.yaml"))

    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        print(f"❌ Template file not found: {template_path}")
        return

    try:
        config = load_config(template_path)

        # Initialize LLM service
        llm_service = OllamaService(
            model=config.model, parameters=config.parameters.model_dump()
        )
        logger.info(f"LLM service initialized with model: {config.model}")

        # Initialize semantic memory store
        memory_store = None
        if os.getenv("MEMORY_DB_URL"):
            try:
                memory_store = MemoryStore()
                logger.info("Semantic memory store initialized")
                print("✓ Semantic memory connected")
            except Exception as e:
                logger.warning(f"Semantic memory unavailable: {e}")
                print(f"⚠️  Semantic memory unavailable: {e}")
                print("   Continuing without semantic memory...")

        chat_loop(
            config,
            context_file="data/memory.json",
            llm_service=llm_service,
            memory_store=memory_store,
        )

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    main()
