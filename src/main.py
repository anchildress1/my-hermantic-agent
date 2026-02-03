import logging
from dotenv import load_dotenv
from src.core.logging import setup_logging
from src.core.config import load_config, get_settings
from src.services.memory.vector_store import MemoryStore
from src.services.llm.ollama_service import OllamaService
from src.interfaces.cli.chat import chat_loop

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def main():
    setup_logging()
    logger.info("Starting Ollama Agent")

    try:
        settings = get_settings()
    except Exception as e:
        print("❌ Configuration error:")
        print(f"  {e}")
        print("\nMake sure your .env file is correctly configured with OPENAI_API_KEY.")
        return 1

    template_path = settings.template_config

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
        if settings.memory_db_url:
            try:
                memory_store = MemoryStore(settings=settings)
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
