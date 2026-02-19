import logging
import signal
from dotenv import load_dotenv

from src.core.logging import setup_logging
from src.core.config import load_config, get_settings, get_config_path
from src.services.memory.auto_writer import AutoMemoryWriter
from src.services.memory.langmem_extractor import LangMemExtractor
from src.services.memory.vector_store import MemoryStore
from src.services.llm.ollama_service import OllamaService
from src.interfaces.cli.chat import chat_loop

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def _install_signal_handlers() -> None:
    """Install process signal handlers for graceful shutdown."""

    def _handle_termination(signum, _frame) -> None:
        signal_name = signal.Signals(signum).name
        logger.warning(f"Received {signal_name}, shutting down gracefully")
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_termination)


def main():
    setup_logging()
    _install_signal_handlers()
    logger.info("Starting Ollama Agent")

    try:
        settings = get_settings()
    except Exception as e:
        print("❌ Configuration error:")
        print(f"  {e}")
        print("\nMake sure your .env file is correctly configured with OPENAI_API_KEY.")
        return 1

    # Resolve config path based on environment
    template_path = get_config_path(settings)
    logger.info(
        f"Using configuration: {template_path} (environment: {settings.environment})"
    )

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
        auto_memory_writer = None
        if settings.memory_db_url:
            try:
                memory_store = MemoryStore(settings=settings)
                logger.info("Semantic memory store initialized")
                print("✓ Semantic memory connected")

                if settings.langmem_enabled:
                    try:
                        extractor_model = settings.langmem_model or config.model
                        extractor = LangMemExtractor(
                            model=extractor_model,
                            model_provider=settings.langmem_model_provider,
                            temperature=settings.langmem_temperature,
                            max_memories_per_turn=settings.langmem_max_memories_per_turn,
                            default_tag=settings.langmem_default_tag,
                        )
                        auto_memory_writer = AutoMemoryWriter(
                            memory_store=memory_store,
                            extractor=extractor,
                        )
                        logger.info("LangMem auto-memory writer initialized")
                        print("✓ LangMem auto-memory enabled")
                    except Exception as e:
                        logger.warning(f"LangMem auto-memory disabled: {e}")
                        print(f"⚠️  LangMem auto-memory disabled: {e}")
            except Exception as e:
                logger.warning(f"Semantic memory unavailable: {e}")
                print(f"⚠️  Semantic memory unavailable: {e}")
                print("   Continuing without semantic memory...")

        chat_loop(
            config,
            context_file="data/memory.json",
            llm_service=llm_service,
            memory_store=memory_store,
            auto_memory_writer=auto_memory_writer,
        )

    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    main()
