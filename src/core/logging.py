import logging
import os
from pathlib import Path


def setup_logging(debug: bool = False):
    """Setup logging with proper configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    log_level = (
        logging.DEBUG
        if (debug or os.getenv("DEBUG", "").lower() in ("1", "true", "yes"))
        else logging.INFO
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ollama_chat.log"),
            logging.StreamHandler(),
        ],
    )
    # Reduce noise on stdout
    logging.getLogger().handlers[1].setLevel(logging.WARNING)

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        f"Logging initialized at {logging.getLevelName(log_level)} level"
    )
