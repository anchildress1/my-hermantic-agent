#!/usr/bin/env python3
"""Main entry point for Ollama agent."""

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from src.agent.chat import setup_logging, load_template, chat_loop

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


def validate_environment() -> bool:
    """Validate required environment variables."""
    required = {
        'OPENAI_API_KEY': 'OpenAI API key for embeddings (get from https://platform.openai.com/api-keys)',
    }
    
    optional = {
        'MEMORY_DB_URL': 'TimescaleDB connection string for semantic memory',
        'OPENAI_EMBEDDING_MODEL': 'OpenAI embedding model (default: text-embedding-3-small)',
        'OPENAI_EMBEDDING_DIM': 'OpenAI embedding dimensions (default: auto-detected)'
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
        print("\nCopy .env.example to .env and fill in values:")
        print("  cp .env.example .env")
        return False
    
    if missing_optional:
        print("⚠️  Optional environment variables not set:")
        print("\n".join(missing_optional))
        print("\nSemantic memory features will be unavailable without MEMORY_DB_URL")
        print()
    
    return True


def main():
    # Setup logging first
    Path("logs").mkdir(exist_ok=True)
    setup_logging()
    
    logger.info("Starting Ollama Agent")
    
    # Validate environment
    if not validate_environment():
        return
    
    # Load template
    template_path = Path("config/template.yaml")
    
    if not template_path.exists():
        logger.error(f"Template file not found: {template_path}")
        print(f"❌ Template file not found: {template_path}")
        return
    
    try:
        template = load_template(template_path)
        chat_loop(template)
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    main()
