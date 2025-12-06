#!/usr/bin/env python3
"""Main entry point for Ollama agent."""

from pathlib import Path
from dotenv import load_dotenv
from src.agent.chat import setup_logging, load_template, chat_loop

# Load environment variables
load_dotenv()


def main():
    # Setup logging
    Path("logs").mkdir(exist_ok=True)
    setup_logging()
    
    # Load template
    template_path = Path("config/template.yaml")
    
    if not template_path.exists():
        print(f"❌ Template file not found: {template_path}")
        return
    
    template = load_template(template_path)
    chat_loop(template)


if __name__ == "__main__":
    main()
