#!/usr/bin/env python3
"""Main entry point for Ollama agent."""

import sys
from pathlib import Path

# Add src to python path if needed (though local import should work)
sys.path.append(str(Path(__file__).parent))

from src.main import main

if __name__ == "__main__":
    main()
