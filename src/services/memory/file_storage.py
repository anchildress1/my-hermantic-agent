import json
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_FILE = "data/memory.json"


def save_chat_history(messages: List[Dict], file_path: str = DEFAULT_CONTEXT_FILE):
    """Save conversation context to file with atomic write."""
    memory_path = Path(file_path)
    memory_path.parent.mkdir(parents=True, exist_ok=True)

    data = {"timestamp": datetime.now().isoformat(), "messages": messages}

    try:
        # Write to temp file first
        with tempfile.NamedTemporaryFile(
            mode="w", dir=memory_path.parent, delete=False, suffix=".tmp"
        ) as tmp:
            json.dump(data, tmp, indent=2)
            tmp_path = tmp.name

        # Backup existing file
        if memory_path.exists():
            backup = memory_path.with_suffix(".json.bak")
            shutil.copy2(memory_path, backup)
            logger.debug(f"Backed up to {backup}")

        # Atomic rename
        shutil.move(tmp_path, memory_path)
        logger.info(f"Saved {len(messages)} messages to {file_path}")
        print(f"💾 Memory saved to {file_path}")

    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise


def load_chat_history(file_path: str = DEFAULT_CONTEXT_FILE) -> List[Dict]:
    """Load conversation context from file."""
    memory_path = Path(file_path)
    if not memory_path.exists():
        logger.info(f"No existing context file at {file_path}")
        return []

    try:
        with open(memory_path, "r") as f:
            data = json.load(f)

        messages = data.get("messages", [])
        timestamp = data.get("timestamp", "unknown")
        logger.info(f"Loaded {len(messages)} messages from {timestamp}")
        print(f"📂 Loaded memory from {timestamp}")
        return messages

    except json.JSONDecodeError as e:
        logger.error(f"Corrupted context file: {e}")
        # Try to load backup
        backup = memory_path.with_suffix(".json.bak")
        if backup.exists():
            logger.info("Attempting to load from backup")
            print("⚠️  Context file corrupted, loading from backup...")
            with open(backup, "r") as f:
                data = json.load(f)
            return data.get("messages", [])
        else:
            logger.error("No backup available")
            print("❌ Memory file corrupted and no backup available")
            return []
    except Exception as e:
        logger.error(f"Failed to load context: {e}")
        return []


def archive_chat_history(file_path: str, prefix: str = "clear") -> Optional[Path]:
    """Archive the current context file before clearing."""
    memory_path = Path(file_path)
    if not memory_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_name = f"{memory_path.stem}-{prefix}-{timestamp}.json"
    backup_path = memory_path.with_name(backup_name)

    try:
        shutil.copy2(memory_path, backup_path)
        logger.info(f"Archived context before clear: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Unable to archive context snapshot: {e}")
        return None
