import yaml
import ollama
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

# Setup module logger
logger = logging.getLogger(__name__)


DEFAULT_CONTEXT_FILE = "data/memory.json"


def setup_logging(debug: bool = False):
    """Setup logging with proper configuration."""
    import os

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
    logging.getLogger().handlers[1].setLevel(logging.WARNING)
    logger.info(f"Logging initialized at {logging.getLevelName(log_level)} level")


def load_template(template_path: Path) -> Dict:
    """Load model template configuration from YAML file."""
    try:
        with open(template_path, "r") as f:
            template = yaml.safe_load(f)
            logger.info(f"Loaded template from {template_path}")
            return template
    except FileNotFoundError:
        logger.error(f"Template file not found: {template_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Invalid YAML in template: {e}")
        raise


def save_memory(messages: List[Dict], memory_file: str = DEFAULT_CONTEXT_FILE):
    """Save conversation context to file with atomic write."""
    save_context(messages, memory_file)


def save_context(messages: List[Dict], context_file: str = DEFAULT_CONTEXT_FILE):
    """Save conversation context to file with atomic write."""
    memory_path = Path(context_file)
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
        logger.info(f"Saved {len(messages)} messages to {context_file}")
        print(f"💾 Memory saved to {context_file}")

    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise


def load_memory(memory_file: str = DEFAULT_CONTEXT_FILE) -> List[Dict]:
    """Load conversation context from file."""
    return load_context(memory_file)


def load_context(context_file: str = DEFAULT_CONTEXT_FILE) -> List[Dict]:
    """Load conversation context from file."""
    memory_path = Path(context_file)
    if not memory_path.exists():
        logger.info(f"No existing context file at {context_file}")
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


def archive_memory_snapshot(memory_file: str, prefix: str = "clear") -> Optional[Path]:
    """Archive the current context file before clearing."""
    return archive_context_snapshot(memory_file, prefix)


def archive_context_snapshot(
    context_file: str, prefix: str = "clear"
) -> Optional[Path]:
    """Archive the current context file before clearing."""
    memory_path = Path(context_file)
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


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


# ANSI color helpers (simple, no external deps)
ANSI_RESET = "\u001b[0m"
ANSI_BOLD = "\u001b[1m"
ANSI_CYAN = "\u001b[36m"
ANSI_YELLOW = "\u001b[33m"
ANSI_GREEN = "\u001b[32m"


def print_help(memory_store: Optional[object]):
    """Print compact, colored help for commands."""
    print()
    print(f"{ANSI_BOLD}Commands{ANSI_RESET}")
    print(f"  {ANSI_CYAN}/?{ANSI_RESET}         Show this help")
    print(f"  {ANSI_CYAN}/quit{ANSI_RESET}        Exit and save")
    print(
        f"  {ANSI_CYAN}/clear{ANSI_RESET}       Clear conversation (keeps system prompt)"
    )
    print(f"  {ANSI_CYAN}/save{ANSI_RESET}        Save conversation manually")
    print(
        f"  {ANSI_CYAN}/load [file]{ANSI_RESET} Load saved context from JSON (defaults to saved context)"
    )
    print(f"  {ANSI_CYAN}/trim{ANSI_RESET}        Trim old messages to fit context")
    if memory_store:
        print()
        print(f"{ANSI_BOLD}Memory Commands{ANSI_RESET} (cloud PostgreSQL only)")
        print(
            f"  {ANSI_CYAN}/remember <text>{ANSI_RESET}      Store a memory (supports type=, tag=, importance=, confidence=)"
        )
        print(
            f"  {ANSI_CYAN}/recall <query>{ANSI_RESET}       Search semantic memories"
        )
        print(
            f"  {ANSI_CYAN}/memories [tag]{ANSI_RESET}       List recent memories or by tag"
        )
        print(f"  {ANSI_CYAN}/forget <id>{ANSI_RESET}          Delete memory by ID")
        print(f"  {ANSI_CYAN}/tags{ANSI_RESET}                 List memory tags")
        print(f"  {ANSI_CYAN}/stats{ANSI_RESET}                Show memory statistics")
    print()


def count_message_tokens(messages: list) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get("content", ""))
    return total


def trim_context(
    messages: List[Dict], max_tokens: int = 6000, keep_recent: int = 10
) -> Tuple[List[Dict], bool]:
    """
    Trim old messages if context is too large.
    Always keeps system message and most recent messages.
    Returns (trimmed_messages, was_trimmed)
    """
    if not messages:
        return messages, False

    total_tokens = count_message_tokens(messages)

    if total_tokens <= max_tokens:
        return messages, False

    # Always keep system message (first) and recent messages
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    recent_messages = messages[-keep_recent:]

    # Calculate tokens for what we're keeping
    kept_tokens = count_message_tokens(recent_messages)
    if system_msg:
        kept_tokens += estimate_tokens(system_msg["content"])

    # Build trimmed context
    trimmed = []
    if system_msg:
        trimmed.append(system_msg)

    # Add a summary message about what was trimmed
    num_trimmed = len(messages) - len(recent_messages) - (1 if system_msg else 0)
    if num_trimmed > 0:
        summary = {
            "role": "system",
            "content": f"[Context trimmed: {num_trimmed} older messages removed to stay within token limit]",
        }
        trimmed.append(summary)

    trimmed.extend(recent_messages)

    logger.info(
        f"Context trimmed: {len(messages)} -> {len(trimmed)} messages, {total_tokens} -> {kept_tokens} tokens"
    )

    return trimmed, True


def print_context(messages: List[Dict], show_full: bool = False):
    """Print current conversation context."""
    print("\n" + "=" * 60)
    print("📋 CURRENT CONTEXT")
    print("=" * 60)
    total_tokens = count_message_tokens(messages)
    print(f"Total messages: {len(messages)} | Estimated tokens: {total_tokens}")
    print("=" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg["content"]
        tokens = estimate_tokens(content)
        if not show_full and len(content) > 100:
            content = content[:100] + "..."
        print(f"\n[{i}] {role} (~{tokens} tokens):")
        print(f"  {content}")
    print("=" * 60 + "\n")


def check_ollama_connection(model: str) -> bool:
    """Verify Ollama is running and model is available."""
    try:
        model_list = ollama.list()
        available_models = [m["model"] for m in model_list.get("models", [])]

        # Check for exact match or partial match (model might have :tag)
        model_found = any(model in m or m in model for m in available_models)

        if not model_found:
            logger.error(f"Model '{model}' not found. Available: {available_models}")
            print(f"❌ Model '{model}' not found")
            print(f"   Available models: {', '.join(available_models)}")
            print(f"   Run: ollama pull {model}")
            return False

        logger.info(f"Ollama connection verified, model '{model}' available")
        return True

    except Exception as e:
        logger.error(f"Ollama service not running: {e}")
        print(f"❌ Ollama service not running: {e}")
        print("   Start it with: ollama serve")
        return False


def chat_loop(
    template: Dict, context_file: str = DEFAULT_CONTEXT_FILE, memory_file: str = None
):  # pragma: no cover
    """Run interactive chat loop with Ollama."""
    if memory_file is None:
        memory_file = context_file
    model = template.get("model", "llama3.2")
    system_prompt = template.get("system", "")
    params = template.get("parameters", {})

    # Validate Ollama connection
    if not check_ollama_connection(model):
        return

    # Initialize semantic memory store
    memory_store = None
    try:
        from src.agent.memory import MemoryStore

        memory_store = MemoryStore()
        logger.info("Semantic memory store initialized")
        print("✓ Semantic memory connected")
    except Exception as e:
        logger.warning(f"Semantic memory unavailable: {e}")
        print(f"⚠️  Semantic memory unavailable: {e}")
        print("   Continuing without semantic memory...")

    # Get context window size from params, default to 8192
    max_context = params.get("num_ctx", 8192)
    max_history_tokens = int(max_context * 0.75)

    memory_path = Path(context_file)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if memory_path.exists():
        print(
            f"{ANSI_YELLOW}⚠️  Found saved conversation at {context_file}.{ANSI_RESET} Use {ANSI_CYAN}/load{ANSI_RESET} to restore it."
        )

    print(f"{ANSI_GREEN}🤖 Ollama Chat{ANSI_RESET} — Model: {model}")
    print(f"Type {ANSI_CYAN}/?{ANSI_RESET} for commands")

    # Memory commands are intentionally not printed at startup.
    # Use '/?' to view memory commands (they will appear only when requested).
    print()

    streaming = True

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() in ["/bye"]:
                save_context(messages, context_file)
                print("Later!")
                break

            if user_input == "/?":
                print_help(memory_store)
                continue

            if user_input == "/context":
                print_context(messages, show_full=True)
                continue

            if user_input == "/context brief":
                print_context(messages, show_full=False)
                continue

            if user_input == "/clear":
                archive_path = archive_context_snapshot(context_file)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                save_context(messages, context_file)
                if archive_path:
                    print(f"📦 Previous conversation archived to {archive_path}")
                print("🗑️  Context cleared and saved!")
                continue

            if user_input == "/save":
                save_context(messages, context_file)
                continue

            if user_input.startswith("/load"):
                tokens = user_input.split()[1:]

                if not tokens:
                    loaded_messages = load_context(context_file)
                    if loaded_messages:
                        messages = loaded_messages
                        if system_prompt:
                            if messages and messages[0].get("role") == "system":
                                messages[0]["content"] = system_prompt
                            else:
                                messages.insert(
                                    0, {"role": "system", "content": system_prompt}
                                )
                        print(
                            f"{ANSI_GREEN}🔄 Context loaded from {context_file}{ANSI_RESET}"
                        )
                    else:
                        messages = []
                        if system_prompt:
                            messages.append(
                                {"role": "system", "content": system_prompt}
                            )
                        print(
                            f"{ANSI_YELLOW}⚠️  No saved context loaded from {context_file}{ANSI_RESET}"
                        )
                    continue

                combined: List[Dict] = []
                any_loaded = False
                for f in tokens:
                    loaded = load_context(f)
                    if loaded:
                        combined.extend(loaded)
                        any_loaded = True

                if any_loaded:
                    messages = combined
                    print(
                        f"{ANSI_GREEN}🔄 Context loaded from: {' '.join(tokens)}{ANSI_RESET}"
                    )
                else:
                    print(
                        f"{ANSI_YELLOW}⚠️  No saved context loaded from: {' '.join(tokens)}{ANSI_RESET}"
                    )
                continue

            if user_input == "/stream":
                streaming = not streaming
                print(f"🔄 Streaming {'enabled' if streaming else 'disabled'}")
                continue

            if user_input == "/trim":
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    save_context(messages, context_file)
                    print(f"✂️  Context trimmed to {len(messages)} messages")
                else:
                    print(
                        f"✓ Context is within limits ({count_message_tokens(messages)} tokens)"
                    )
                continue

            if memory_store:
                if user_input.startswith("/remember "):
                    import re

                    text = user_input[10:].strip()
                    if not text:
                        print(
                            "Usage: /remember [type=<type>] [tag=<tag>] [importance=<float>] [confidence=<float>] [source=<src>] <text>"
                        )
                        continue

                    params = {}
                    param_pattern = r"(type|tag|importance|confidence|source)=(\S+)"
                    matches = re.findall(param_pattern, text)

                    for key, value in matches:
                        params[key] = value
                        text = re.sub(
                            rf"{key}={re.escape(value)}\s*", "", text, count=1
                        )

                    text = text.strip()
                    if not text:
                        print("❌ Memory text cannot be empty")
                        continue

                    mem_type = params.get("type")
                    if not mem_type:
                        print("\nMemory type (preference/fact/task/insight):")
                        mem_type = input("  Type: ").strip().lower()

                    if mem_type not in memory_store.VALID_TYPES:
                        print(
                            f"❌ Invalid type. Must be one of: {memory_store.VALID_TYPES}"
                        )
                        continue

                    tag = params.get("tag")
                    if not tag:
                        print("\nTag (e.g., work, personal, project-name):")
                        tag = input("  Tag: ").strip()

                    if not tag:
                        print("❌ Tag cannot be empty")
                        continue

                    importance = float(params.get("importance", 1.0))
                    if importance > 2.0:
                        print(
                            f"⚠️  High importance ({importance}) - this memory will be prioritized in recall"
                        )

                    confidence = float(params.get("confidence", 1.0))
                    source = params.get("source")

                    try:
                        mem_id = memory_store.remember(
                            text,
                            mem_type,
                            context=tag,
                            importance=importance,
                            confidence=confidence,
                            source=source,
                        )
                        if mem_id:
                            print(f"✓ Memory stored with ID {mem_id}")
                        else:
                            print("❌ Failed to store memory")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input.startswith("/recall "):
                    query = user_input[8:].strip()
                    if not query:
                        print("Usage: /recall <query>")
                        continue

                    try:
                        results = memory_store.recall(query, limit=5)
                        if results:
                            print(f"\n🔍 Found {len(results)} relevant memories:\n")
                            for r in results:
                                importance_marker = (
                                    "🔴"
                                    if r["importance"] > 2.0
                                    else "🟡"
                                    if r["importance"] > 1.0
                                    else "🟢"
                                )
                                print(
                                    f"  [{r['id']:>4}] {r['type'].upper():<10} | {r['tag']:<20} {importance_marker}"
                                )
                                print(f"         {r['memory_text']}")
                                print(
                                    f"         Score: {r['similarity']:.3f} | Importance: {r['importance']:.1f} | Accessed: {r['access_count']}x\n"
                                )
                        else:
                            print("  No memories found")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input.startswith("/memories"):
                    parts = user_input.split(maxsplit=1)
                    tag_filter = parts[1] if len(parts) > 1 else None

                    try:
                        results = (
                            memory_store.recall(
                                query="",
                                tag=tag_filter,
                                limit=10,
                                use_semantic=False,
                            )
                            if tag_filter
                            else []
                        )

                        if not tag_filter:
                            stats = memory_store.stats()
                            if stats:
                                print("\n📊 Memory Statistics:")
                                print(f"  Total memories:   {stats['total_memories']}")
                                print(f"  Unique types:     {stats['unique_types']}")
                                print(f"  Unique tags:      {stats['unique_tags']}")
                                print(
                                    f"  Avg importance:   {stats['avg_importance']:.2f}"
                                )
                                print(
                                    f"  Avg confidence:   {stats['avg_confidence']:.2f}"
                                )
                                print(
                                    f"  Last memory:      {stats['last_memory_at']}\n"
                                )
                            else:
                                print("  No memories stored yet")
                        else:
                            if results:
                                print(f"\n📋 Memories with tag '{tag_filter}':\n")
                                for r in results:
                                    print(f"  [{r['id']:>4}] {r['type'].upper():<10}")
                                    print(f"         {r['memory_text'][:80]}...\n")
                            else:
                                print(f"  No memories found with tag '{tag_filter}'")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input.startswith("/forget "):
                    try:
                        mem_id = int(user_input[8:].strip())
                        if memory_store.forget(mem_id):
                            print(f"✓ Memory {mem_id} deleted")
                        else:
                            print(f"❌ Memory {mem_id} not found")
                    except ValueError:
                        print("Usage: /forget <id>")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input == "/tags":
                    try:
                        tags = memory_store.list_tags()
                        if tags:
                            print(f"\n📁 Available tags ({len(tags)}):")
                            for t in tags:
                                print(f"  • {t}")
                            print()
                        else:
                            print("  No tags found")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input == "/stats":
                    try:
                        stats = memory_store.stats()
                        if stats:
                            print("\n📊 Memory Statistics:")
                            print(f"  Total memories:   {stats['total_memories']}")
                            print(f"  Unique types:     {stats['unique_types']}")
                            print(f"  Unique tags:      {stats['unique_tags']}")
                            print(f"  Avg importance:   {stats['avg_importance']:.2f}")
                            print(f"  Avg confidence:   {stats['avg_confidence']:.2f}")
                            print(f"  Last memory:      {stats['last_memory_at']}\n")
                        else:
                            print("  Failed to retrieve stats")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

            if not user_input:
                continue

            current_tokens = count_message_tokens(messages)
            if current_tokens > max_history_tokens:
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    print(
                        f"✂️  Auto-trimmed context to fit within {max_history_tokens} tokens"
                    )
                    save_context(messages, context_file)

            # Add user message to history
            messages.append({"role": "user", "content": user_input})

            print("\n🤖 Assistant: ", end="", flush=True)

            # Get response from Ollama
            if streaming:
                full_response = ""
                last_100_chars = ""
                repetition_warnings = []
                repetition_count = 0

                stream = ollama.chat(
                    model=model, messages=messages, options=params, stream=True
                )

                for chunk in stream:
                    content = chunk["message"]["content"]

                    # Simple repetition detection
                    if len(content) > 10 and content in last_100_chars:
                        repetition_count += 1
                        if repetition_count == 3:  # First time hitting threshold
                            warning_msg = (
                                f"Repetition detected at position {len(full_response)}"
                            )
                            repetition_warnings.append(warning_msg)
                            logging.warning(
                                f"{warning_msg} | Content: '{content}' | Last 100: '{last_100_chars[-50:]}'"
                            )
                    else:
                        repetition_count = 0

                    print(content, end="", flush=True)
                    full_response += content
                    last_100_chars = (last_100_chars + content)[-100:]

                print()  # newline after stream

                # Show warnings if any repetition detected
                if repetition_warnings:
                    print(
                        f"\n⚠️  Repetition detected: {len(repetition_warnings)} instance(s) - check logs/ollama_chat.log"
                    )
                    logger.info(f"Full response with repetition:\n{full_response}\n")

                messages.append({"role": "assistant", "content": full_response})
            else:
                response = ollama.chat(model=model, messages=messages, options=params)
                assistant_message = response["message"]["content"]
                messages.append({"role": "assistant", "content": assistant_message})
                print(assistant_message)
                logger.debug(f"Assistant response: {len(assistant_message)} chars")

            # Show token count and context status
            current_tokens = count_message_tokens(messages)
            usage_pct = (current_tokens / max_history_tokens) * 100
            print(
                f"\n📊 Messages: {len(messages)} | Tokens: {current_tokens}/{max_history_tokens} ({usage_pct:.1f}%)"
            )

            if usage_pct > 90:
                print("⚠️  Context nearly full - will auto-trim on next message")

        except KeyboardInterrupt:
            print("\n\nSaving before exit...")
            save_context(messages, context_file)
            if memory_store:
                memory_store.close()
            print("Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
