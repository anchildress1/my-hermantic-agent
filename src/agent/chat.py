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


DEFAULT_MEMORY_FILE = "data/memory.json"


def setup_logging():
    """Setup logging with proper configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("logs/ollama_chat.log"),
            logging.StreamHandler(),  # Also log to console for errors
        ],
    )
    # Set console handler to WARNING and above only
    logging.getLogger().handlers[1].setLevel(logging.WARNING)
    logger.info("Logging initialized")


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


def save_memory(messages: List[Dict], memory_file: str = DEFAULT_MEMORY_FILE):
    """Save conversation history to file with atomic write."""
    memory_path = Path(memory_file)
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
        logger.info(f"Saved {len(messages)} messages to {memory_file}")
        print(f"💾 Memory saved to {memory_file}")

    except Exception as e:
        logger.error(f"Failed to save memory: {e}")
        if "tmp_path" in locals():
            Path(tmp_path).unlink(missing_ok=True)
        raise


def load_memory(memory_file: str = DEFAULT_MEMORY_FILE) -> List[Dict]:
    """Load conversation history from file."""
    memory_path = Path(memory_file)
    if not memory_path.exists():
        logger.info(f"No existing memory file at {memory_file}")
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
        logger.error(f"Corrupted memory file: {e}")
        # Try to load backup
        backup = memory_path.with_suffix(".json.bak")
        if backup.exists():
            logger.info("Attempting to load from backup")
            print("⚠️  Memory file corrupted, loading from backup...")
            with open(backup, "r") as f:
                data = json.load(f)
            return data.get("messages", [])
        else:
            logger.error("No backup available")
            print("❌ Memory file corrupted and no backup available")
            return []
    except Exception as e:
        logger.error(f"Failed to load memory: {e}")
        return []


def archive_memory_snapshot(memory_file: str, prefix: str = "clear") -> Optional[Path]:
    """Archive the current memory file before clearing context."""
    memory_path = Path(memory_file)
    if not memory_path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_name = f"{memory_path.stem}-{prefix}-{timestamp}.json"
    backup_path = memory_path.with_name(backup_name)

    try:
        shutil.copy2(memory_path, backup_path)
        logger.info(f"Archived memory before clear: {backup_path}")
        return backup_path
    except Exception as e:
        logger.warning(f"Unable to archive memory snapshot: {e}")
        return None


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


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


def chat_loop(template: Dict, memory_file: str = DEFAULT_MEMORY_FILE):
    """Run interactive chat loop with Ollama."""
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
    # Keep 75% for history, 25% for generation
    max_history_tokens = int(max_context * 0.75)

    memory_path = Path(memory_file)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if memory_path.exists():
        print(
            f"⚠️  Found saved conversation at {memory_file}. Use '/load' to restore it manually."
        )

    print(f"🤖 Ollama Chat (Model: {model})")
    print(
        f"📊 Context window: {max_context} tokens (keeping {max_history_tokens} for history)"
    )
    print(f"📊 Parameters: {json.dumps(params, indent=2)}")
    print("\nConversation Commands:")
    print("  '/quit', '/exit', or '/bye' - End conversation and save")
    print("  '/context' - Show full conversation context with token counts")
    print("  '/context brief' - Show brief context summary")
    print("  '/clear' - Clear conversation history (keeps system prompt)")
    print("  '/save' - Manually save current conversation")
    print(
        "  '/load [file]' - Reload saved memory from a JSON file (defaults to saved memory)"
    )
    print("  '/stream' - Toggle streaming mode")
    print("  '/trim' - Manually trim old messages")

    if memory_store:
        print("\nMemory Commands:")
        print("  '/remember <text>' - Store a memory manually")
        print("  '/recall <query>' - Search semantic memories")
        print("  '/memories [context]' - List recent memories")
        print("  '/forget <id>' - Delete a memory by ID")
        print("  '/contexts' - List all memory contexts")
        print("  '/stats' - Show memory statistics")
    print()

    streaming = True

    while True:
        try:
            user_input = input("\n💬 You: ").strip()

            if user_input.lower() in ["/quit", "/exit", "/bye"]:
                save_memory(messages, memory_file)
                print("Later!")
                break

            if user_input == "/context":
                print_context(messages, show_full=True)
                continue

            if user_input == "/context brief":
                print_context(messages, show_full=False)
                continue

            if user_input == "/clear":
                archive_path = archive_memory_snapshot(memory_file)
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                save_memory(messages, memory_file)
                if archive_path:
                    print(f"📦 Previous conversation archived to {archive_path}")
                print("🗑️  Context cleared and saved!")
                continue

            if user_input == "/save":
                save_memory(messages, memory_file)
                continue

            if user_input.startswith("/load"):
                parts = user_input.split(maxsplit=1)
                target_file = parts[1] if len(parts) > 1 else memory_file
                loaded_messages = load_memory(target_file)

                if loaded_messages:
                    messages = loaded_messages
                    if system_prompt:
                        if messages and messages[0].get("role") == "system":
                            messages[0]["content"] = system_prompt
                        else:
                            messages.insert(
                                0, {"role": "system", "content": system_prompt}
                            )
                    print(f"🔄 Memory loaded from {target_file}")
                else:
                    messages = []
                    if system_prompt:
                        messages.append({"role": "system", "content": system_prompt})
                    print(f"⚠️  No saved memory loaded from {target_file}")
                continue

            if user_input == "/stream":
                streaming = not streaming
                print(f"🔄 Streaming {'enabled' if streaming else 'disabled'}")
                continue

            if user_input == "/trim":
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    save_memory(messages, memory_file)
                    print(f"✂️  Context trimmed to {len(messages)} messages")
                else:
                    print(
                        f"✓ Context is within limits ({count_message_tokens(messages)} tokens)"
                    )
                continue

            # Memory commands
            if memory_store:
                if user_input.startswith("/remember "):
                    text = user_input[10:].strip()
                    if not text:
                        print("Usage: /remember <text>")
                        continue

                    # Prompt for type and context
                    print("\nMemory type (preference/fact/task/insight):")
                    mem_type = input("  Type: ").strip().lower()
                    if mem_type not in memory_store.VALID_TYPES:
                        print(
                            f"❌ Invalid type. Must be one of: {memory_store.VALID_TYPES}"
                        )
                        continue

                    print("\nContext (e.g., work, personal, project-name):")
                    context = input("  Context: ").strip()
                    if not context:
                        print("❌ Context cannot be empty")
                        continue

                    try:
                        mem_id = memory_store.remember(text, mem_type, context)
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
                                print(
                                    f"  [{r['id']}] {r['type'].upper()} | {r['context']}"
                                )
                                print(f"      {r['memory_text']}")
                                print(
                                    f"      Similarity: {r['similarity']:.3f} | Accessed: {r['access_count']} times"
                                )
                                print()
                        else:
                            print("No memories found")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input.startswith("/memories"):
                    parts = user_input.split(maxsplit=1)
                    context_filter = parts[1] if len(parts) > 1 else None

                    try:
                        results = (
                            memory_store.recall(
                                query="",
                                context=context_filter,
                                limit=10,
                                use_semantic=False,
                            )
                            if context_filter
                            else []
                        )

                        # If no filter, get recent memories via stats
                        if not context_filter:
                            # Just show stats instead
                            stats = memory_store.stats()
                            if stats:
                                print("\n📊 Memory Statistics:")
                                print(f"  Total memories: {stats['total_memories']}")
                                print(f"  Unique types: {stats['unique_types']}")
                                print(f"  Unique contexts: {stats['unique_contexts']}")
                                print(
                                    f"  Avg confidence: {stats['avg_confidence']:.2f}"
                                )
                                print(f"  Last memory: {stats['last_memory_at']}")
                            else:
                                print("No memories stored yet")
                        else:
                            if results:
                                print(f"\n📋 Memories in context '{context_filter}':\n")
                                for r in results:
                                    print(f"  [{r['id']}] {r['type'].upper()}")
                                    print(f"      {r['memory_text'][:80]}...")
                                    print()
                            else:
                                print(
                                    f"No memories found in context '{context_filter}'"
                                )
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

                if user_input == "/contexts":
                    try:
                        contexts = memory_store.list_contexts()
                        if contexts:
                            print(f"\n📁 Available contexts ({len(contexts)}):")
                            for ctx in contexts:
                                print(f"  - {ctx}")
                            print()
                        else:
                            print("No contexts found")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

                if user_input == "/stats":
                    try:
                        stats = memory_store.stats()
                        if stats:
                            print("\n📊 Memory Statistics:")
                            print(f"  Total memories: {stats['total_memories']}")
                            print(f"  Unique types: {stats['unique_types']}")
                            print(f"  Unique contexts: {stats['unique_contexts']}")
                            print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
                            print(f"  Last memory: {stats['last_memory_at']}")
                            print()
                        else:
                            print("Failed to retrieve stats")
                    except Exception as e:
                        print(f"❌ Error: {e}")
                    continue

            if not user_input:
                continue

            # Check if we need to trim context before adding new message
            current_tokens = count_message_tokens(messages)
            if current_tokens > max_history_tokens:
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    print(
                        f"✂️  Auto-trimmed context to fit within {max_history_tokens} tokens"
                    )
                    save_memory(messages, memory_file)

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
            save_memory(messages, memory_file)
            if memory_store:
                memory_store.close()
            print("Goodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)
            print(f"\n❌ Error: {e}\n")
