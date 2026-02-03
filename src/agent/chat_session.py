"""Chat session management with command handling."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import AgentConfig
from src.core.utils import count_message_tokens, estimate_tokens, trim_context
from src.services.llm.ollama_service import OllamaService
from src.services.memory.vector_store import MemoryStore
from src.services.memory.file_storage import (
    load_chat_history,
    save_chat_history,
    archive_chat_history,
)
from src.tools.memory_tool import create_store_memory_tool

logger = logging.getLogger(__name__)


class ChatSession:
    """Manages chat session state and command execution."""

    ANSI_RESET = "\u001b[0m"
    ANSI_BOLD = "\u001b[1m"
    ANSI_CYAN = "\u001b[36m"
    ANSI_YELLOW = "\u001b[33m"
    ANSI_GREEN = "\u001b[32m"

    def __init__(
        self,
        config: AgentConfig,
        context_file: str,
        llm_service: OllamaService,
        memory_store: Optional[MemoryStore] = None,
    ):
        """Initialize chat session.

        Args:
            config: Chat configuration object
            context_file: Path to save/load conversation history
            llm_service: Injected LLM service instance
            memory_store: Optional vector memory store for semantic memory operations
        """
        self.config = config
        self.context_file = context_file
        self.memory_store = memory_store
        self.ollama_service = llm_service

        self.model = config.model
        self.system_prompt = config.system
        self.params = config.parameters

        self.max_context = self.params.num_ctx
        self.max_history_tokens = int(self.max_context * 0.75)

        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]

        self.tools = []
        if self.memory_store:
            self.tools.append(create_store_memory_tool(self.memory_store))
            logger.info("Memory tool enabled")

    def cmd_help(self) -> None:
        """Print available commands."""
        print()
        print(f"{self.ANSI_BOLD}Commands{self.ANSI_RESET}")
        print(f"  {self.ANSI_CYAN}/?{self.ANSI_RESET}         Show this help")
        print(f"  {self.ANSI_CYAN}/quit{self.ANSI_RESET}        Exit and save")
        print(
            f"  {self.ANSI_CYAN}/clear{self.ANSI_RESET}       Clear conversation (keeps system prompt)"
        )
        print(
            f"  {self.ANSI_CYAN}/save{self.ANSI_RESET}        Save conversation manually"
        )
        print(
            f"  {self.ANSI_CYAN}/load [file]{self.ANSI_RESET} Load saved context from JSON (defaults to saved context)"
        )
        print(
            f"  {self.ANSI_CYAN}/trim{self.ANSI_RESET}        Trim old messages to fit context"
        )
        if self.memory_store:
            print()
            print(
                f"{self.ANSI_BOLD}Memory Commands{self.ANSI_RESET} (cloud PostgreSQL only)"
            )
            print(
                f"  {self.ANSI_CYAN}/remember <text>{self.ANSI_RESET}      Store a memory (supports type=, tag=, importance=, confidence=)"
            )
            print(
                f"  {self.ANSI_CYAN}/recall <query>{self.ANSI_RESET}       Search semantic memories"
            )
            print(
                f"  {self.ANSI_CYAN}/memories [tag]{self.ANSI_RESET}       List recent memories or by tag"
            )
            print(
                f"  {self.ANSI_CYAN}/forget <id>{self.ANSI_RESET}          Delete memory by ID"
            )
            print(
                f"  {self.ANSI_CYAN}/tags{self.ANSI_RESET}                 List memory tags"
            )
            print(
                f"  {self.ANSI_CYAN}/stats{self.ANSI_RESET}                Show memory statistics"
            )
        print()

    def cmd_quit(self) -> bool:
        """Save and quit. Returns True to signal exit."""
        save_chat_history(self.messages, self.context_file)
        print("Later!")
        return True

    def cmd_clear(self) -> None:
        """Clear conversation history and archive old messages."""
        archive_path = archive_chat_history(self.context_file)
        self.messages = [{"role": "system", "content": self.system_prompt}]
        save_chat_history(self.messages, self.context_file)
        if archive_path:
            print(f"📦 Previous conversation archived to {archive_path}")
        print("🗑️  Context cleared and saved!")

    def cmd_save(self) -> None:
        """Save current conversation history."""
        save_chat_history(self.messages, self.context_file)

    def cmd_load(self, files: Optional[List[str]] = None) -> None:
        """Load conversation history from file(s).

        Args:
            files: Optional list of file paths. If None, loads from default context_file.
        """
        if not files:
            loaded_messages = load_chat_history(self.context_file)
            if loaded_messages:
                self.messages = loaded_messages
                self.messages[0] = {"role": "system", "content": self.system_prompt}
                print(
                    f"{self.ANSI_GREEN}🔄 Context loaded from {self.context_file}{self.ANSI_RESET}"
                )
            else:
                self.messages = [{"role": "system", "content": self.system_prompt}]
                print(
                    f"{self.ANSI_YELLOW}⚠️  No saved context loaded from {self.context_file}{self.ANSI_RESET}"
                )
            return

        combined: List[Dict] = []
        any_loaded = False
        for f in files:
            loaded = load_chat_history(f)
            if loaded:
                combined.extend(loaded)
                any_loaded = True

        if any_loaded:
            self.messages = combined
            print(
                f"{self.ANSI_GREEN}🔄 Context loaded from: {' '.join(files)}{self.ANSI_RESET}"
            )
        else:
            print(
                f"{self.ANSI_YELLOW}⚠️  No saved context loaded from: {' '.join(files)}{self.ANSI_RESET}"
            )

    def cmd_trim(self) -> None:
        """Trim conversation to fit within token limits."""
        self.messages, was_trimmed = trim_context(
            self.messages, self.max_history_tokens
        )
        if was_trimmed:
            save_chat_history(self.messages, self.context_file)
            print(f"✂️  Context trimmed to {len(self.messages)} messages")
        else:
            print(
                f"✓ Context is within limits ({count_message_tokens(self.messages)} tokens)"
            )

    def cmd_context(self, show_full: bool = False) -> None:
        """Print current conversation context.

        Args:
            show_full: If True, show full message content; otherwise truncate long messages.
        """
        print("\n" + "=" * 60)
        print("📋 CURRENT CONTEXT")
        print("=" * 60)
        total_tokens = count_message_tokens(self.messages)
        print(
            f"Total messages: {len(self.messages)} | Estimated tokens: {total_tokens}"
        )
        print("=" * 60)
        for i, msg in enumerate(self.messages):
            role = msg.get("role", "").upper()
            content = msg.get("content", "")
            tokens = estimate_tokens(content)
            if not show_full and len(content) > 100:
                content = content[:100] + "..."
            print(f"\n[{i}] {role} (~{tokens} tokens):")
            print(f"  {content}")
        print("=" * 60 + "\n")

    def cmd_remember(self, args: str) -> None:
        """Store a memory.

        Args:
            args: Memory text with optional type=, tag=, importance=, confidence= parameters
        """
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        if not args:
            print("Usage: /remember [params] <text>")
            return

        memory_type = "fact"
        tags = []
        importance = 5
        confidence = 0.8

        type_match = re.search(r"type=(\w+)", args)
        if type_match:
            memory_type = type_match.group(1)
            args = re.sub(r"type=\w+\s*", "", args)

        tag_match = re.search(r"tag=(\w+)", args)
        if tag_match:
            tags.append(tag_match.group(1))
            args = re.sub(r"tag=\w+\s*", "", args)

        importance_match = re.search(r"importance=(\d+)", args)
        if importance_match:
            importance = int(importance_match.group(1))
            args = re.sub(r"importance=\d+\s*", "", args)

        confidence_match = re.search(r"confidence=([\d.]+)", args)
        if confidence_match:
            confidence = float(confidence_match.group(1))
            args = re.sub(r"confidence=[\d.]+\s*", "", args)

        text = args.strip()
        if not text:
            print("Usage: /remember [params] <text>")
            return

        try:
            mem_id = self.memory_store.remember(
                text,
                memory_type,
                context="chat",
                tags=tags,
                importance=importance,
                confidence=confidence,
            )
            if mem_id:
                print(f"✓ Memory stored with ID {mem_id}")
        except Exception as e:
            logger.error(f"Error storing memory: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def cmd_recall(self, query: str) -> None:
        """Search semantic memories.

        Args:
            query: Search query
        """
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        if not query:
            print("Usage: /recall <query>")
            return

        try:
            results = self.memory_store.recall(query, limit=5)
            if results:
                print(f"\n🔍 Found {len(results)} relevant memories:\n")
                for r in results:
                    print(
                        f"  [{r['id']}] {r['type']} | {r['tag']} - {r['memory_text']}"
                    )
            else:
                print("  No memories found")
        except Exception as e:
            logger.error(f"Error recalling memories: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def cmd_memories(self, tag: Optional[str] = None) -> None:
        """List recent memories or memories by tag.

        Args:
            tag: Optional tag filter
        """
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        try:
            if tag:
                results = self.memory_store.list_by_tag(tag, limit=20)
                print(f"\n📚 Memories tagged '{tag}':\n")
            else:
                results = self.memory_store.list_recent(limit=20)
                print("\n📚 Recent memories:\n")

            if results:
                for r in results:
                    print(
                        f"  [{r['id']}] {r['type']} | {r['tag']} - {r['memory_text']}"
                    )
            else:
                print("  No memories found")
        except Exception as e:
            logger.error(f"Error listing memories: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def cmd_forget(self, memory_id: str) -> None:
        """Delete a memory by ID.

        Args:
            memory_id: Memory ID to delete
        """
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        if not memory_id:
            print("Usage: /forget <id>")
            return

        try:
            self.memory_store.forget(memory_id)
            print(f"✓ Memory {memory_id} deleted")
        except Exception as e:
            logger.error(f"Error deleting memory: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def cmd_tags(self) -> None:
        """List all memory tags."""
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        try:
            tags = self.memory_store.list_tags()
            if tags:
                print("\n🏷️  Available tags:\n")
                for tag in tags:
                    print(f"  - {tag}")
            else:
                print("  No tags found")
        except Exception as e:
            logger.error(f"Error listing tags: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def cmd_stats(self) -> None:
        """Show memory statistics."""
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        try:
            stats = self.memory_store.get_stats()
            print("\n📊 Memory Statistics:\n")
            print(f"  Total memories: {stats.get('total_memories', 0)}")
            print(f"  Memory types: {stats.get('memory_types', {})}")
            print(f"  Total tags: {stats.get('total_tags', 0)}")
        except Exception as e:
            logger.error(f"Error getting stats: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def _handle_tool_calls(self, tool_calls: List) -> None:
        """Process tool calls from LLM response.

        Args:
            tool_calls: List of tool call objects from LLM
        """
        for call in tool_calls:
            func = call.function
            fname = func.name
            fargs = func.arguments

            if fname == "store_memory_tool" and self.memory_store:
                tool_func = create_store_memory_tool(self.memory_store)
                result = tool_func(**fargs)

                logger.info(f"Tool result: {result}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": fname,
                        "content": result,
                    }
                )
                print(f"🧠 {result}")

    def _send_message(self, user_input: str) -> None:
        """Send user message to LLM and handle response.

        Args:
            user_input: User's message text
        """
        current_tokens = count_message_tokens(self.messages)
        if current_tokens > self.max_history_tokens:
            self.messages, was_trimmed = trim_context(
                self.messages, self.max_history_tokens
            )
            if was_trimmed:
                print(
                    f"✂️  Auto-trimmed context to fit within {self.max_history_tokens} tokens"
                )
                save_chat_history(self.messages, self.context_file)

        self.messages.append({"role": "user", "content": user_input})

        print("\n🤖 Assistant: ", end="", flush=True)

        self._handle_response()

        current_tokens = count_message_tokens(self.messages)
        usage_pct = (current_tokens / self.max_history_tokens) * 100
        print(
            f"\n📊 Messages: {len(self.messages)} | Tokens: {current_tokens}/{self.max_history_tokens} ({usage_pct:.1f}%)"
        )

        if usage_pct > 90:
            print("⚠️  Context nearly full - will auto-trim on next message")

    def _handle_response(self) -> None:
        """Handle LLM streaming response."""
        stream = self.ollama_service.chat(
            self.messages, tools=self.tools or None, stream=True
        )

        full_response = ""
        thinking = ""
        tool_calls_list = []

        for chunk in stream:
            msg = chunk.message

            content = msg.content or ""
            chunk_thinking = msg.thinking or ""
            chunk_tools = msg.tool_calls or []

            if chunk_thinking:
                thinking += chunk_thinking

            if content:
                print(content, end="", flush=True)
                full_response += content

            if chunk_tools:
                tool_calls_list.extend(chunk_tools)

        print()

        assistant_msg = {"role": "assistant", "content": full_response}
        if thinking:
            assistant_msg["thinking"] = thinking
        if tool_calls_list:
            assistant_msg["tool_calls"] = tool_calls_list

        self.messages.append(assistant_msg)

        if tool_calls_list:
            self._handle_tool_calls(tool_calls_list)

    def run(self) -> None:
        """Run the interactive chat loop."""
        if not self.ollama_service.check_connection():
            return

        memory_path = Path(self.context_file)
        if memory_path.exists():
            print(
                f"{self.ANSI_YELLOW}⚠️  Found saved conversation at {self.context_file}.{self.ANSI_RESET} Use {self.ANSI_CYAN}/load{self.ANSI_RESET} to restore it."
            )

        print(f"{self.ANSI_GREEN}🤖 Ollama Chat{self.ANSI_RESET} — Model: {self.model}")
        print(f"Type {self.ANSI_CYAN}/?{self.ANSI_RESET} for commands")
        print()

        while True:
            try:
                user_input = input("\n💬 You: ").strip()

                if user_input.lower() in ["/bye", "/quit"]:
                    if self.cmd_quit():
                        break
                    continue

                if user_input == "/?":
                    self.cmd_help()
                    continue

                if user_input == "/context":
                    self.cmd_context(show_full=True)
                    continue

                if user_input == "/context brief":
                    self.cmd_context(show_full=False)
                    continue

                if user_input == "/clear":
                    self.cmd_clear()
                    continue

                if user_input == "/save":
                    self.cmd_save()
                    continue

                if user_input.startswith("/load"):
                    tokens = user_input.split()[1:]
                    self.cmd_load(files=tokens if tokens else None)
                    continue

                if user_input == "/trim":
                    self.cmd_trim()
                    continue

                if self.memory_store:
                    if user_input.startswith("/remember "):
                        args = user_input[10:].strip()
                        self.cmd_remember(args)
                        continue

                    if user_input.startswith("/recall "):
                        query = user_input[8:].strip()
                        self.cmd_recall(query)
                        continue

                    if user_input.startswith("/memories"):
                        parts = user_input.split(maxsplit=1)
                        tag = parts[1] if len(parts) > 1 else None
                        self.cmd_memories(tag)
                        continue

                    if user_input.startswith("/forget "):
                        memory_id = user_input[8:].strip()
                        self.cmd_forget(memory_id)
                        continue

                    if user_input == "/tags":
                        self.cmd_tags()
                        continue

                    if user_input == "/stats":
                        self.cmd_stats()
                        continue

                if not user_input:
                    continue

                self._send_message(user_input)

            except KeyboardInterrupt:
                print("\n\nSaving before exit...")
                save_chat_history(self.messages, self.context_file)
                if self.memory_store:
                    self.memory_store.close()
                print("Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"\n❌ Error: {e}\n")
