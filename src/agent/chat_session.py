"""Chat session management with command handling."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config import AgentConfig
from src.core.utils import count_message_tokens, estimate_tokens, trim_context
from src.services.llm.ollama_service import OllamaService
from src.services.memory.auto_writer import AutoMemoryWriter
from src.services.memory.vector_store import MemoryStore
from src.services.memory.file_storage import (
    load_chat_history,
    save_chat_history,
    archive_chat_history,
)
from src.tools.memory_tool import create_store_memory_tool
from src.tools.tool_utils import format_tools_xml, parse_tool_calls

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
        auto_memory_writer: Optional[AutoMemoryWriter] = None,
    ):
        """Initialize chat session.

        Args:
            config: Chat configuration object
            context_file: Path to save/load conversation history
            llm_service: Injected LLM service instance
            memory_store: Optional vector memory store for semantic memory operations
            auto_memory_writer: Optional policy-guided auto-memory writer
        """
        self.config = config
        self.context_file = context_file
        self.memory_store = memory_store
        self.auto_memory_writer = auto_memory_writer
        self.ollama_service = llm_service

        self.model = config.model
        self.system_prompt = config.system
        self.params = config.parameters

        self.max_context = self.params.num_ctx
        self.max_history_tokens = int(self.max_context * 0.75)

        if self.memory_store:
            self.system_prompt = (
                f"{self.system_prompt}\n\n"
                "Memory policy:\n"
                "- Memory is automatic. Do not ask the user for slash commands.\n"
                "- If the user explicitly asks to remember something, call store_memory_tool.\n"
                "- Explicit remember intent should use high importance (>=2.0).\n"
                "- Keep memories atomic and durable."
            )

        self.messages: List[Dict] = [{"role": "system", "content": self.system_prompt}]

        self.tools = []
        if self.memory_store:
            tool_func = create_store_memory_tool(self.memory_store)
            self.tools.append(tool_func)
            logger.info("Memory tool enabled")

        # Support manual tool injection via XML (Hermes style)
        self.use_xml_tools = self.config.parameters.use_xml_tools
        if self.use_xml_tools and self.tools:
            tools_xml = format_tools_xml(self.tools)
            if self.system_prompt:
                self.messages[0]["content"] = f"{self.system_prompt}\n\n{tools_xml}"
            else:
                self.messages[0]["content"] = tools_xml
            logger.info("Manual XML tool wiring enabled")

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
            print(f"{self.ANSI_BOLD}Memory{self.ANSI_RESET} (automatic)")
            print(
                "  Ask naturally, for example: "
                "'remember that I prefer Python for backend work'."
            )
            print(
                f"  {self.ANSI_CYAN}/audit [operation]{self.ANSI_RESET}    Show recent memory audit events (operator view)"
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

    def cmd_audit(self, operation: Optional[str] = None) -> None:
        """Show recent memory audit events.

        Args:
            operation: Optional operation filter (remember, recall, forget, auto_remember)
        """
        if not self.memory_store:
            print("❌ Memory store not available")
            return

        try:
            events = self.memory_store.list_events(limit=20, operation=operation)
            if not events:
                print("No memory events found")
                return

            print("\n🧾 Memory Events:\n")
            for event in events:
                details = event.get("details") or {}
                memory_id = event.get("memory_id")
                print(
                    f"  [{event.get('id')}] {event.get('created_at')} | "
                    f"{event.get('operation')} | {event.get('status')} | memory_id={memory_id}"
                )
                if details:
                    summary = str(details)
                    print(f"      details: {summary[:200]}")
        except Exception as e:
            logger.error(f"Error getting audit events: {e}", exc_info=True)
            print(f"❌ Error: {e}")

    def _handle_tool_calls(self, tool_calls: List) -> bool:
        """Process tool calls from LLM response.

        Args:
            tool_calls: List of tool call objects from LLM

        Returns:
            True if store_memory_tool was executed.
        """
        memory_tool_called = False
        for call in tool_calls:
            func = call.function
            fname = func.name
            fargs = func.arguments

            if fname == "store_memory_tool" and self.memory_store:
                tool_func = create_store_memory_tool(self.memory_store)
                result = tool_func(**fargs)
                memory_tool_called = True

                logger.info(f"Tool result: {result}")
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_name": fname,
                        "content": result,
                    }
                )
                print(f"🧠 {result}")
        return memory_tool_called

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

        assistant_text, memory_tool_called = self._handle_response()

        if (
            self.auto_memory_writer
            and assistant_text.strip()
            and not memory_tool_called
        ):
            try:
                self.auto_memory_writer.process_turn(
                    user_message=user_input,
                    assistant_message=assistant_text,
                )
                auto_result = self.auto_memory_writer.last_result
                if auto_result.inserted_ids:
                    print(
                        f"🧠 Auto-memory stored: {', '.join(str(i) for i in auto_result.inserted_ids)}"
                    )
                if auto_result.revived_ids:
                    print(
                        f"🧠 Auto-memory refreshed: {', '.join(str(i) for i in auto_result.revived_ids)}"
                    )
                for failure in auto_result.failures:
                    print(
                        f"⚠️ Auto-memory failed for '{failure.memory_text}': {failure.error}"
                    )
            except Exception as e:
                logger.error(f"Auto-memory write failed: {e}", exc_info=True)

        current_tokens = count_message_tokens(self.messages)
        usage_pct = (current_tokens / self.max_history_tokens) * 100
        print(
            f"\n📊 Messages: {len(self.messages)} | Tokens: {current_tokens}/{self.max_history_tokens} ({usage_pct:.1f}%)"
        )

        if usage_pct > 90:
            print("⚠️  Context nearly full - will auto-trim on next message")

        # Persist each turn so abrupt termination loses less context.
        save_chat_history(self.messages, self.context_file)

    def _handle_response(self) -> tuple[str, bool]:
        """Handle LLM streaming response."""
        # If using XML tools, we don't pass 'tools' to Ollama API to avoid double-handling
        # or confusing models that perform better with manual XML instructions.
        ollama_tools = None if self.use_xml_tools else (self.tools or None)

        stream = self.ollama_service.chat(
            self.messages, tools=ollama_tools, stream=True
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

        memory_tool_called = False

        # Check for native tool calls
        if tool_calls_list:
            memory_tool_called = self._handle_tool_calls(tool_calls_list)

        # Check for manual XML tool calls if enabled
        if self.use_xml_tools:
            xml_tool_calls = parse_tool_calls(full_response)
            if xml_tool_calls:
                xml_memory_tool_called = self._handle_xml_tool_calls(xml_tool_calls)
                memory_tool_called = memory_tool_called or xml_memory_tool_called
        return full_response, memory_tool_called

    def _handle_xml_tool_calls(self, tool_calls: List[Dict]) -> bool:
        """Process manual XML tool calls.

        Args:
            tool_calls: List of parsed tool calls {"name":..., "arguments":...}

        Returns:
            True if store_memory_tool was executed.
        """
        tool_map = {t.__name__: t for t in self.tools}
        memory_tool_called = False

        for call in tool_calls:
            fname = call.get("name")
            fargs = call.get("arguments", {})

            if fname in tool_map:
                try:
                    result = tool_map[fname](**fargs)
                    if fname == "store_memory_tool":
                        memory_tool_called = True
                    logger.info(f"XML Tool {fname} executed: {result}")

                    # Store result in a format the model expects (XML response)
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"<tool_response>{result}</tool_response>",
                        }
                    )
                    print(f"🧠 Tool Output: {result}")
                except Exception as e:
                    logger.error(f"Error executing XML tool {fname}: {e}")
                    self.messages.append(
                        {
                            "role": "user",
                            "content": f"<tool_response>Error: {e}</tool_response>",
                        }
                    )
            else:
                logger.warning(f"XML Tool {fname} not found in available tools")

        # After tool execution, continue the conversation automatically
        print("\nAssistant (continuing): ", end="", flush=True)
        self._handle_response()
        return memory_tool_called

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

        try:
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()

                    if user_input.lower() in ["/bye", "/quit", "bye", "quit", "exit"]:
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
                        if user_input.startswith("/audit"):
                            parts = user_input.split(maxsplit=1)
                            operation = parts[1].strip() if len(parts) > 1 else None
                            self.cmd_audit(operation=operation)
                            continue

                    if not user_input:
                        continue

                    self._send_message(user_input)

                except KeyboardInterrupt:
                    print("\n\nSaving before exit...")
                    save_chat_history(self.messages, self.context_file)
                    print("Goodbye!")
                    break
                except Exception as e:
                    logger.error(f"Error in chat loop: {e}", exc_info=True)
                    print(f"\n❌ Error: {e}\n")
        finally:
            if self.memory_store:
                self.memory_store.close()
