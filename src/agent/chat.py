import yaml
import ollama
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logging():
    """Setup logging for repetition detection."""
    logging.basicConfig(
        filename='logs/ollama_chat.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_template(template_path: str = "config/template.yaml") -> dict:
    """Load model template configuration from YAML file."""
    with open(template_path, "r") as f:
        return yaml.safe_load(f)


def save_memory(messages: list, memory_file: str = "data/memory.json"):
    """Save conversation history to file."""
    memory_path = Path(memory_file)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }
    with open(memory_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Memory saved to {memory_file}")


def load_memory(memory_file: str = "data/memory.json") -> list:
    """Load conversation history from file."""
    memory_path = Path(memory_file)
    if not memory_path.exists():
        return []
    
    with open(memory_path, "r") as f:
        data = json.load(f)
    
    messages = data.get("messages", [])
    timestamp = data.get("timestamp", "unknown")
    print(f"📂 Loaded memory from {timestamp}")
    return messages


def estimate_tokens(text: str) -> int:
    """Rough token estimation (1 token ≈ 4 chars)."""
    return len(text) // 4


def count_message_tokens(messages: list) -> int:
    """Estimate total tokens in message history."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.get('content', ''))
    return total


def trim_context(messages: list, max_tokens: int = 6000, keep_recent: int = 10) -> tuple[list, bool]:
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
    system_msg = messages[0] if messages[0]['role'] == 'system' else None
    recent_messages = messages[-keep_recent:]
    
    # Calculate tokens for what we're keeping
    kept_tokens = count_message_tokens(recent_messages)
    if system_msg:
        kept_tokens += estimate_tokens(system_msg['content'])
    
    # Build trimmed context
    trimmed = []
    if system_msg:
        trimmed.append(system_msg)
    
    # Add a summary message about what was trimmed
    num_trimmed = len(messages) - len(recent_messages) - (1 if system_msg else 0)
    if num_trimmed > 0:
        summary = {
            'role': 'system',
            'content': f'[Context trimmed: {num_trimmed} older messages removed to stay within token limit]'
        }
        trimmed.append(summary)
    
    trimmed.extend(recent_messages)
    
    logging.info(f"Context trimmed: {len(messages)} -> {len(trimmed)} messages, {total_tokens} -> {kept_tokens} tokens")
    
    return trimmed, True


def print_context(messages: list, show_full: bool = False):
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


def chat_loop(template: dict, memory_file: str = "data/memory.json"):
    """Run interactive chat loop with Ollama."""
    model = template.get('model', 'llama3.2')
    system_prompt = template.get('system', '')
    params = template.get('parameters', {})
    
    # Get context window size from params, default to 8192
    max_context = params.get('num_ctx', 8192)
    # Keep 75% for history, 25% for generation
    max_history_tokens = int(max_context * 0.75)
    
    # Try to load existing memory
    messages = load_memory(memory_file)
    
    # If no memory or system prompt changed, reinitialize
    if not messages or (messages and messages[0].get('role') != 'system'):
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
    elif system_prompt and messages[0]['content'] != system_prompt:
        # Update system prompt if it changed
        messages[0] = {'role': 'system', 'content': system_prompt}
    
    print(f"🤖 Ollama Chat (Model: {model})")
    print(f"📊 Context window: {max_context} tokens (keeping {max_history_tokens} for history)")
    print(f"📊 Parameters: {json.dumps(params, indent=2)}")
    print("\nCommands:")
    print("  'quit' or 'exit' - End conversation and save")
    print("  '/context' - Show full conversation context with token counts")
    print("  '/context brief' - Show brief context summary")
    print("  '/clear' - Clear conversation history (keeps system prompt)")
    print("  '/save' - Manually save current conversation")
    print("  '/load' - Reload from saved memory")
    print("  '/stream' - Toggle streaming mode")
    print("  '/trim' - Manually trim old messages\n")
    
    streaming = True
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                save_memory(messages, memory_file)
                print("Goodbye!")
                break
            
            if user_input == '/context':
                print_context(messages, show_full=True)
                continue
            
            if user_input == '/context brief':
                print_context(messages, show_full=False)
                continue
            
            if user_input == '/clear':
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                save_memory(messages, memory_file)
                print("🗑️  Context cleared and saved!")
                continue
            
            if user_input == '/save':
                save_memory(messages, memory_file)
                continue
            
            if user_input == '/load':
                messages = load_memory(memory_file)
                if not messages and system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
                continue
            
            if user_input == '/stream':
                streaming = not streaming
                print(f"🔄 Streaming {'enabled' if streaming else 'disabled'}")
                continue
            
            if user_input == '/trim':
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    save_memory(messages, memory_file)
                    print(f"✂️  Context trimmed to {len(messages)} messages")
                else:
                    print(f"✓ Context is within limits ({count_message_tokens(messages)} tokens)")
                continue
            
            if not user_input:
                continue
            
            # Check if we need to trim context before adding new message
            current_tokens = count_message_tokens(messages)
            if current_tokens > max_history_tokens:
                messages, was_trimmed = trim_context(messages, max_history_tokens)
                if was_trimmed:
                    print(f"✂️  Auto-trimmed context to fit within {max_history_tokens} tokens")
                    save_memory(messages, memory_file)
            
            # Add user message to history
            messages.append({'role': 'user', 'content': user_input})
            
            print("\n🤖 Assistant: ", end="", flush=True)
            
            # Get response from Ollama
            if streaming:
                full_response = ""
                last_100_chars = ""
                repetition_warnings = []
                repetition_count = 0
                
                stream = ollama.chat(
                    model=model,
                    messages=messages,
                    options=params,
                    stream=True
                )
                
                for chunk in stream:
                    content = chunk['message']['content']
                    
                    # Simple repetition detection
                    if len(content) > 10 and content in last_100_chars:
                        repetition_count += 1
                        if repetition_count == 3:  # First time hitting threshold
                            warning_msg = f"Repetition detected at position {len(full_response)}"
                            repetition_warnings.append(warning_msg)
                            logging.warning(f"{warning_msg} | Content: '{content}' | Last 100: '{last_100_chars[-50:]}'")
                    else:
                        repetition_count = 0
                    
                    print(content, end="", flush=True)
                    full_response += content
                    last_100_chars = (last_100_chars + content)[-100:]
                
                print()  # newline after stream
                
                # Show warnings if any repetition detected
                if repetition_warnings:
                    print(f"\n⚠️  Repetition detected: {len(repetition_warnings)} instance(s) - check logs/ollama_chat.log")
                    logging.info(f"Full response with repetition:\n{full_response}\n")
                
                messages.append({'role': 'assistant', 'content': full_response})
            else:
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options=params
                )
                assistant_message = response['message']['content']
                messages.append({'role': 'assistant', 'content': assistant_message})
                print(assistant_message)
            
            # Show token count and context status
            current_tokens = count_message_tokens(messages)
            usage_pct = (current_tokens / max_history_tokens) * 100
            print(f"\n📊 Messages: {len(messages)} | Tokens: {current_tokens}/{max_history_tokens} ({usage_pct:.1f}%)")
            
            if usage_pct > 90:
                print("⚠️  Context nearly full - will auto-trim on next message")
            
        except KeyboardInterrupt:
            print("\n\nSaving before exit...")
            save_memory(messages, memory_file)
            print("Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
