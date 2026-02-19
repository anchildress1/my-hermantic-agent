# User Guide

Quick guide to using Hermes Agent effectively.

> [!WARNING]
> **This Is Hermes, Not a Hall Monitor**
>
> Hermes ships without the usual corporate-grade guardrails. He's a hybrid reasoning model with tool access and an attitude, and he will absolutely follow your instructions even when you probably shouldn't have written them. Before you grab this code and run, go read the [model card](https://huggingface.co/NousResearch/Hermes-4-14B) to understand what Hermes actually is and what he is not.

## Getting Started

### First Run

1. Start the agent:

   ```bash
   make run
   ```

1. You'll see:

   ```bash
   🤖 Ollama Chat (Model: hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0)
   ✓ Semantic memory connected

   💬 You:
   ```

1. Just start typing! The agent remembers your conversation.

### Basic Chat

Simply type your message and press Enter:

```bash
💬 You: What's the weather like?
🤖 Assistant: I don't have access to real-time weather...
```

### Exiting

Type `/bye` or press `Ctrl+C`:

- Conversation context auto-saves to `data/memory.json`
- Next time you start, it continues where you left off

______________________________________________________________________

## Terminology

**CONTEXT** vs **MEMORY**:

- **CONTEXT**: Local JSON conversation history (`data/memory.json`) managed by `/save`, `/load`, `/clear`
- **MEMORY**: Cloud PostgreSQL semantic memory managed automatically from normal conversation

These are separate systems serving different purposes.

______________________________________________________________________

## Conversation Commands

### View Context

```bash
/context        # Show full conversation with token counts
/context brief  # Show abbreviated version
```

### Manage History

```bash
/clear   # Clear conversation context (keeps system prompt)
/save    # Manually save conversation context
/load    # Load saved context from JSON (defaults to data/memory.json)
/trim    # Manually trim old messages
```

> ✅ The previous conversation is archived to `data/memory-clear-<timestamp>.json` each time `/clear` runs.

## Memory Behavior

Semantic memory is written automatically to cloud PostgreSQL (separate from conversation context).

### How To Use It

```bash
💬 You: Remember that I prefer Python over JavaScript for backend work.

🤖 Assistant: ...
🧠 Auto-memory stored: 12
```

When you explicitly ask to remember something, that memory is automatically treated as high importance.

### Memory Types

- `preference` - Likes/dislikes ("prefers dark mode")
- `fact` - Information ("Python 3.12 released Oct 2023")
- `task` - Todos ("review PR #456")
- `insight` - Observations ("most productive in morning")

### Audit View

```bash
/audit              # Show recent memory audit events
```

______________________________________________________________________

## Tips & Tricks

### Organizing Memories

Use tags to organize:

```bash
work              # Work-related
personal          # Personal stuff
project-alpha     # Specific project
coding            # Coding preferences
```

### Token Management

Monitor token usage:

```bash
📊 Messages: 15 | Tokens: 4500/6144 (73.2%)
⚠️  Context nearly full - will auto-trim on next message
```

The agent automatically trims old messages when approaching limits.

### Confidence Scores

When storing memories, use confidence to indicate certainty:

- `1.0` - Certain, verified information
- `0.8` - Pretty sure
- `0.5` - Uncertain, might change

______________________________________________________________________

## Common Workflows

### Daily Standup

```bash
Remember that I completed PR review for #456.
```

### Project Context

```bash
Remember that project-alpha API endpoint is /api/v1/users.
```

### Learning & Notes

```bash
Remember that Python 3.12 adds better error messages.
```

______________________________________________________________________

## Troubleshooting

### "Semantic memory unavailable"

- Check `.env` has `MEMORY_DB_URL` and `OPENAI_API_KEY`
- Run `make setup-db` to initialize database
- Agent still works without semantic memory (conversation context only)

### "Context nearly full"

- Agent auto-trims old messages
- Use `/trim` to manually trim
- Increase `num_ctx` in `config/template.yaml` (requires more RAM)

### "Repetition detected"

- Check `logs/ollama_chat.log` for details
- Adjust `repeat_penalty` in `config/template.yaml`
- Try `/clear` to start fresh

### Slow Responses

- Larger models are slower
- Check Ollama is using GPU: `ollama ps`
- Reduce `num_predict` for shorter responses

______________________________________________________________________

## Advanced Usage

### Custom System Prompt

Edit `config/template.yaml`:

```yaml
system: |
  You are a helpful coding assistant specializing in Python.
  Always provide working code examples.
```

### Multiple Tags

Organize by project:

```bash
Remember that project-alpha uses PostgreSQL 15.

Remember that project-beta uses MongoDB 6.
```

### Bulk Memory Management

```python
# Python script to bulk import memories
from src.services.memory.vector_store import MemoryStore

store = MemoryStore()

facts = [
    ("Python is dynamically typed", "fact", "learning"),
    ("User prefers tabs", "preference", "coding"),
]

for text, type, context in facts:
    store.remember(text, type, context)
```

______________________________________________________________________

## Keyboard Shortcuts

- `Ctrl+C` - Save and exit
- `Ctrl+D` - Exit without saving (not recommended)
- `Up Arrow` - Previous command (terminal feature)

______________________________________________________________________

## Best Practices

1. **Use descriptive tags** - Makes searching easier
1. **Store atomic memories** - One fact per memory
1. **Regular cleanup** - Delete outdated tasks
1. **Check `/audit` regularly** - Monitor memory write/read behavior
1. **Backup** - Cloud database + local `data/memory.json` context

______________________________________________________________________

## Getting Help

- Check logs: `make logs`
- Run tests: `make test`
- See all commands: `make help`
- Read docs: `docs/` directory
