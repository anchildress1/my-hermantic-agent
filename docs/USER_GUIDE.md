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

Type `quit`, `exit`, or press `Ctrl+C`:

- Conversation auto-saves to `data/memory.json`
- Next time you start, it continues where you left off

______________________________________________________________________

## Conversation Commands

### View Context

```bash
/context        # Show full conversation with token counts
/context brief  # Show abbreviated version
```

### Manage History

```bash
/clear   # Clear conversation (keeps system prompt)
/save    # Manually save conversation
/load    # Load most recently saved context or file specified
/load    # Reload from saved file
/trim    # Manually trim old messages
```

> ✅ The previous conversation is archived to `data/memory-clear-<timestamp>.json` each time `/clear` runs, so you can inspect or restore the cleared context later if needed.

### Toggle Features

```bash
/stream  # Toggle streaming mode on/off
```

______________________________________________________________________

## Memory Commands

```bash
💬 You: /remember I prefer Python over JavaScript

Memory type (preference/fact/task/insight):
  Type: preference

Context (e.g., work, personal, project-name):
  Context: coding

✓ Memory stored with ID 1
```

**Memory Types:**

- `preference` - Likes/dislikes ("prefers dark mode")
- `fact` - Information ("Python 3.12 released Oct 2023")
- `task` - Todos ("review PR #456")
- `insight` - Observations ("most productive in morning")

### Search Memories

```bash
💬 You: /recall programming preferences

🔍 Found 1 relevant memories:

  [1] PREFERENCE | coding
      I prefer Python over JavaScript
      Similarity: 0.923 | Accessed: 1 times
```

### List Memories

```bash
/memories           # Show statistics
/memories work      # Show memories in 'work' context
/contexts           # List all contexts
```

### Delete Memory

```bash
/forget 1           # Delete memory with ID 1
```

### View Statistics

```bash
/stats

📊 Memory Statistics:
  Total memories: 42
  Unique types: 4
  Unique contexts: 3
  Avg confidence: 0.95
  Last memory: 2025-12-06T13:27:20Z
```

______________________________________________________________________

## Tips & Tricks

### Organizing Memories

Use contexts to organize:

```bash
work              # Work-related
personal          # Personal stuff
project-alpha     # Specific project
coding            # Coding preferences
```

### Context Patterns

Use wildcards in searches:

```bash
/memories project-%    # All project memories
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
/recall today's tasks
# Review what needs to be done

/remember Completed PR review for #456
Type: task
Context: work
```

### Project Context

```bash
/memories project-alpha
# See all project-related memories

/remember API endpoint is /api/v1/users
Type: fact
Context: project-alpha
```

### Learning & Notes

```bash
/remember Python 3.12 adds better error messages
Type: fact
Context: learning

/recall Python features
# Later, search what you learned
```

______________________________________________________________________

## Troubleshooting

### "Semantic memory unavailable"

- Check `.env` has `MEMORY_DB_URL`
- Run `make setup-db` to initialize database
- Agent still works without semantic memory (conversation history only)

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

### Multiple Contexts

Organize by project:

```bash
/remember Uses PostgreSQL 15
Context: project-alpha

/remember Uses MongoDB 6
Context: project-beta
```

### Bulk Memory Management

```python
# Python script to bulk import memories
from src.agent.memory import MemoryStore

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

1. **Use descriptive contexts** - Makes searching easier
1. **Store atomic memories** - One fact per memory
1. **Regular cleanup** - Delete outdated tasks
1. **Check stats** - Monitor memory growth
1. **Backup** - `data/memory.json` and database

______________________________________________________________________

## Getting Help

- Check logs: `make logs`
- Run tests: `make test`
- See all commands: `make help`
- Read docs: `docs/` directory
