# Quick Start Guide

Get up and running with Hermes Agent in 5 minutes.

______________________________________________________________________

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [Ollama](https://ollama.ai) installed and running
- OpenAI API key (for embeddings)
- TimescaleDB instance (optional, for semantic memory)

______________________________________________________________________

## 1. Install Dependencies

```bash
uv sync
```

______________________________________________________________________

## 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your keys
# Required:
OPENAI_API_KEY=sk-your-key-here

# Optional (for semantic memory):
MEMORY_DB_URL=postgresql://user:pass@host:port/db?sslmode=require
```

______________________________________________________________________

## 3. Setup Ollama Model

> [!WARNING]
> **This Is Hermes, Not a Hall Monitor**
>
> Hermes ships without the usual corporate-grade guardrails. He's a hybrid reasoning model with tool access and an attitude, and he will absolutely follow your instructions even when you probably shouldn't have written them. Before you grab this code and run, go read the [model card](https://huggingface.co/NousResearch/Hermes-4-14B) to understand what Hermes actually is and what he is not.

```bash
# Start Ollama (if not running)
ollama serve

# Pull the model (in another terminal)
ollama pull hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0

# Or use any other model and update config/template.yaml
```

**Model Card:** https://huggingface.co/NousResearch/Hermes-4-14B

______________________________________________________________________

## 4. Setup Database (Optional)

If you want semantic memory features:

```bash
# Initialize the database schema
make setup-db

# Verify it works
uv run python -c "from src.agent.memory import MemoryStore; print(MemoryStore().stats())"
```

______________________________________________________________________

## 5. Start Chatting

```bash
make run
```

______________________________________________________________________

## Basic Commands

### Conversation

- Type normally to chat
- `quit` or `exit` - Save and exit
- `/clear` - Clear conversation history
- `/save` - Save conversation manually
- `/context` - Show conversation context

### Memory (if database configured)

- `/remember <text>` - Store a memory
- `/recall <query>` - Search memories
- `/stats` - Show memory statistics
- `/contexts` - List all contexts

______________________________________________________________________

## Example Session

```bash
🤖 Ollama Chat (Model: hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0)
✓ Semantic memory connected

💬 You: I prefer Python over JavaScript for backend work

🤖 Assistant: Got it. Python for backend - solid choice...

💬 You: /remember I prefer Python over JavaScript for backend work
Memory type (preference/fact/task/insight):
  Type: preference
Context (e.g., work, personal, project-name):
  Context: coding
✓ Memory stored with ID 1

💬 You: /recall programming language preferences

🔍 Found 1 relevant memories:

  [1] PREFERENCE | coding
      I prefer Python over JavaScript for backend work
      Similarity: 0.923 | Accessed: 1 times

💬 You: quit
💾 Memory saved to data/memory.json
Goodbye!
```

______________________________________________________________________

## Troubleshooting

### "Ollama service not running"

```bash
ollama serve
```

### "Model not found"

```bash
ollama pull hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0
```

### "OPENAI_API_KEY not set"

Add your key to `.env`:

```plaintext
OPENAI_API_KEY=sk-...
```

### "Semantic memory unavailable"

Either:

1. Add `MEMORY_DB_URL` to `.env` and run `make setup-db`
1. Or continue without semantic memory (conversation history still works)

______________________________________________________________________

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [docs/MEMORY_API.md](docs/MEMORY_API.md) for API reference
- Customize `config/template.yaml` for different models/prompts
- Run tests: `make test`

______________________________________________________________________

## Tips

1. **Conversation history** is auto-saved to `data/memory.json`
1. **Semantic memories** are stored in TimescaleDB for long-term recall
1. **Logs** are in `logs/ollama_chat.log` for debugging
1. **Context trimming** happens automatically when approaching token limits
1. **Backups** are created automatically when saving conversations

______________________________________________________________________

## Getting Help

- Check logs: `make logs`
- Run tests: `make test`
- See all commands: `make help`
- Read the analysis: [ANALYSIS.md](ANALYSIS.md)
- Implementation details: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
