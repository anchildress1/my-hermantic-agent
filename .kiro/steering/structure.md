---
inclusion: always
---

# Project Structure

## Directory Layout

```plaintext
hermes-agent/
├── config/              # Configuration files
│   └── template.yaml    # Model template and parameters
├── src/
│   └── agent/          # Core agent modules
│       ├── chat.py     # Chat interface and conversation management
│       └── memory.py   # Persistent memory with semantic search
├── tests/              # Test scripts
├── data/               # Local conversation history (gitignored)
├── logs/               # Application logs (gitignored)
├── main.py             # Entry point
├── .env                # Environment variables (gitignored)
└── pyproject.toml      # Project dependencies
```

## Module Organization

### `main.py`

Entry point that:

- Loads environment variables
- Sets up logging directory
- Loads template configuration
- Starts chat loop

### `src/agent/chat.py`

Chat interface module handling:

- Ollama model interaction
- Conversation history management
- Context trimming and token counting
- Memory persistence (JSON file)
- Streaming responses
- Repetition detection
- Interactive commands (`/context`, `/clear`, `/save`, etc.)

### `src/agent/memory.py`

Semantic memory store with:

- TimescaleDB connection management
- OpenAI embedding generation
- Vector similarity search
- Memory CRUD operations (remember, recall, forget)
- Context and type filtering
- Access tracking and statistics

## Key Patterns

### Configuration

- YAML for model templates (`config/template.yaml`)
- Environment variables for secrets (`.env`)
- Defaults with overrides pattern

### Memory Management

- Dual-layer: JSON for conversations, TimescaleDB for semantic memory
- Auto-save on exit and manual save command
- Context trimming keeps system prompt + recent messages
- Token estimation for context window management

### Error Handling

- Graceful KeyboardInterrupt handling with auto-save
- Try-catch around chat loop for resilience
- Logging for debugging repetition issues

### Code Style

- Type hints on function signatures
- Docstrings for public functions
- Snake_case naming convention
- 4-space indentation
- Pathlib for file operations
