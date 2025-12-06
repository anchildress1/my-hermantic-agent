---
inclusion: always
---

# Technology Stack

## Build System & Package Management

- **Python**: 3.12+
- **Package Manager**: `uv` (modern Python package manager)
- **Dependency File**: `pyproject.toml`

## Core Dependencies

- `ollama>=0.4.0` - Local LLM inference
- `pyyaml>=6.0.1` - Configuration file parsing
- `psycopg2-binary>=2.9.9` - PostgreSQL database adapter
- `openai>=1.0.0` - Embeddings generation
- `python-dotenv>=1.0.0` - Environment variable management

## External Services

- **Ollama**: Local model inference (must be running separately)
- **TimescaleDB**: Vector database for semantic memory (Tiger Cloud)
- **OpenAI API**: Text embeddings via `text-embedding-3-small`

## Common Commands

### Setup

```bash
# Install dependencies
uv sync

# Setup environment
cp .env.example .env
# Edit .env to add OPENAI_API_KEY

# Pull Ollama model
ollama pull hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0
```

### Running

```bash
# Start chat interface
uv run python main.py

# Test memory store
uv run python tests/test_memory.py
```

### Development

```bash
# Ensure Ollama is running
ollama serve

# Check logs
tail -f logs/ollama_chat.log
```

## Configuration

- **Model Config**: `config/template.yaml` - model selection, system prompt, parameters
- **Environment**: `.env` - API keys and database URLs
- **Logs**: `logs/` directory (auto-created)
- **Data**: `data/` directory for conversation history (auto-created)
