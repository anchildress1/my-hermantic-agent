# My Hermantic Agent

> [!INFO]
>
> 🦄 I picked Hermes because he’s a hybrid reasoning model who can actually think, use tools, and throw shade right back when I deserve it. Most models crumble when I push them; Hermes leans in. So this repo is my no-plan, see-what-happens agent playground. Under the hood it’s a CLI-based conversational system built on NousResearch's Hermes-4-14B hosted locally with Ollama. I've started with persistent semantic memory backed by Tiger's TimescaleDB and OpenAI embeddings. It runs a dual-memory setup for short-term chat context and long-term recall, manages its own state without whining, and communicates the same way I do: direct, no BS. I’m basically giving a capable model a sandbox and too much freedom, and seeing what grows teeth. 🧛‍♂️

---

## Quick Start

```bash
# Install dependencies
make install

# Setup environment
make setup
# Edit .env with your OPENAI_API_KEY

# Initialize database (optional, for semantic memory)
make setup-db

# Start chatting
make run
```

**Full setup guide:** [QUICKSTART.md](QUICKSTART.md)

---

## Features

- 🤖 **Local LLM** - Runs Ollama models locally, no cloud dependency
- 💾 **Dual Memory** - Short-term conversation history + long-term semantic memory
- 🔍 **Semantic Search** - Find relevant memories by meaning, not just keywords
- 🎯 **Smart Context** - Auto-trims conversations to stay within token limits
- 📝 **Persistent** - Conversations auto-save and resume where you left off
- ⚡ **Fast** - Connection pooling, embedding caching, optimized queries
- 🛡️ **Robust** - Comprehensive error handling, atomic file writes, graceful degradation

---

## Architecture

```mermaid
%%{init: {'theme':'base'}}%%
graph TB
    CLI[CLI Interface] --> Chat[Chat Manager]
    Chat --> Ollama[Ollama LLM]
    Chat --> Memory[Memory Store]
    Chat --> JSON[Conversation JSON]
    Memory --> OpenAI[OpenAI Embeddings]
    Memory --> TimescaleDB[(TimescaleDB + pgvector)]
    
    style CLI fill:#4A90E2
    style Ollama fill:#FF6B6B
    style OpenAI fill:#4ECDC4
    style TimescaleDB fill:#95E1D3
```

**Detailed diagrams:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Documentation

- 📖 **[User Guide](docs/USER_GUIDE.md)** - How to use the agent effectively
- 🏗️ **[Architecture](docs/ARCHITECTURE.md)** - System design and data flow
- 🎛️ **[Model Parameters](docs/MODEL_PARAMETERS.md)** - Hermes-4 configuration guide
- 🔧 **[Memory API](docs/MEMORY_API.md)** - Semantic memory reference
- 🚀 **[Quick Start](QUICKSTART.md)** - 5-minute setup guide

---

## Usage

### Basic Chat

```bash
make run
```

```bash
💬 You: What's the capital of France?
🤖 Assistant: Paris.

💬 You: quit
💾 Memory saved to data/memory.json
Goodbye!
```

### Memory Commands

```bash
# Store a memory
/remember I prefer Python over JavaScript
Type: preference
Context: coding
✓ Memory stored with ID 1

# Search memories
/recall programming preferences
🔍 Found 1 relevant memories:
  [1] PREFERENCE | coding
      I prefer Python over JavaScript
      Similarity: 0.923

# View statistics
/stats
📊 Total memories: 42 | Contexts: 3 | Types: 4
```

**Full command reference:** [docs/USER_GUIDE.md](docs/USER_GUIDE.md)

---

## Requirements

- **Python 3.12+**
- **Hugging Face** - Any model with Ollama support ([huggingface.com](https://huggingface.co/))
- **Ollama** - Local LLM runtime ([ollama.ai](https://ollama.ai))
- **OpenAI API Key** - For embeddings ([platform.openai.com](https://platform.openai.com/api-keys))
- **TimescaleDB** - Optional, for semantic memory ([timescale.cloud](https://console.timescale.cloud))

---

## Installation

### 1. Install Dependencies

```bash
make install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add:
# - OPENAI_API_KEY (required)
# - MEMORY_DB_URL (optional, for semantic memory)
```

### 3. Setup Ollama

> [!WARN] **This Is Hermes, Not a Hall Monitor**
> 
> ⚠️ Hermes ships without the usual corporate-grade guardrails, seatbelts, bumpers, or soft edges. He’s a hybrid reasoning model with tool access and an attitude, and he will absolutely follow your instructions even when you probably shouldn’t have written them. Before you grab this code and run, go read the docs on what Hermes actually is and what he is not. If you treat him like a safe, shrink-wrapped assistant, that’s on you. This project is an experiment, not a babysitter.

```bash
# Pull the Hermes-4 model
ollama pull hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0

# Start Ollama service
ollama serve
```

### 4. Initialize Database (Optional)

```bash
make setup-db
```

---

## Project Structure

```plaintext
hermes-agent/
├── config/
│   └── template.yaml      # Model configuration
├── src/agent/
│   ├── chat.py            # Chat interface
│   └── memory.py          # Semantic memory
├── docs/                  # Documentation
├── schema/                # Database schema
├── tests/                 # Unit tests
├── main.py                # Entry point
└── .env                   # Environment variables
```

---

## Configuration

### Model Settings

Edit `config/template.yaml` to customize:

```yaml
model: hf.co/DevQuasar/NousResearch.Hermes-4-14B-GGUF:Q8_0
system: |
  You are Hermes, a personal assistant...
parameters:
  temperature: 0.85
  num_ctx: 8192
  # ... more parameters
```

**Parameter guide:** [docs/MODEL_PARAMETERS.md](docs/MODEL_PARAMETERS.md)

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Optional
MEMORY_DB_URL=postgresql://...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_EMBEDDING_DIM=1536
```

---

## Development

```bash
# Run tests
make test

# View logs
make logs

# Clean artifacts
make clean

# See all commands
make help
```

---

## Memory System

### Dual-Memory Architecture

1. **Short-term** - Full conversation history in `data/memory.json`
   - Auto-saves on exit
   - Auto-loads on startup
   - Smart context trimming

2. **Long-term** - Semantic memories in TimescaleDB
   - Vector embeddings for similarity search
   - Organized by type and context
   - Persistent across conversations

### Memory Types

- **preference** - User likes/dislikes
- **fact** - Factual information
- **task** - Todos and action items
- **insight** - Observations and patterns

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white) ![TimescaleDB](https://img.shields.io/badge/TimescaleDB-FDB515?style=for-the-badge&logo=timescale&logoColor=black) 

![Hermes](https://img.shields.io/badge/Hermes-14B%20Hybrid%20Reasoner-0B3D91?style=for-the-badge&logo=lightning&logoColor=white) ![Hugging Face Badge](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge) ![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white) ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

![Kiro](https://img.shields.io/badge/Built_with-Kiro-7C3AED?style=for-the-badge)
![Verdant](https://img.shields.io/badge/Powered_by-Verdant-00D486?style=for-the-badge)
![ChatGPT](https://img.shields.io/badge/Assisted_by-ChatGPT-74AA9C?style=for-the-badge&logo=openai&logoColor=white)

</div>

- **Runtime**: Python 3.12+ with [uv package manager](https://docs.astral.sh/uv/)
- **Model**: [NousResearch/Hermes-4-14B](https://huggingface.co/NousResearch/Hermes-4-14B)  
- **Ollama**: [ollama.ai](https://ollama.ai)  
- **TigerData Agentic Postgres**: [tigerdata.com](https://www.tigerdata.com)
- **Embeddings**: OpenAI API (text-embedding-3-small)
- **Storage**: JSON for conversations, PostgreSQL for semantic memory

---

## License

This project is released under the [Polyform Shield License 1.0.0](https://polyformproject.org/licenses/shield/1.0.0/). In plain language: use it, study it, fork it, remix it, build weird things with it — just don’t make money from it or wrap it into anything commercial without getting my permission first. No loopholes, no “but technically,” no marketplace shenanigans. The full legal text lives in the [LICENSE](LICENSE) file if you need the exact wording. 📜🛡️
