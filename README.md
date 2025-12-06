# Ollama Agent

A CLI agent for exploring Ollama model templates with persistent memory storage.

## Project Structure

```plaintext
ollama-test/
├── config/              # Configuration files
│   └── template.yaml    # Model template and parameters
├── src/
│   └── agent/
│       ├── chat.py      # Chat interface and conversation management
│       └── memory.py    # Persistent memory with semantic search
├── tests/               # Test scripts
├── data/                # Local conversation history (gitignored)
├── logs/                # Application logs (gitignored)
├── main.py              # Entry point
└── .env                 # Environment variables (gitignored)
```

## Setup

1. Install dependencies:

```bash
uv sync
```

1. Create `.env` file:

```bash
cp .env.example .env
# Add your OPENAI_API_KEY
```

1. Make sure Ollama is running and pull a model:

```bash
ollama pull hermes3:14b
```

## Usage

Run the chat interface:

```bash
uv run python main.py
```

### Chat Commands

- `quit` or `exit` - End conversation and save
- `/context` - Show full conversation context with token counts
- `/context brief` - Show brief context summary
- `/clear` - Clear conversation history (keeps system prompt)
- `/save` - Manually save current conversation
- `/load` - Reload from saved memory
- `/stream` - Toggle streaming mode
- `/trim` - Manually trim old messages

### Memory Features

- Conversations auto-save to `data/memory.json`
- Auto-loads on startup - continues where you left off
- Smart context trimming when approaching token limits
- Repetition detection with logging

## Memory Store

The agent uses TimescaleDB for persistent semantic memory:

- **Distilled memories** - Stores key facts, not full conversations
- **Semantic search** - Find relevant memories by meaning
- **Context filtering** - Organize by project, work, personal, etc.
- **Type filtering** - preference, fact, task, insight

### Test Memory Store

```bash
uv run python tests/test_memory.py
```

## Template Configuration

Edit `config/template.yaml` to customize:

- **model**: Which Ollama model to use
- **system**: System prompt that sets the assistant's behavior
- **parameters**: Model parameters (see below)

## Parameters

### temperature: 0.8

- Controls randomness/creativity (0.0 to 2.0)
- Lower = more focused and deterministic
- Higher = more creative and unpredictable
- 0.8 balances creativity with coherence for assistant tasks

### top_p: 0.95

- Nucleus sampling - considers tokens until their cumulative probability hits this threshold
- 0.95 means it samples from the top 95% most likely next tokens
- Higher = more variety in word choice
- Works with temperature to control creativity

### top_k: 40

- Limits sampling to the top K most likely tokens
- 40 means it only considers the 40 most probable next words
- Prevents completely random/nonsensical outputs
- Balances variety with coherence

### num_predict: 4096

- Maximum tokens the model will generate in one response
- 4096 is good for most responses
- One token ≈ 0.75 words, so ~3000 words max

### repeat_penalty: 1.18

- Penalizes the model for repeating the same words/phrases
- 1.0 = no penalty, higher = stronger penalty
- 1.18 prevents annoying repetition without making it sound unnatural

### repeat_last_n: 256

- Look back N tokens for repetition detection
- 256 means it checks the last ~200 words for repeated patterns
- Prevents getting stuck in loops

### presence_penalty: 0.5

- Penalizes tokens that have appeared anywhere in the conversation
- Encourages using diverse vocabulary
- 0.5 is moderate - keeps responses fresh without being forced

### frequency_penalty: 0.5

- Penalizes tokens based on how often they've appeared
- Stronger penalty for frequently used words
- 0.5 reduces repetitive phrasing

### mirostat: 2

- Adaptive sampling mode (0=off, 1/2=on)
- Mode 2 dynamically adjusts sampling to maintain consistent quality
- Prevents the model from getting too repetitive or too random
- Best for longer conversations

### mirostat_tau: 8.0

- Target perplexity for mirostat - controls how predictable vs surprising the output is
- **Low (3.0-4.0)**: More predictable, safer responses, conventional phrasing
- **Moderate (5.0-6.0)**: Natural conversation, balanced and human-like
- **High (7.0-9.0)**: More creative and varied, takes conversational risks
- **Very High (10.0+)**: Unpredictable, experimental, can get weird
- 8.0 encourages more colorful and varied language for personality

### mirostat_eta: 0.1

- Learning rate for mirostat adjustments
- How quickly mirostat adapts to maintain target perplexity
- 0.1 is a moderate adjustment speed

## Requirements

- Python 3.12+
- Ollama running locally (install from <https://ollama.ai>)
- OpenAI API key (for memory embeddings)
- TimescaleDB instance (provided via Tiger Cloud)
