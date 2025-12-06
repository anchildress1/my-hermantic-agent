# Memory Store API

Semantic memory storage using TimescaleDB + pgvector + OpenAI embeddings.

## Quick Reference

```python
from src.agent.memory import MemoryStore

store = MemoryStore()

# Store
mem_id = store.remember("User prefers Python", "preference", "coding")

# Search
results = store.recall("programming preferences", limit=5)

# Delete
store.forget(mem_id)

# Stats
stats = store.stats()
contexts = store.list_contexts()
```

## Methods

### remember(text, type, context, confidence=1.0, source_context=None)

Store a memory. Returns memory ID or None.

**Types:** `preference`, `fact`, `task`, `insight`

**Validation:**

- Text: 1-8000 chars
- Confidence: 0.0-1.0
- Rate limit: 10/min

### recall(query, type=None, context=None, limit=5, use_semantic=True)

Search memories. Returns list of dicts with `id`, `memory_text`, `type`, `context`, `similarity`, etc.

**Filters:**

- `type`: Filter by memory type
- `context`: Exact match or SQL LIKE pattern (`project-%`)
- `use_semantic`: Vector search (True) or full-text (False)
- Rate limit: 20/min

### forget(memory_id)

Delete memory. Returns True if deleted, False if not found.

### list_contexts()

Get all unique context tags. Returns list of strings.

### stats()

Get statistics. Returns dict with `total_memories`, `unique_types`, `unique_contexts`, `avg_confidence`, `last_memory_at`.

### close()

Close connection pool. Call on shutdown.

## Memory Types

- **preference**: User likes/dislikes ("prefers dark mode")
- **fact**: Factual info ("Python 3.12 released Oct 2023")
- **task**: Todos ("review PR #456")
- **insight**: Observations ("most productive in morning")

## Error Handling

All methods handle errors gracefully:

- Database errors → None/empty list/False
- API errors → Retry on rate limit, raise on timeout
- Validation errors → Raise ValueError
- Check `logs/ollama_chat.log` for details

## Examples

```python
# Basic usage
store = MemoryStore()
mem_id = store.remember("Prefers tabs over spaces", "preference", "coding")
results = store.recall("code formatting preferences")

# Filtered search
tasks = store.recall("", type="task", context="work", limit=10)

# Pattern matching
projects = store.recall("", context="project-%", use_semantic=False)

# Cleanup
old_tasks = store.recall("", type="task", limit=100)
for task in old_tasks:
    if task['created_at'] < cutoff_date:
        store.forget(task['id'])
```

## Troubleshooting

**"MEMORY_DB_URL not set"** → Add to `.env`  
**"Rate limit exceeded"** → Wait or reduce frequency  
**"Database connection failed"** → Run `make setup-db`  
**"No memories found"** → Check `stats()` to verify data exists
