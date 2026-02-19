# Memory System (Implementation Truth)

This document describes the memory system as implemented in code.

## Scope

- Single global user.
- Durable semantic memory in TigerData/PostgreSQL.
- OpenAI embeddings for semantic storage/recall.
- Automatic memory writing via LangMem extraction.
- Auditable memory operations through `hermes.memory_events`.

## Data Model

### `hermes.memories`

- Long-term semantic memories.
- Core fields: `memory_text`, `type`, `tag`, `importance`, `confidence`, `source`, `embedding`.
- Embedding column is currently `vector(1536)`.

### `hermes.memory_events`

- Audit trail for memory operations.
- Core fields: `memory_id`, `operation`, `status`, `details`, `created_at`.
- Operations currently include:
  - `remember`
  - `recall`
  - `forget`
  - `auto_remember`

## Runtime Components

- `src/services/memory/vector_store.py`
  - DB reads/writes for memories.
  - Embedding generation via OpenAI.
  - Audit event creation for memory operations.
- `src/services/memory/langmem_extractor.py`
  - LangMem + LangChain extraction of structured memory candidates.
  - Relevance-focused settings:
    - inserts only
    - updates disabled
    - deletes disabled
- `src/services/memory/auto_writer.py`
  - Turn-level automatic memory writing.
  - Duplicate check before insert.

## CLI Commands

When semantic memory is enabled:

- Memory writes and recall behavior are driven through normal conversation.
- The agent should call memory tools automatically when appropriate.
- `/audit [operation]` is available as an operator view for event history.

## Automatic Memory Writing

- Runs on normal user/assistant turns.
- Extracts memory candidates from latest turn.
- Inserts new memories and revives exact duplicates (refreshes access metadata and bumps importance slightly).
- Emits `auto_remember` audit events for successful auto writes.
- Surfaces failed write attempts in chat output with the attempted memory text and exact DB error.
- Explicit user remember intent (messages containing "remember") is boosted to high importance automatically.

## Relevance Notes

- LangMem extraction is configured for relevance-first behavior:
  - model provider defaults to local `ollama` (overrideable via env).
  - model defaults to the active chat model when `LANGMEM_MODEL` is not set.
  - temperature is configurable and defaults to `0.2`.
  - only inserts are allowed in extractor decisions.
- Memory writes are model-mediated with policy gates (duplicate checks, explicit remember priority boost).

## Current Limits

- Embedding dimension mismatch is still possible if model/env changes away from 1536 without schema migration.
- Memory deletion path is soft delete (`deleted_at`), preserving historical auditability.
- `memory_events` is append-only but intentionally simple (no actor/session partitioning yet).

## Setup

Run migrations:

```bash
make setup-db
```

This applies all SQL files in `schema/` in lexical order.
