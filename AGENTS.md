# AI Assistant Context & Rules

This document defines the coding standards, architectural patterns, and context for AI assistants (GitHub Copilot, Claude, Cursor, etc.) working on the **Hermantic Agent** repository.

## 1. Project Identity

- **Name:** My Hermantic Agent
- **Status:** Personal Toy / Experiment (Not for release/deployment)
- **Purpose:** Autonomous agent system experimenting with embeddings, orchestration, and local LLMs.
- **Core Stack:** Python 3.12+, PostgreSQL (`psycopg2`), Ollama, OpenAI API.

## 2. Tech Stack & Dependencies

- **Language:** Python 3.12 (Strict requirement) via `uv`
- **Database:** PostgreSQL.
  - Driver: `psycopg2-binary`.
  - Schema: Defined in `schema/000-init.sql`.
- **LLM Integration:**
  - Local: `ollama` with Hermes 4 LLM 14B.
  - Cloud: `openai`.
- **Configuration:** `pyyaml` for YAML config, `python-dotenv` for secrets.
- **Testing:** `pytest`, `pytest-mock`.

## 3. Coding Standards

### Evolution Strategy

- **Breaking Changes:** Explicitly encouraged over complicated solutions or backward compatibility.
- **Hard-coding:** Acceptable and expected at this stage. We can change code freely.
- **Versioning:** Prerelease patch versions only. Manual changelog updates by the agent.

### General

- **Style:** Follow PEP 8 strictly.
- **Type Hints:** **MANDATORY** for all function signatures (arguments and return types). Use `typing` module or standard collection types (Python 3.9+ style `list[]`, `dict[]` is preferred).
- **Docstrings:** Required for all public modules, classes, and functions. Use Google-style docstrings.
- **Imports:**
  - Group imports: Standard library -> Third-party -> Local application.
  - Use absolute imports for local modules (e.g., `from src.agent import memory`).
- **Inline code comments:** must add value not otherwise apparent from reading the code

### Error Handling

- Use specific exceptions (e.g., `ValueError`, `ConnectionError`) rather than bare `Exception`.
- Wrap database operations in `try/except` blocks with rollback logic where appropriate.
- Log errors using the standard `logging` module, not `print()`.

### Database Interactions

- **Tiger MCP:** should be used whenever database changes are required
- **SQL:** Write raw SQL in `src/` is acceptable but prefer parameterized queries to prevent injection.
- **Connections:** Manage connections using context managers (`with conn: ...`) to ensure closure.
- **Credentials:** Always check the root `.env` file for database credentials and API keys before prompting the user. The `MEMORY_DB_URL` variable contains the full connection string for the Tiger Cloud database.

## 4. Project Structure

- `src/`: Source code root.
  - `agent/`: Core agent logic (chat, memory, orchestration).
- `schema/`: Database initialization and migrations.
- `config/`: Configuration templates.
- `tests/`: Pytest suite.
- `scripts/`: Utility scripts for setup and maintenance.

## 5. AI Behavior Guidelines

When generating code or answering questions:

1. **Context Awareness:** Always check `pyproject.toml` for available dependencies before suggesting new imports.
1. **Security:** Never hardcode API keys or passwords. Use `os.getenv` or `config` objects.
1. **Testing:**
   - Focus on highly used, high-priority functionality only.
   - Maintain a minimum of 80% coverage for critical paths.
   - No integration tests required at this time.
1. **Brevity:** Provide concise explanations. Focus on the code solution.
1. **Pathing:** Assume the workspace root is the current working directory.
1. **Documentation:** All documentation in this repository should be written based on AI agent chat functionality, describing chat commands and workflows rather than programmatic APIs.

## 6. Common Tasks

- **Adding a Migration:** Create a new `.`sql file in `schema/` with a sequential prefix.
- **Executing commands:** Use `make` from the root directory pointing to a valid `uv` call

## 7. Agent Memory (Source of Truth)

This section is the canonical memory system reference. Do not duplicate or fork this content under `docs/`.

### Scope

- Single global user.
- Durable semantic memory in TigerData/PostgreSQL.
- OpenAI embeddings for semantic storage/recall.
- Automatic memory writing via LangMem extraction.
- Auditable memory operations through `hermes.memory_events`.

### Data Model

#### `hermes.memories`

- Long-term semantic memories.
- Core fields: `memory_text`, `type`, `tag`, `importance`, `confidence`, `source`, `embedding`.
- Embedding column is currently `vector(1536)`.

#### `hermes.memory_events`

- Audit trail for memory operations.
- Core fields: `memory_id`, `operation`, `status`, `details`, `created_at`.
- Time-based retention is enforced by pruning events older than `MEMORY_EVENTS_RETENTION_DAYS` (default `90`).
- Operations currently include:
  - `remember`
  - `recall`
  - `forget`
  - `auto_remember`

### Runtime Components

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

### CLI Commands

When semantic memory is enabled:

- Memory writes and recall behavior are driven through normal conversation.
- The agent should call memory tools automatically when appropriate.
- `/audit [operation]` is available as an operator view for event history.

### Automatic Memory Writing

- Runs on normal user/assistant turns.
- Extracts memory candidates from latest turn.
- Inserts new memories and revives exact duplicates (refreshes access metadata and bumps importance slightly).
- Emits `auto_remember` audit events for successful auto writes.
- Surfaces failed write attempts in chat output with the attempted memory text and exact DB error.
- Explicit user remember intent (messages containing "remember") is boosted to high importance automatically.

### Tombstone Lifecycle Policy

- Exact duplicate remembers reconcile before insert:
  - merge into active match (`action=merged_active`)
  - revive soft-deleted match (`action=revived_tombstone`)
  - insert only when no exact match exists (`action=insert_new`)
- Forget operations are fully auditable with lifecycle actions:
  - `action=tombstone_created`
  - `action=already_tombstoned`
  - `action=not_found`

### Relevance Notes

- LangMem extraction is configured for relevance-first behavior:
  - model provider defaults to local `ollama` (overrideable via env).
  - model defaults to the active chat model when `LANGMEM_MODEL` is not set.
  - temperature is configurable and defaults to `0.2`.
  - only inserts are allowed in extractor decisions.
- Memory writes are model-mediated with policy gates (duplicate checks, explicit remember priority boost).
- Relevance regressions are guarded by fixture-based tolerance tests in `tests/services/memory/fixtures/relevance_regression.json`.

### Current Limits

- Embedding dimension mismatch is still possible if model/env changes away from 1536 without schema migration.
- Memory deletion path is soft delete (`deleted_at`), preserving historical auditability.
- `memory_events` audit records are retained by age only (no actor/session partitioning yet).

### Setup

Run migrations:

```bash
make setup-db
```

This applies all SQL files in `schema/` in lexical order.

## 8. Agent Memory Baseline Charter

This section is the baseline memory contract for future implementation work.

### Baseline Intent

Build a production-grade conversational agent memory system that is:

- Durable across crashes, restarts, and upgrades.
- Auditable with explicit write/read/delete history.
- Sustainable for long-term operation (retention, versioning, migrations).
- Centered on Ollama + Hermes local inference.
- Backed by TigerData for persistent memory storage and retrieval.
- Integrated with LangChain + LangMem SDK (not ad-hoc glue only).

### Non-Negotiable Requirements

- Memory writes must be traceable to source conversation context.
- Memory updates/deletes must preserve audit evidence (soft-delete/tombstone/event log), not silent hard delete.
- Embedding pipeline must support offline/local mode (no mandatory cloud API dependency).
- Schema and code must support embedding model evolution without dimension lock failures.
- Reliability posture must include retries, timeouts, and explicit failure surfacing.
- Operational readiness must include migration strategy, observability, and backup/restore drills.

### Baseline Seed Record

- Seed date: `2026-02-19`
- TigerData service id: `faapo6i9vp`
- Seed memory id: `1`
- Seed tag: `agent-baseline`

### Implementation Questions To Resolve

- Which local embedding model replaces mandatory OpenAI embeddings for offline mode?
- How should LangMem memory lifecycle events map into TigerData audit tables?
- What retention/compaction policy governs short-term context vs long-term semantic memory?
- What is the migration strategy when embedding dimension/model changes?
- What SLOs define memory system reliability (write success, recall latency, recall relevance)?
- What validation gate prevents docs from drifting away from implemented commands?
