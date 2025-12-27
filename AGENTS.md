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
