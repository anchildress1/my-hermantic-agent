.PHONY: help install setup setup-db run test coverage logs clean format lint

help:
	@echo "Ollama Agent - Available Commands:"
	@echo ""
	@echo "  make install         - Install dependencies with uv"
	@echo "  make setup           - Setup environment (.env file)"
	@echo "  make setup-db        - Initialize TimescaleDB schema"
	@echo "  make run             - Start the chat interface"
	@echo "  make format          - Format code with ruff"
	@echo "  make lint            - Lint code with ruff"
	@echo "  make test            - Run tests"
	@echo "  make coverage        - Run tests with coverage report"
	@echo "  make logs            - Tail application logs"
	@echo "  make clean           - Clean generated files and logs"

install:
	uv sync --all-extras

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file - please add your OPENAI_API_KEY and MEMORY_DB_URL"; \
	else \
		echo "✓ .env file already exists"; \
	fi

setup-db:
	@echo "Initializing TimescaleDB schema..."
	uv run python scripts/setup_db.py

run:
	uv run python main.py

format:
	uv run ruff format .
	uv run mdformat .

lint:
	uv run ruff check . --fix
	uv run mdformat --check .

test:
	uv run pytest tests/ -v

coverage:
	uv run pytest tests/ --cov=src --cov-report=term-missing --cov-report=html

logs:
	tail -f logs/ollama_chat.log

clean:
	rm -rf logs/*.logs
	rm -rf .coverage .pytest_cache htmlcov
	rm -rf src/__pycache__ src/agent/__pycache__ tests/__pycache__ tests/integration/__pycache__
	rm -rf .venv .ruff_cache
	@echo "✓ Cleaned logs, memory files, and test artifacts"
