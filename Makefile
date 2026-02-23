.PHONY: help install setup setup-db setup-hooks run \
	format format-py format-md lint lint-py lint-md \
	test test-main test-core test-interfaces test-services test-tools test-flows \
	coverage logs clean ai-checks

help:
	@echo "Ollama Agent - Available Commands:"
	@echo ""
	@echo "  make install         - Install dependencies with uv"
	@echo "  make setup           - Setup environment (.env file)"
	@echo "  make setup-db        - Initialize TimescaleDB schema"
	@echo "  make setup-hooks     - Install git hooks with lefthook"
	@echo "  make run             - Start the chat interface"
	@echo "  make format-py       - Format Python code with ruff"
	@echo "  make format-md       - Format Markdown files"
	@echo "  make format          - Format code with ruff"
	@echo "  make lint-py         - Lint Python code with ruff"
	@echo "  make lint-md         - Check Markdown formatting"
	@echo "  make lint            - Lint code with ruff"
	@echo "  make test-main       - Run root-level tests"
	@echo "  make test-core       - Run core tests"
	@echo "  make test-interfaces - Run interface/CLI tests"
	@echo "  make test-services   - Run service tests"
	@echo "  make test-tools      - Run tool tests"
	@echo "  make test-flows      - Run all test flows by area"
	@echo "  make test            - Run full test suite with coverage + sonar reports"
	@echo "  make ai-checks       - Run format + lint + all test flows + coverage gate"
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

setup-hooks:
	@echo "Installing git hooks..."
	uv run lefthook install
	@echo "✓ Git hooks installed"

run:
	uv run python main.py

format: format-py format-md

format-py:
	uv run ruff format .

format-md:
	uv run mdformat *.md docs/ src/ tests/

lint: lint-py lint-md

lint-py:
	uv run ruff check .

lint-md:
	uv run mdformat --check *.md docs/ src/ tests/

test-main:
	uv run pytest tests/test_main.py

test-core:
	uv run pytest tests/core

test-interfaces:
	uv run pytest tests/interfaces

test-services:
	uv run pytest tests/services

test-tools:
	uv run pytest tests/tools

test-flows: test-main test-core test-interfaces test-services test-tools

test:
	@mkdir -p coverage
	COVERAGE_FILE=coverage/.coverage uv run pytest --cov-config=.coveragerc --cov=src --cov-report=html:coverage/html --cov-report=term --cov-report=xml:coverage/coverage.xml --junitxml=coverage/junit.xml --cov-fail-under=80

ai-checks: format lint test-flows test

logs:
	tail -f logs/ollama_chat.log

clean:
	rm -rf logs/*.logs
	rm -rf coverage .pytest_cache htmlcov .coverage
	rm -rf src/__pycache__ src/agent/__pycache__ tests/__pycache__ tests/integration/__pycache__
	rm -rf .venv .ruff_cache
	@echo "✓ Cleaned logs, env files, and test artifacts"
