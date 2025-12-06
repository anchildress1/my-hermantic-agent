.PHONY: help install setup run test coverage logs clean

help:
	@echo "Ollama Agent - Available Commands:"
	@echo ""
	@echo "  make install    - Install dependencies with uv"
	@echo "  make setup      - Setup environment (.env file)"
	@echo "  make run        - Start the chat interface"
	@echo "  make test       - Run tests with pytest"
	@echo "  make coverage   - Run tests with coverage report"
	@echo "  make logs       - Tail application logs"
	@echo "  make clean      - Clean generated files and logs"

install:
	uv sync

setup:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env file - please add your OPENAI_API_KEY"; \
	else \
		echo "✓ .env file already exists"; \
	fi

run:
	uv run python main.py

test:
	uv run --with pytest pytest tests/ -v

coverage:
	uv run --with pytest --with pytest-cov pytest tests/ --cov=src --cov-report=term-missing

logs:
	tail -f logs/ollama_chat.log

clean:
	rm -rf logs/*.log
	rm -rf data/memory.json
	rm -rf .coverage .pytest_cache htmlcov
	@echo "✓ Cleaned logs, memory files, and test artifacts"
