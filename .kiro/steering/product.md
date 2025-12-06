---
inclusion: always
---

# Product Overview

Ollama Agent is a CLI-based conversational agent that explores Ollama model templates with persistent memory capabilities.

## Core Features

- Interactive chat interface with local Ollama models
- Persistent conversation history with auto-save/load
- Semantic memory storage using TimescaleDB and OpenAI embeddings
- Smart context management with automatic trimming
- Configurable model parameters via YAML templates
- Repetition detection and logging

## Memory System

The agent uses a dual-memory approach:

1. **Conversation Memory**: Full chat history stored in `data/memory.json` for context continuity
2. **Semantic Memory**: Distilled facts, preferences, tasks, and insights stored in TimescaleDB with vector embeddings for semantic search

## Target Use Case

Personal assistant for developers who want:
- Direct, no-BS communication style
- Persistent context across sessions
- Local model control with Ollama
- Semantic memory for long-term knowledge retention
