# Agent Memory Baseline Charter

This file is the baseline memory contract for future implementation work.

## Baseline Intent

Build a production-grade conversational agent memory system that is:

- Durable across crashes, restarts, and upgrades.
- Auditable with explicit write/read/delete history.
- Sustainable for long-term operation (retention, versioning, migrations).
- Centered on Ollama + Hermes local inference.
- Backed by TigerData for persistent memory storage and retrieval.
- Integrated with LangChain + LangMem SDK (not ad-hoc glue only).

## Non-Negotiable Requirements

- Memory writes must be traceable to source conversation context.
- Memory updates/deletes must preserve audit evidence (soft-delete/tombstone/event log), not silent hard delete.
- Embedding pipeline must support offline/local mode (no mandatory cloud API dependency).
- Schema and code must support embedding model evolution without dimension lock failures.
- Reliability posture must include retries, timeouts, and explicit failure surfacing.
- Operational readiness must include migration strategy, observability, and backup/restore drills.

## Baseline Seed Record

- Seed date: `2026-02-19`
- TigerData service id: `faapo6i9vp`
- Seed memory id: `1`
- Seed tag: `agent-baseline`

## Implementation Questions To Resolve

- Which local embedding model replaces mandatory OpenAI embeddings for offline mode?
- How should LangMem memory lifecycle events map into TigerData audit tables?
- What retention/compaction policy governs short-term context vs long-term semantic memory?
- What is the migration strategy when embedding dimension/model changes?
- What SLOs define memory system reliability (write success, recall latency, recall relevance)?
- What validation gate prevents docs from drifting away from implemented commands?
