-- ============================================================================
-- TimescaleDB + pgvector setup for agent semantic memory
-- ============================================================================
-- This script is idempotent - safe to run multiple times
-- Run with: psql $MEMORY_DB_URL < schema/init.sql

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS timescaledb; -- TimescaleDB for time-series

-- ============================================================================
-- Main memories table
-- ============================================================================
CREATE TABLE IF NOT EXISTS agent_memories (
    id SERIAL PRIMARY KEY,
    memory_text TEXT NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('preference', 'fact', 'task', 'insight')),
    context VARCHAR(100) NOT NULL,
    confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    source_context TEXT,
    embedding vector(1536) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_dim INTEGER DEFAULT 1536 NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0
);

-- ============================================================================
-- Indexes for performance
-- ============================================================================

-- Vector similarity search using IVFFlat algorithm
-- lists=100 is good for up to 100K memories, increase for larger datasets
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_memories_embedding'
    ) THEN
        CREATE INDEX idx_memories_embedding ON agent_memories
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memories_type ON agent_memories(type);

CREATE INDEX IF NOT EXISTS idx_memories_context ON agent_memories(context);

CREATE INDEX IF NOT EXISTS idx_memories_created_at ON agent_memories(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_fts ON agent_memories
    USING gin(to_tsvector('english', memory_text));

-- ============================================================================
-- Convenience view for browsing memories
-- ============================================================================
CREATE OR REPLACE VIEW memory_summary AS
SELECT
    id,
    -- Truncate long text for preview
    LEFT(memory_text, 100) || CASE WHEN LENGTH(memory_text) > 100 THEN '...' ELSE '' END as preview,
    type,
    context,
    confidence,
    created_at,
    access_count
FROM agent_memories
ORDER BY created_at DESC;

-- ============================================================================
-- Grant permissions to tsdbadmin role
-- ============================================================================
GRANT ALL PRIVILEGES ON TABLE agent_memories TO tsdbadmin;
GRANT ALL PRIVILEGES ON SEQUENCE agent_memories_id_seq TO tsdbadmin;
GRANT ALL PRIVILEGES ON TABLE memory_summary TO tsdbadmin;

-- ============================================================================
-- PostgreSQL comments for schema documentation
-- ============================================================================
COMMENT ON TABLE agent_memories IS 'Semantic memory storage for agent with vector embeddings';

COMMENT ON COLUMN agent_memories.id IS 'Primary key';
COMMENT ON COLUMN agent_memories.memory_text IS 'Memory content (max 8000 chars for OpenAI embedding limits)';
COMMENT ON COLUMN agent_memories.type IS 'Memory classification: preference, fact, task, or insight';
COMMENT ON COLUMN agent_memories.context IS 'Context tag for organization (e.g., work, personal, project-name)';
COMMENT ON COLUMN agent_memories.confidence IS 'Confidence score 0.0-1.0 (0=uncertain, 1=certain)';
COMMENT ON COLUMN agent_memories.source_context IS 'Optional snippet of conversation that triggered this memory';
COMMENT ON COLUMN agent_memories.embedding IS 'Vector embedding (default 1536 dimensions for text-embedding-3-small, configurable via EMBEDDING_DIM)';
COMMENT ON COLUMN agent_memories.embedding_model IS 'Model used to generate embedding (for tracking/migration)';
COMMENT ON COLUMN agent_memories.embedding_dim IS 'Embedding vector dimensions (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)';
COMMENT ON COLUMN agent_memories.created_at IS 'Timestamp when memory was created';
COMMENT ON COLUMN agent_memories.last_accessed IS 'Timestamp of last access (updated on recall)';
COMMENT ON COLUMN agent_memories.access_count IS 'Number of times this memory has been recalled';

COMMENT ON INDEX idx_memories_embedding IS 'IVFFlat index for vector similarity search (lists=100 for up to 100K memories)';
COMMENT ON INDEX idx_memories_type IS 'B-tree index for filtering by memory type';
COMMENT ON INDEX idx_memories_context IS 'B-tree index for filtering by context';
COMMENT ON INDEX idx_memories_created_at IS 'B-tree index for sorting by creation time (DESC for recent-first)';
COMMENT ON INDEX idx_memories_fts IS 'GIN index for full-text search on memory content';

COMMENT ON VIEW memory_summary IS 'Convenience view for browsing memories with truncated text';

-- ============================================================================
-- Verification queries (uncomment to test)
-- ============================================================================
-- SELECT COUNT(*) as total_memories FROM agent_memories;
-- SELECT type, COUNT(*) as count FROM agent_memories GROUP BY type;
-- SELECT context, COUNT(*) as count FROM agent_memories GROUP BY context;
