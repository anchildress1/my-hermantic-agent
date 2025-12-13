-- ============================================================================
-- Hermes Agent Memory Schema
-- TimescaleDB + pgvector semantic memory with role-based access control
-- ============================================================================
-- This script is idempotent - safe to run multiple times
-- Run with: psql $MEMORY_DB_URL < schema/000-init.sql

-- ============================================================================
-- Role Management
-- ============================================================================

-- Application role (read/write access)
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'hermes_app') THEN
        CREATE ROLE hermes_app WITH LOGIN PASSWORD 'hermes_app_password_placeholder';
        RAISE NOTICE 'Created role: hermes_app';
    END IF;
END $$;

-- Grant necessary privileges
ALTER ROLE hermes_app WITH NOCREATEDB NOCREATEROLE;

-- ============================================================================
-- Schema Setup
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS hermes AUTHORIZATION tsdbadmin;

GRANT USAGE ON SCHEMA hermes TO hermes_app;

SET search_path TO hermes;

-- ============================================================================
-- Main Memories Table
-- ============================================================================

CREATE TABLE IF NOT EXISTS memories (
    id SERIAL PRIMARY KEY,
    memory_text TEXT NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('preference', 'fact', 'task', 'insight')),
    tag VARCHAR(100) NOT NULL,
    importance FLOAT NOT NULL DEFAULT 1.0 CHECK (importance >= 0 AND importance <= 3),
    confidence FLOAT NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    source TEXT,
    embedding vector(1536) NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    embedding_dim INTEGER DEFAULT 1536 NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ,
    access_count INTEGER NOT NULL DEFAULT 0
);

-- Set explicit ownership
ALTER TABLE memories OWNER TO tsdbadmin;
ALTER SEQUENCE memories_id_seq OWNER TO tsdbadmin;

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Vector similarity search using IVFFlat algorithm
-- lists=100 is good for up to 100K memories, increase for larger datasets
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = 'hermes' AND indexname = 'idx_memories_embedding'
    ) THEN
        CREATE INDEX idx_memories_embedding ON memories
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        RAISE NOTICE 'Created index: idx_memories_embedding';
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_memories_tag ON memories(tag);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_fts ON memories
    USING gin(to_tsvector('english', memory_text));

-- ============================================================================
-- Convenience Views
-- ============================================================================

CREATE OR REPLACE VIEW memory_summary AS
SELECT
    id,
    LEFT(memory_text, 100) || CASE WHEN LENGTH(memory_text) > 100 THEN '...' ELSE '' END as preview,
    type,
    tag,
    importance,
    confidence,
    created_at,
    access_count
FROM memories
ORDER BY importance DESC, created_at DESC;

ALTER VIEW memory_summary OWNER TO tsdbadmin;

-- ============================================================================
-- Application Role Permissions
-- ============================================================================

-- Grant read/write on tables
GRANT SELECT, INSERT, UPDATE, DELETE ON memories TO hermes_app;
GRANT USAGE, SELECT ON SEQUENCE memories_id_seq TO hermes_app;
GRANT SELECT ON memory_summary TO hermes_app;

-- Allow app to set search parameters
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA hermes TO hermes_app;

-- ============================================================================
-- Schema Documentation
-- ============================================================================

COMMENT ON SCHEMA hermes IS 'Hermes agent semantic memory storage with vector embeddings';

COMMENT ON TABLE memories IS 'Semantic memory storage with vector embeddings and importance scoring';

COMMENT ON COLUMN memories.id IS 'Primary key';
COMMENT ON COLUMN memories.memory_text IS 'Memory content (max 8000 chars for OpenAI embedding limits)';
COMMENT ON COLUMN memories.type IS 'Memory classification: preference, fact, task, or insight';
COMMENT ON COLUMN memories.tag IS 'Tag for organization (e.g., work, personal, project-name)';
COMMENT ON COLUMN memories.importance IS 'Importance score 0.0-3.0 (0=low, 1=normal, 2=high, 3=critical); >2.0 should be prompted explicitly';
COMMENT ON COLUMN memories.confidence IS 'Confidence score 0.0-1.0 (0=uncertain, 1=certain)';
COMMENT ON COLUMN memories.source IS 'Optional snippet of conversation that triggered this memory';
COMMENT ON COLUMN memories.embedding IS 'Vector embedding (1536 dims for text-embedding-3-small, 3072 for 3-large)';
COMMENT ON COLUMN memories.embedding_model IS 'Model used to generate embedding (for tracking/migration)';
COMMENT ON COLUMN memories.embedding_dim IS 'Embedding vector dimensions';
COMMENT ON COLUMN memories.created_at IS 'Timestamp when memory was created';
COMMENT ON COLUMN memories.last_accessed IS 'Timestamp of last access (updated on recall)';
COMMENT ON COLUMN memories.access_count IS 'Number of times this memory has been recalled';

COMMENT ON INDEX idx_memories_embedding IS 'IVFFlat index for vector similarity search (lists=100 for up to 100K memories)';
COMMENT ON INDEX idx_memories_type IS 'B-tree index for filtering by memory type';
COMMENT ON INDEX idx_memories_tag IS 'B-tree index for filtering by tag';
COMMENT ON INDEX idx_memories_importance IS 'B-tree index for sorting by importance (DESC for high-first)';
COMMENT ON INDEX idx_memories_created_at IS 'B-tree index for sorting by creation time (DESC for recent-first)';
COMMENT ON INDEX idx_memories_fts IS 'GIN index for full-text search on memory content';

COMMENT ON VIEW memory_summary IS 'Convenience view for browsing memories with truncated text, sorted by importance';

-- ============================================================================
-- Verification
-- ============================================================================

DO $$
DECLARE
    mem_count INTEGER;
    schema_owner TEXT;
    table_owner TEXT;
BEGIN
    SELECT COUNT(*) INTO mem_count FROM memories;
    SELECT nspowner::regrole::text INTO schema_owner FROM pg_namespace WHERE nspname = 'hermes';
    SELECT tableowner INTO table_owner FROM pg_tables WHERE schemaname = 'hermes' AND tablename = 'memories';

    RAISE NOTICE 'Schema setup complete:';
    RAISE NOTICE '  Schema: hermes (owner: %)', schema_owner;
    RAISE NOTICE '  Table: memories (owner: %)', table_owner;
    RAISE NOTICE '  Total memories: %', mem_count;
    RAISE NOTICE '  App role: hermes_app (read/write access)';
END $$;
