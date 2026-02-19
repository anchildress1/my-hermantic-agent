-- ============================================================================
-- Hermes Agent Soft Delete
-- Preserve memory records while hiding deleted rows from active queries
-- ============================================================================
-- This script is idempotent - safe to run multiple times

ALTER TABLE hermes.memories
    ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ;

COMMENT ON COLUMN hermes.memories.deleted_at IS 'Soft-delete timestamp. NULL means active memory.';

CREATE INDEX IF NOT EXISTS idx_memories_deleted_at
    ON hermes.memories(deleted_at DESC);

CREATE INDEX IF NOT EXISTS idx_memories_active_created_at
    ON hermes.memories(created_at DESC)
    WHERE deleted_at IS NULL;
