-- ============================================================================
-- Hermes Agent Memory Events
-- Auditable event trail for memory operations
-- ============================================================================
-- This script is idempotent - safe to run multiple times

CREATE TABLE IF NOT EXISTS hermes.memory_events (
    id BIGSERIAL PRIMARY KEY,
    memory_id INTEGER NULL REFERENCES hermes.memories(id) ON DELETE SET NULL,
    operation VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL CHECK (status IN ('success', 'error')),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_events_created_at
    ON hermes.memory_events(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_memory_events_operation
    ON hermes.memory_events(operation);

CREATE INDEX IF NOT EXISTS idx_memory_events_memory_id
    ON hermes.memory_events(memory_id);

GRANT SELECT, INSERT ON hermes.memory_events TO hermes_app;
GRANT USAGE, SELECT ON SEQUENCE hermes.memory_events_id_seq TO hermes_app;

COMMENT ON TABLE hermes.memory_events IS 'Audit trail for memory operations';
COMMENT ON COLUMN hermes.memory_events.memory_id IS 'Referenced memory row when available';
COMMENT ON COLUMN hermes.memory_events.operation IS 'Operation name (remember, recall, forget, auto_remember, etc.)';
COMMENT ON COLUMN hermes.memory_events.status IS 'Operation result status (success or error)';
COMMENT ON COLUMN hermes.memory_events.details IS 'Structured payload for operation context and diagnostics';
COMMENT ON COLUMN hermes.memory_events.created_at IS 'Event timestamp';
