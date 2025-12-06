-- Migration: Add embedding_dim column to track dimension size
-- This allows supporting multiple embedding models with different dimensions

-- Add column to track embedding dimensions
ALTER TABLE agent_memories 
ADD COLUMN IF NOT EXISTS embedding_dim INTEGER DEFAULT 1536 NOT NULL;

-- Add comment
COMMENT ON COLUMN agent_memories.embedding_dim IS 'Embedding vector dimensions (1536 for text-embedding-3-small, 3072 for text-embedding-3-large)';

-- Note: Existing embeddings are 1536 dimensions (text-embedding-3-small)
-- To use different dimensions, you'll need to:
-- 1. Set EMBEDDING_DIM in .env
-- 2. Recreate the table or add new column with different vector size
-- 3. Re-generate all embeddings with the new model
