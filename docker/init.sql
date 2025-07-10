-- Initialize PostgreSQL for fichat-rag

-- Create pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create initial tables
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(384),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_documents_metadata 
ON documents USING GIN (metadata);

-- Add full-text search
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS content_tsvector tsvector
GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
ON documents USING GIN (content_tsvector);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO rag_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO rag_user;