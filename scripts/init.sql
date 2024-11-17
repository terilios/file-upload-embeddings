-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;

-- Create custom functions
CREATE OR REPLACE FUNCTION cosine_similarity(a vector, b vector) 
RETURNS float
AS $$
SELECT 1 - (a <=> b);
$$ LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE;

-- Create indexes if tables exist
DO $$
BEGIN
    -- Create indexes for document_chunks if table exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = 'document_chunks'
    ) THEN
        -- Create index for vector similarity search
        CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx 
        ON document_chunks 
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
        
        -- Create index for document_id for faster joins
        CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
        ON document_chunks(document_id);
        
        -- Create index for chunk_index for ordered retrieval
        CREATE INDEX IF NOT EXISTS document_chunks_chunk_index_idx
        ON document_chunks(chunk_index);
    END IF;

    -- Create indexes for documents if table exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = 'documents'
    ) THEN
        -- Create index for filename search
        CREATE INDEX IF NOT EXISTS documents_filename_idx
        ON documents(filename);
        
        -- Create index for created_at timestamp
        CREATE INDEX IF NOT EXISTS documents_created_at_idx
        ON documents(created_at);
    END IF;

    -- Create indexes for chat_messages if table exists
    IF EXISTS (
        SELECT 1 
        FROM information_schema.tables 
        WHERE table_name = 'chat_messages'
    ) THEN
        -- Create index for session_id for faster message retrieval
        CREATE INDEX IF NOT EXISTS chat_messages_session_id_idx
        ON chat_messages(session_id);
        
        -- Create index for created_at timestamp
        CREATE INDEX IF NOT EXISTS chat_messages_created_at_idx
        ON chat_messages(created_at);
    END IF;
END
$$;

-- Set up any necessary permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Configure pgvector parameters
ALTER SYSTEM SET maintenance_work_mem = '1GB';
ALTER SYSTEM SET max_parallel_workers_per_gather = '4';
ALTER SYSTEM SET max_parallel_workers = '8';
ALTER SYSTEM SET max_parallel_maintenance_workers = '4';

-- Optimize for vector operations
ALTER SYSTEM SET effective_cache_size = '4GB';
ALTER SYSTEM SET shared_buffers = '1GB';
ALTER SYSTEM SET work_mem = '64MB';

-- Reload configuration
SELECT pg_reload_conf();
