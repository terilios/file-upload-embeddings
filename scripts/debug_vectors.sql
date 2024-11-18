-- Drop existing index
DROP INDEX IF EXISTS document_chunks_embedding_idx;

-- Verify vector storage
SELECT 
    id,
    document_id,
    embedding IS NULL as is_null_embedding,
    embedding <=> embedding as self_similarity  -- Should be 0
FROM document_chunks
LIMIT 5;

-- Check for null embeddings
SELECT 
    COUNT(*) as total_chunks,
    COUNT(embedding) as chunks_with_embeddings,
    COUNT(*) FILTER (WHERE embedding IS NULL) as null_embeddings
FROM document_chunks;

-- Sample similarity scores between consecutive chunks
WITH chunk_pairs AS (
    SELECT 
        a.id as id1,
        b.id as id2,
        a.document_id,
        1 - (a.embedding <=> b.embedding) as similarity,
        a.chunk_index as chunk_index1,
        b.chunk_index as chunk_index2
    FROM document_chunks a
    JOIN document_chunks b ON a.document_id = b.document_id 
        AND a.chunk_index = b.chunk_index - 1
    LIMIT 5
)
SELECT * FROM chunk_pairs;

-- Test similarity search with a sample query
WITH sample_query AS (
    SELECT embedding 
    FROM document_chunks 
    WHERE id = 1  -- Use first chunk as sample query
),
similarity_test AS (
    SELECT 
        dc.id,
        dc.document_id,
        dc.chunk_index,
        1 - (dc.embedding <=> sq.embedding) as similarity
    FROM document_chunks dc, sample_query sq
    WHERE 1 - (dc.embedding <=> sq.embedding) > 0.5
    ORDER BY similarity DESC
    LIMIT 5
)
SELECT * FROM similarity_test;

-- Recreate index with new settings
CREATE INDEX document_chunks_embedding_idx 
ON document_chunks 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 50);  -- Reduced for better recall

-- Analyze the table to update statistics
ANALYZE document_chunks;
