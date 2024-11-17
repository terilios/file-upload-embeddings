import pytest
import numpy as np
from datetime import datetime
from typing import List, Dict

from app.database.vector_store import VectorStore
from app.database.models import Document, DocumentChunk
from config.settings import settings

@pytest.fixture
def vector_store(db_session):
    """Create a vector store instance for testing."""
    return VectorStore(db_session)

@pytest.fixture
def test_chunks(mock_embeddings):
    """Create test chunks with embeddings."""
    return [
        {
            "content": "First test chunk",
            "embedding": mock_embeddings["simple"],
            "token_count": 3,
            "metadata": {"index": 0, "total_chunks": 2}
        },
        {
            "content": "Second test chunk",
            "embedding": mock_embeddings["simple"],
            "token_count": 3,
            "metadata": {"index": 1, "total_chunks": 2}
        }
    ]

async def test_store_document(vector_store, test_chunks):
    """Test storing a document with chunks."""
    # Store test document
    document = await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks,
        metadata={"test": "metadata"}
    )
    
    # Verify document was stored
    assert document.id is not None
    assert document.filename == "test.txt"
    assert document.content_type == "text/plain"
    assert document.file_size == 100
    assert document.metadata == {"test": "metadata"}
    
    # Verify chunks were stored
    assert len(document.chunks) == len(test_chunks)
    for chunk, test_chunk in zip(document.chunks, test_chunks):
        assert chunk.content == test_chunk["content"]
        assert chunk.token_count == test_chunk["token_count"]
        assert chunk.metadata == test_chunk["metadata"]
        assert np.allclose(chunk.embedding, test_chunk["embedding"])

async def test_similarity_search(vector_store, test_chunks):
    """Test similarity search functionality."""
    # Store test document
    document = await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks
    )
    
    # Perform similarity search
    query_embedding = test_chunks[0]["embedding"]
    results = await vector_store.similarity_search(
        query_embedding=query_embedding,
        top_k=2
    )
    
    # Verify results
    assert len(results) > 0
    for chunk, score in results:
        assert isinstance(chunk, DocumentChunk)
        assert isinstance(score, float)
        assert 0 <= score <= 1
        assert chunk.document_id == document.id

async def test_similarity_search_with_threshold(vector_store, test_chunks):
    """Test similarity search with custom threshold."""
    # Store test document
    await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks
    )
    
    # Create a dissimilar query embedding
    query_embedding = [-x for x in test_chunks[0]["embedding"]]
    
    # Search with high threshold
    results = await vector_store.similarity_search(
        query_embedding=query_embedding,
        threshold=0.9
    )
    assert len(results) == 0
    
    # Search with low threshold
    results = await vector_store.similarity_search(
        query_embedding=query_embedding,
        threshold=0.1
    )
    assert len(results) > 0

async def test_update_document_embeddings(vector_store, test_chunks, mock_openai):
    """Test updating document embeddings."""
    # Store test document
    document = await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks
    )
    
    # Update embeddings
    updated_doc = await vector_store.update_document_embeddings(
        document_id=document.id,
        force=True
    )
    
    # Verify embeddings were updated
    assert updated_doc.id == document.id
    for chunk in updated_doc.chunks:
        assert len(chunk.embedding) == settings.VECTOR_DIMENSION

async def test_delete_document(vector_store, test_chunks):
    """Test document deletion."""
    # Store test document
    document = await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks
    )
    
    # Delete document
    success = await vector_store.delete_document(document.id)
    assert success
    
    # Verify document was deleted
    deleted_doc = await vector_store.get_document_by_id(document.id)
    assert deleted_doc is None
    
    # Try deleting non-existent document
    success = await vector_store.delete_document(999)
    assert not success

async def test_get_document_by_id(vector_store, test_chunks):
    """Test retrieving document by ID."""
    # Store test document
    original_doc = await vector_store.store_document(
        filename="test.txt",
        content_type="text/plain",
        file_size=100,
        chunks=test_chunks
    )
    
    # Retrieve document
    retrieved_doc = await vector_store.get_document_by_id(original_doc.id)
    
    # Verify document data
    assert retrieved_doc is not None
    assert retrieved_doc.id == original_doc.id
    assert retrieved_doc.filename == original_doc.filename
    assert len(retrieved_doc.chunks) == len(test_chunks)

async def test_get_all_documents(vector_store, test_chunks):
    """Test retrieving all documents with pagination."""
    # Store multiple test documents
    docs = []
    for i in range(3):
        doc = await vector_store.store_document(
            filename=f"test_{i}.txt",
            content_type="text/plain",
            file_size=100,
            chunks=test_chunks
        )
        docs.append(doc)
    
    # Test pagination
    page1 = await vector_store.get_all_documents(skip=0, limit=2)
    assert len(page1) == 2
    
    page2 = await vector_store.get_all_documents(skip=2, limit=2)
    assert len(page2) == 1
    
    # Verify no duplicate documents
    all_ids = [doc.id for doc in page1 + page2]
    assert len(all_ids) == len(set(all_ids))

async def test_error_handling(vector_store):
    """Test error handling in vector store operations."""
    # Test invalid document ID
    with pytest.raises(ValueError):
        await vector_store.update_document_embeddings(999)
    
    # Test empty chunks list
    with pytest.raises(Exception):
        await vector_store.store_document(
            filename="test.txt",
            content_type="text/plain",
            file_size=100,
            chunks=[]
        )
    
    # Test invalid embedding dimension
    with pytest.raises(Exception):
        await vector_store.similarity_search(
            query_embedding=[1.0] * (settings.VECTOR_DIMENSION - 1)
        )
