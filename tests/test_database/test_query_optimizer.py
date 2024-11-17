import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict

from app.database.query_optimizer import QueryOptimizer, query_optimizer
from app.database.vector_store import VectorStore
from config.settings import settings

@pytest.fixture
async def optimizer():
    """Create a query optimizer instance for testing."""
    optimizer = QueryOptimizer()
    yield optimizer
    await optimizer.reset_stats()

@pytest.fixture
async def test_data(db_session, mock_openai):
    """Create test documents and embeddings."""
    vector_store = VectorStore(db_session)
    
    # Create test documents with different content types
    documents = []
    for i in range(3):
        chunks = [
            {
                "content": f"Test content {i} chunk {j}",
                "embedding": np.random.rand(settings.VECTOR_DIMENSION).tolist(),
                "token_count": 10,
                "metadata": {"chunk_index": j}
            }
            for j in range(2)
        ]
        
        doc = await vector_store.store_document(
            filename=f"test_{i}.txt",
            content_type=f"text/type_{i}",
            file_size=100,
            chunks=chunks,
            metadata={"test_id": i}
        )
        documents.append(doc)
    
    return documents

@pytest.mark.asyncio
async def test_optimize_similarity_query(optimizer):
    """Test query optimization without filters."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    query = await optimizer.optimize_similarity_query(
        query_embedding,
        top_k=5,
        threshold=0.5
    )
    
    assert "SELECT" in query
    assert "similarity_scores" in query
    assert "ORDER BY" in query
    assert "LIMIT" in query

@pytest.mark.asyncio
async def test_optimize_query_with_filters(optimizer):
    """Test query optimization with filters."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    filters = {
        "document_ids": [1, 2],
        "content_types": ["text/plain"],
        "date_range": {
            "date_from": datetime.now() - timedelta(days=1),
            "date_to": datetime.now()
        }
    }
    
    query = await optimizer.optimize_similarity_query(
        query_embedding,
        filters=filters
    )
    
    assert "document_id = ANY(:document_ids)" in query
    assert "content_type = ANY(:content_types)" in query
    assert "created_at BETWEEN" in query

@pytest.mark.asyncio
async def test_execute_similarity_query(optimizer, db_session, test_data):
    """Test query execution with real data."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    results = await optimizer.execute_similarity_query(
        query_embedding,
        db_session,
        top_k=5
    )
    
    assert len(results) > 0
    for result in results:
        assert "id" in result
        assert "document_id" in result
        assert "content" in result
        assert "similarity" in result
        assert 0 <= result["similarity"] <= 1

@pytest.mark.asyncio
async def test_filtered_query_execution(optimizer, db_session, test_data):
    """Test query execution with filters."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    # Filter by specific document IDs
    doc_ids = [doc.id for doc in test_data[:2]]
    results = await optimizer.execute_similarity_query(
        query_embedding,
        db_session,
        filters={"document_ids": doc_ids}
    )
    
    assert len(results) > 0
    assert all(r["document_id"] in doc_ids for r in results)

@pytest.mark.asyncio
async def test_query_performance_analysis(optimizer, db_session, test_data):
    """Test query performance analysis."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    analysis = await optimizer.analyze_query_performance(
        query_embedding,
        db_session
    )
    
    assert "query_plan" in analysis
    assert "statistics" in analysis
    assert "recommendations" in analysis
    assert isinstance(analysis["recommendations"], list)

@pytest.mark.asyncio
async def test_optimization_statistics(optimizer, db_session, test_data):
    """Test optimization statistics tracking."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    # Execute multiple queries
    for _ in range(3):
        await optimizer.execute_similarity_query(
            query_embedding,
            db_session
        )
    
    stats = await optimizer.get_optimization_stats()
    assert stats["queries_executed"] == 3
    assert "avg_query_time" in stats
    assert "cache_hit_ratio" in stats

@pytest.mark.asyncio
async def test_query_with_threshold(optimizer, db_session, test_data):
    """Test query execution with similarity threshold."""
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    # Set high threshold
    results = await optimizer.execute_similarity_query(
        query_embedding,
        db_session,
        threshold=0.9
    )
    
    # Set low threshold
    results_low = await optimizer.execute_similarity_query(
        query_embedding,
        db_session,
        threshold=0.1
    )
    
    assert len(results) <= len(results_low)

@pytest.mark.asyncio
async def test_error_handling(optimizer, db_session):
    """Test error handling in query optimization."""
    # Test with invalid embedding
    with pytest.raises(Exception):
        await optimizer.execute_similarity_query(
            query_embedding=[1.0],  # Invalid dimension
            session=db_session
        )
    
    # Test with invalid filters
    with pytest.raises(Exception):
        await optimizer.execute_similarity_query(
            query_embedding=np.random.rand(settings.VECTOR_DIMENSION).tolist(),
            session=db_session,
            filters={"invalid_filter": "value"}
        )

@pytest.mark.asyncio
async def test_global_optimizer_instance():
    """Test global query optimizer instance."""
    assert query_optimizer is not None
    
    # Reset stats
    await query_optimizer.reset_stats()
    stats = await query_optimizer.get_optimization_stats()
    
    assert stats["queries_executed"] == 0
    assert stats["avg_query_time"] == 0

@pytest.mark.asyncio
async def test_concurrent_queries(optimizer, db_session, test_data):
    """Test handling concurrent query optimization."""
    import asyncio
    
    query_embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    async def run_query():
        return await optimizer.execute_similarity_query(
            query_embedding,
            db_session
        )
    
    # Run multiple queries concurrently
    tasks = [run_query() for _ in range(5)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert all(len(r) > 0 for r in results)
