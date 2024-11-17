import pytest
import numpy as np
from datetime import datetime
import time
from typing import List

from app.cache.redis_cache import RedisCache, cache_embedding, cache_query
from config.settings import settings

@pytest.fixture
async def redis_cache():
    """Create a Redis cache instance and clear it before/after tests."""
    cache = RedisCache()
    await cache.clear_all()
    yield cache
    await cache.clear_all()

@pytest.mark.asyncio
async def test_embedding_cache(redis_cache):
    """Test embedding caching functionality."""
    text = "Test text for embedding"
    embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    # Test setting embedding
    success = await redis_cache.set_embedding(text, embedding)
    assert success
    
    # Test getting embedding
    cached = await redis_cache.get_embedding(text)
    assert cached is not None
    assert len(cached) == len(embedding)
    assert np.allclose(cached, embedding)
    
    # Test non-existent embedding
    missing = await redis_cache.get_embedding("nonexistent")
    assert missing is None

@pytest.mark.asyncio
async def test_query_cache(redis_cache):
    """Test query result caching."""
    query = "Test query"
    doc_id = 1
    result = {
        "response": "Test response",
        "sources": [{"content": "source", "score": 0.9}]
    }
    
    # Test setting query result
    success = await redis_cache.set_query_result(query, result, doc_id)
    assert success
    
    # Test getting query result
    cached = await redis_cache.get_query_result(query, doc_id)
    assert cached == result
    
    # Test query without document ID
    success = await redis_cache.set_query_result(query, result)
    assert success
    cached = await redis_cache.get_query_result(query)
    assert cached == result

@pytest.mark.asyncio
async def test_document_metadata_cache(redis_cache):
    """Test document metadata caching."""
    doc_id = 1
    metadata = {
        "filename": "test.txt",
        "content_type": "text/plain",
        "created_at": datetime.utcnow().isoformat()
    }
    
    # Test setting metadata
    success = await redis_cache.set_document_metadata(doc_id, metadata)
    assert success
    
    # Test getting metadata
    cached = await redis_cache.get_document_metadata(doc_id)
    assert cached == metadata
    
    # Test non-existent document
    missing = await redis_cache.get_document_metadata(999)
    assert missing is None

@pytest.mark.asyncio
async def test_cache_invalidation(redis_cache):
    """Test cache invalidation functionality."""
    doc_id = 1
    
    # Set up test data
    metadata = {"filename": "test.txt"}
    query1 = "First query"
    query2 = "Second query"
    result = {"response": "Test response"}
    
    await redis_cache.set_document_metadata(doc_id, metadata)
    await redis_cache.set_query_result(query1, result, doc_id)
    await redis_cache.set_query_result(query2, result, doc_id)
    
    # Test invalidation
    success = await redis_cache.invalidate_document(doc_id)
    assert success
    
    # Verify data is invalidated
    assert await redis_cache.get_document_metadata(doc_id) is None
    assert await redis_cache.get_query_result(query1, doc_id) is None
    assert await redis_cache.get_query_result(query2, doc_id) is None

@pytest.mark.asyncio
async def test_cache_ttl(redis_cache):
    """Test time-to-live functionality."""
    # Test with short TTL
    ttl = 1  # 1 second
    
    # Test embedding TTL
    text = "Test text"
    embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    await redis_cache.set_embedding(text, embedding, ttl)
    
    # Test query result TTL
    query = "Test query"
    result = {"response": "Test"}
    await redis_cache.set_query_result(query, result, ttl=ttl)
    
    # Verify data exists
    assert await redis_cache.get_embedding(text) is not None
    assert await redis_cache.get_query_result(query) is not None
    
    # Wait for TTL to expire
    time.sleep(ttl + 1)
    
    # Verify data is expired
    assert await redis_cache.get_embedding(text) is None
    assert await redis_cache.get_query_result(query) is None

@pytest.mark.asyncio
async def test_cache_decorators():
    """Test cache decorators functionality."""
    
    # Test embedding cache decorator
    @cache_embedding(ttl=300)
    async def mock_generate_embedding(text: str) -> List[float]:
        return np.random.rand(settings.VECTOR_DIMENSION).tolist()
    
    # Test query cache decorator
    @cache_query(ttl=300)
    async def mock_query_documents(query: str, doc_id: int = None) -> dict:
        return {"response": f"Response for {query}"}
    
    # Test embedding caching
    text = "Test text"
    embedding1 = await mock_generate_embedding(text)
    embedding2 = await mock_generate_embedding(text)
    assert np.allclose(embedding1, embedding2)
    
    # Test query caching
    query = "Test query"
    result1 = await mock_query_documents(query)
    result2 = await mock_query_documents(query)
    assert result1 == result2

@pytest.mark.asyncio
async def test_cache_clear(redis_cache):
    """Test cache clearing functionality."""
    # Set up test data
    text = "Test text"
    embedding = np.random.rand(settings.VECTOR_DIMENSION).tolist()
    await redis_cache.set_embedding(text, embedding)
    
    query = "Test query"
    result = {"response": "Test"}
    await redis_cache.set_query_result(query, result)
    
    doc_id = 1
    metadata = {"filename": "test.txt"}
    await redis_cache.set_document_metadata(doc_id, metadata)
    
    # Clear cache
    success = await redis_cache.clear_all()
    assert success
    
    # Verify all data is cleared
    assert await redis_cache.get_embedding(text) is None
    assert await redis_cache.get_query_result(query) is None
    assert await redis_cache.get_document_metadata(doc_id) is None

@pytest.mark.asyncio
async def test_error_handling(redis_cache):
    """Test error handling in cache operations."""
    # Test invalid embedding data
    success = await redis_cache.set_embedding("test", "invalid")
    assert not success
    
    # Test invalid JSON data
    success = await redis_cache.set_query_result("test", object())
    assert not success
    
    # Test invalid document metadata
    success = await redis_cache.set_document_metadata(1, object())
    assert not success
