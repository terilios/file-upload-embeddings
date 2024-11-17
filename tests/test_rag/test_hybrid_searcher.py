import pytest
from unittest.mock import Mock, patch
import numpy as np

from app.rag.hybrid_searcher import HybridSearcher
from app.database.vector_store import VectorStore
from app.cache.redis_cache import RedisCache

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock(spec=VectorStore)
    store.similarity_search.return_value = [
        (Mock(document_id=1), 0.8),
        (Mock(document_id=2), 0.6)
    ]
    return store

@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = Mock(spec=RedisCache)
    cache.get.return_value = None
    return cache

@pytest.fixture
def searcher(mock_vector_store, mock_cache):
    """Create hybrid searcher instance."""
    return HybridSearcher(
        vector_store=mock_vector_store,
        cache=mock_cache
    )

@pytest.fixture
def test_documents():
    """Create test documents."""
    return [
        {
            "id": 1,
            "content": "The quick brown fox jumps over the lazy dog"
        },
        {
            "id": 2,
            "content": "A quick brown dog jumps over the lazy fox"
        },
        {
            "id": 3,
            "content": "The lazy fox sleeps while the quick brown dog watches"
        }
    ]

@pytest.mark.asyncio
async def test_index_documents(searcher, test_documents):
    """Test document indexing for BM25."""
    await searcher.index_documents(test_documents)
    
    assert searcher.bm25 is not None
    assert len(searcher.documents) == len(test_documents)
    assert len(searcher.doc_ids) == len(test_documents)

@pytest.mark.asyncio
async def test_get_bm25_scores(searcher, test_documents):
    """Test BM25 scoring."""
    await searcher.index_documents(test_documents)
    
    scores = await searcher._get_bm25_scores("quick fox", top_k=2)
    
    assert len(scores) > 0
    assert all(isinstance(score[1], float) for score in scores)
    assert all(score[1] >= 0 for score in scores)

@pytest.mark.asyncio
async def test_get_vector_scores(searcher):
    """Test vector similarity scoring."""
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        
        scores = await searcher._get_vector_scores(
            "test query",
            top_k=2,
            threshold=0.5,
            filters=None
        )
        
        assert len(scores) > 0
        assert all(isinstance(score[1], float) for score in scores)

def test_combine_scores(searcher):
    """Test score combination."""
    bm25_scores = [(1, 0.8), (2, 0.6)]
    vector_scores = [(1, 0.7), (3, 0.5)]
    
    combined = searcher._combine_scores(bm25_scores, vector_scores, top_k=2)
    
    assert len(combined) <= 2
    assert all("score" in result for result in combined)
    assert all("bm25_score" in result for result in combined)
    assert all("vector_score" in result for result in combined)

@pytest.mark.asyncio
async def test_search(searcher, test_documents):
    """Test complete hybrid search."""
    await searcher.index_documents(test_documents)
    
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        
        results = await searcher.search(
            "quick fox",
            top_k=2,
            threshold=0.5
        )
        
        assert len(results) <= 2
        assert all("score" in result for result in results)
        assert all("document_id" in result for result in results)

@pytest.mark.asyncio
async def test_search_with_cache(searcher, mock_cache):
    """Test search with cache hit."""
    cached_results = [
        {"document_id": 1, "score": 0.9},
        {"document_id": 2, "score": 0.7}
    ]
    mock_cache.get.return_value = cached_results
    
    results = await searcher.search("test query")
    
    assert results == cached_results
    mock_cache.get.assert_called_once()

@pytest.mark.asyncio
async def test_search_with_filters(searcher):
    """Test search with filters."""
    filters = {"content_type": "text"}
    
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = [0.1] * 1536
        
        await searcher.search(
            "test query",
            filters=filters
        )
        
        searcher.vector_store.similarity_search.assert_called_with(
            mock_embed.return_value,
            top_k=10,
            threshold=None,
            filters=filters
        )

def test_update_weights(searcher):
    """Test weight updating."""
    new_bm25 = 0.4
    new_vector = 0.6
    
    searcher.update_weights(new_bm25, new_vector)
    
    assert searcher.bm25_weight == new_bm25
    assert searcher.vector_weight == new_vector

def test_update_weights_validation(searcher):
    """Test weight validation."""
    with pytest.raises(ValueError):
        searcher.update_weights(0.5, 0.6)  # Sum > 1
    
    with pytest.raises(ValueError):
        searcher.update_weights(-0.1, 1.1)  # Invalid weights

@pytest.mark.asyncio
async def test_empty_results(searcher):
    """Test handling of empty search results."""
    # Mock empty results
    searcher.vector_store.similarity_search.return_value = []
    await searcher.index_documents([])  # Empty BM25 index
    
    results = await searcher.search("test query")
    
    assert isinstance(results, list)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_error_handling(searcher):
    """Test error handling in search."""
    # Mock error in vector search
    searcher.vector_store.similarity_search.side_effect = Exception("Vector search error")
    
    with pytest.raises(Exception):
        await searcher.search("test query")

@pytest.mark.asyncio
async def test_score_normalization(searcher, test_documents):
    """Test score normalization in combination."""
    await searcher.index_documents(test_documents)
    
    bm25_scores = [(1, 10.0), (2, 5.0)]  # Large scores
    vector_scores = [(1, 0.9), (2, 0.5)]  # Small scores
    
    combined = searcher._combine_scores(bm25_scores, vector_scores, top_k=2)
    
    assert all(0 <= result["score"] <= 1 for result in combined)
    assert all(0 <= result["bm25_score"] <= 1 for result in combined)
    assert all(0 <= result["vector_score"] <= 1 for result in combined)
