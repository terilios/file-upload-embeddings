import pytest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

from app.rag.context_manager import ContextManager, ContextWindow
from app.cache.redis_cache import RedisCache

@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = Mock(spec=RedisCache)
    cache.get.return_value = None
    return cache

@pytest.fixture
def manager(mock_cache):
    """Create context manager instance."""
    return ContextManager(
        cache=mock_cache,
        min_window_size=2,
        max_window_size=4,
        overlap_size=1,
        relevance_threshold=0.6
    )

@pytest.fixture
def test_chunks():
    """Create test document chunks."""
    return [
        {
            "content": f"Chunk {i}",
            "embedding": np.random.rand(1536).tolist(),
            "metadata": {
                "source_id": i // 2,
                "chunk_id": i
            }
        }
        for i in range(6)
    ]

@pytest.fixture
def test_query_embedding():
    """Create test query embedding."""
    return np.random.rand(1536).tolist()

@pytest.mark.asyncio
async def test_create_context_windows(manager, test_chunks, test_query_embedding):
    """Test context window creation."""
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    
    assert isinstance(windows, list)
    assert all(isinstance(w, ContextWindow) for w in windows)
    assert all(w.relevance_score >= manager.relevance_threshold for w in windows)

@pytest.mark.asyncio
async def test_calculate_chunk_scores(manager, test_chunks, test_query_embedding):
    """Test chunk relevance scoring."""
    scores = await manager._calculate_chunk_scores(
        test_chunks,
        test_query_embedding
    )
    
    assert len(scores) == len(test_chunks)
    assert all(isinstance(s, float) for s in scores)
    assert all(0 <= s <= 1 for s in scores)

@pytest.mark.asyncio
async def test_create_window(manager, test_chunks):
    """Test individual window creation."""
    window_chunks = test_chunks[:2]
    chunk_scores = [0.8, 0.7]
    
    window = await manager._create_window(
        window_chunks,
        chunk_scores,
        0,
        1
    )
    
    assert isinstance(window, ContextWindow)
    assert window.chunks == window_chunks
    assert window.start_idx == 0
    assert window.end_idx == 1
    assert 0 <= window.relevance_score <= 1
    assert "chunk_count" in window.metadata
    assert "avg_chunk_length" in window.metadata
    assert "source_documents" in window.metadata

@pytest.mark.asyncio
async def test_expand_window(manager, test_chunks):
    """Test window expansion."""
    # Create initial window
    initial_window = await manager._create_window(
        test_chunks[:2],
        [0.8, 0.7],
        0,
        1
    )
    
    # Try expanding
    expanded = await manager._expand_window(
        test_chunks,
        [0.8, 0.7, 0.9, 0.6, 0.7, 0.6],
        initial_window
    )
    
    assert isinstance(expanded, ContextWindow)
    assert len(expanded.chunks) <= manager.max_window_size
    assert expanded.start_idx >= 0
    assert expanded.end_idx < len(test_chunks)

@pytest.mark.asyncio
async def test_try_expand(manager, test_chunks):
    """Test expansion in specific direction."""
    initial_window = await manager._create_window(
        test_chunks[:2],
        [0.8, 0.7],
        0,
        1
    )
    
    # Test right expansion
    right_expanded = await manager._try_expand(
        test_chunks,
        [0.8, 0.7, 0.9, 0.6, 0.7, 0.6],
        initial_window,
        "right"
    )
    
    assert right_expanded is not None
    assert right_expanded.end_idx > initial_window.end_idx
    
    # Test left expansion
    left_expanded = await manager._try_expand(
        test_chunks,
        [0.8, 0.7, 0.9, 0.6, 0.7, 0.6],
        initial_window,
        "left"
    )
    
    assert left_expanded is None  # Can't expand left from start

@pytest.mark.asyncio
async def test_merge_windows(manager):
    """Test window merging."""
    # Create overlapping windows
    window1 = ContextWindow(
        chunks=[{"content": "1"}, {"content": "2"}],
        start_idx=0,
        end_idx=1,
        relevance_score=0.8,
        metadata={}
    )
    window2 = ContextWindow(
        chunks=[{"content": "2"}, {"content": "3"}],
        start_idx=1,
        end_idx=2,
        relevance_score=0.7,
        metadata={}
    )
    
    merged = await manager._merge_windows([window1, window2])
    
    assert len(merged) <= len([window1, window2])
    if merged:
        assert merged[0].start_idx == 0
        assert merged[0].end_idx == 2

@pytest.mark.asyncio
async def test_cache_usage(manager, test_chunks, test_query_embedding, mock_cache):
    """Test cache usage."""
    # Test cache miss
    mock_cache.get.return_value = None
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    mock_cache.set.assert_called()
    
    # Test cache hit
    cached_windows = [
        {
            "chunks": [{"content": "test"}],
            "start_idx": 0,
            "end_idx": 0,
            "relevance_score": 0.8,
            "metadata": {}
        }
    ]
    mock_cache.get.return_value = cached_windows
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    assert len(windows) == len(cached_windows)

@pytest.mark.asyncio
async def test_window_size_limits(manager, test_chunks, test_query_embedding):
    """Test window size constraints."""
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    
    for window in windows:
        assert len(window.chunks) >= manager.min_window_size
        assert len(window.chunks) <= manager.max_window_size

@pytest.mark.asyncio
async def test_relevance_threshold(manager, test_chunks, test_query_embedding):
    """Test relevance threshold filtering."""
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    
    assert all(
        window.relevance_score >= manager.relevance_threshold
        for window in windows
    )

@pytest.mark.asyncio
async def test_error_handling(manager):
    """Test error handling."""
    # Test with invalid chunks
    windows = await manager.create_context_windows(
        [],  # Empty chunks
        []   # Invalid embedding
    )
    assert windows == []
    
    # Test with invalid scores
    scores = await manager._calculate_chunk_scores(
        [],  # Empty chunks
        []   # Invalid embedding
    )
    assert scores == []

@pytest.mark.asyncio
async def test_window_metadata(manager, test_chunks):
    """Test window metadata generation."""
    window = await manager._create_window(
        test_chunks[:2],
        [0.8, 0.7],
        0,
        1
    )
    
    assert isinstance(window.metadata, dict)
    assert "chunk_count" in window.metadata
    assert "avg_chunk_length" in window.metadata
    assert "source_documents" in window.metadata
    assert "timestamp" in window.metadata

@pytest.mark.asyncio
async def test_window_sorting(manager, test_chunks, test_query_embedding):
    """Test window sorting by relevance."""
    windows = await manager.create_context_windows(
        test_chunks,
        test_query_embedding
    )
    
    if len(windows) > 1:
        assert all(
            windows[i].relevance_score >= windows[i+1].relevance_score
            for i in range(len(windows)-1)
        )
