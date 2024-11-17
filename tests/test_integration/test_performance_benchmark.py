import pytest
from unittest.mock import Mock, patch
import numpy as np
from datetime import datetime

from .performance_benchmark import PerformanceBenchmark
from app.database.vector_store import VectorStore
from app.cache.redis_cache import RedisCache

@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    return Mock()

@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = Mock(spec=RedisCache)
    cache.get.return_value = None
    return cache

@pytest.fixture
def benchmark(mock_db_session, mock_cache):
    """Create performance benchmark instance."""
    with patch('app.cache.redis_cache.RedisCache', return_value=mock_cache):
        return PerformanceBenchmark(mock_db_session)

@pytest.mark.asyncio
async def test_run_benchmarks(benchmark):
    """Test complete benchmark suite execution."""
    # Run benchmarks with small test data
    results = await benchmark.run_benchmarks(
        doc_count=10,
        query_count=5,
        concurrent_users=2
    )
    
    # Verify results structure
    assert "results" in results
    assert "summary" in results
    assert "metadata" in results
    
    # Verify all benchmark categories
    assert all(
        category in results["results"]
        for category in [
            "document_processing",
            "search_performance",
            "graph_operations",
            "context_windows",
            "query_expansion",
            "cache_performance",
            "concurrent_operations"
        ]
    )
    
    # Verify metadata
    assert results["metadata"]["doc_count"] == 10
    assert results["metadata"]["query_count"] == 5
    assert results["metadata"]["concurrent_users"] == 2
    assert "duration" in results["metadata"]
    assert "timestamp" in results["metadata"]

@pytest.mark.asyncio
async def test_document_processing_benchmark(benchmark):
    """Test document processing benchmarks."""
    docs = benchmark.data_generator.generate_documents(count=5)
    
    await benchmark._benchmark_document_processing(docs)
    results = benchmark.results["document_processing"]
    
    assert len(results) > 0
    for result in results:
        assert "batch_size" in result
        assert "duration" in result
        assert "docs_per_second" in result
        assert result["docs_per_second"] > 0

@pytest.mark.asyncio
async def test_search_performance_benchmark(benchmark):
    """Test search performance benchmarks."""
    queries = benchmark.data_generator.generate_queries(count=5)
    
    with patch('app.rag.hybrid_searcher.HybridSearcher.search') as mock_search:
        mock_search.return_value = [{"id": 1, "score": 0.9}]
        
        await benchmark._benchmark_search_performance(queries)
        results = benchmark.results["search_performance"]
        
        assert len(results) > 0
        for result in results:
            assert "query_type" in result
            assert "duration" in result
            assert "result_count" in result

@pytest.mark.asyncio
async def test_graph_operations_benchmark(benchmark):
    """Test graph operations benchmarks."""
    docs = benchmark.data_generator.generate_documents(count=5)
    
    await benchmark._benchmark_graph_operations(docs)
    results = benchmark.results["graph_operations"]
    
    assert len(results) > 0
    for result in results:
        assert "doc_count" in result
        assert "build_duration" in result
        assert "avg_traversal_time" in result

@pytest.mark.asyncio
async def test_context_windows_benchmark(benchmark):
    """Test context window benchmarks."""
    docs = benchmark.data_generator.generate_documents(count=5)
    queries = benchmark.data_generator.generate_queries(count=5)
    
    await benchmark._benchmark_context_windows(docs, queries)
    results = benchmark.results["context_windows"]
    
    assert len(results) > 0
    for result in results:
        assert "chunk_count" in result
        assert "window_count" in result
        assert "duration" in result

@pytest.mark.asyncio
async def test_query_expansion_benchmark(benchmark):
    """Test query expansion benchmarks."""
    queries = benchmark.data_generator.generate_queries(count=5)
    
    await benchmark._benchmark_query_expansion(queries)
    results = benchmark.results["query_expansion"]
    
    assert len(results) > 0
    for result in results:
        assert "query_type" in result
        assert "duration" in result
        assert "expansion_count" in result

@pytest.mark.asyncio
async def test_cache_performance_benchmark(benchmark, mock_cache):
    """Test cache performance benchmarks."""
    queries = benchmark.data_generator.generate_queries(count=5)
    
    await benchmark._benchmark_cache_performance(queries)
    results = benchmark.results["cache_performance"]
    
    assert len(results) > 0
    for result in results:
        assert "avg_miss_time" in result
        assert "avg_hit_time" in result
        assert "speedup_factor" in result
        assert result["speedup_factor"] > 1  # Hits should be faster than misses

@pytest.mark.asyncio
async def test_concurrent_operations_benchmark(benchmark):
    """Test concurrent operations benchmarks."""
    docs = benchmark.data_generator.generate_documents(count=5)
    queries = benchmark.data_generator.generate_queries(count=5)
    
    await benchmark._benchmark_concurrent_operations(docs, queries, 3)
    results = benchmark.results["concurrent_operations"]
    
    assert len(results) > 0
    for result in results:
        assert "concurrent_users" in result
        assert "avg_duration" in result
        assert "max_duration" in result
        assert "min_duration" in result
        assert result["max_duration"] >= result["avg_duration"]
        assert result["min_duration"] <= result["avg_duration"]

def test_generate_summary(benchmark):
    """Test summary generation."""
    # Add some test results
    benchmark.results = {
        "document_processing": [{"docs_per_second": 10}, {"docs_per_second": 20}],
        "search_performance": [{"duration": 0.1}, {"duration": 0.2}],
        "graph_operations": [{"avg_traversal_time": 0.1}, {"avg_traversal_time": 0.2}],
        "context_windows": [{"duration": 0.1}, {"duration": 0.2}],
        "query_expansion": [{"duration": 0.1}, {"duration": 0.2}],
        "cache_performance": [{"speedup_factor": 2}, {"speedup_factor": 3}],
        "concurrent_operations": [{"avg_duration": 0.1}, {"avg_duration": 0.2}]
    }
    
    summary = benchmark._generate_summary()
    
    # Verify summary structure
    assert "document_processing" in summary
    assert "search_performance" in summary
    assert "graph_operations" in summary
    assert "context_windows" in summary
    assert "query_expansion" in summary
    assert "cache_performance" in summary
    assert "concurrent_operations" in summary
    
    # Verify averages
    assert summary["document_processing"]["avg_docs_per_second"] == 15
    assert summary["cache_performance"]["avg_speedup"] == 2.5

@pytest.mark.asyncio
async def test_error_handling(benchmark):
    """Test error handling in benchmarks."""
    # Test with invalid inputs
    with pytest.raises(Exception):
        await benchmark.run_benchmarks(doc_count=0)
    
    with pytest.raises(Exception):
        await benchmark.run_benchmarks(query_count=0)
    
    with pytest.raises(Exception):
        await benchmark.run_benchmarks(concurrent_users=0)

@pytest.mark.asyncio
async def test_benchmark_runner(mock_db_session):
    """Test benchmark runner function."""
    from .performance_benchmark import run_benchmarks
    
    results = await run_benchmarks(
        mock_db_session,
        doc_count=5,
        query_count=3,
        concurrent_users=2
    )
    
    assert isinstance(results, dict)
    assert "results" in results
    assert "summary" in results
    assert "metadata" in results
