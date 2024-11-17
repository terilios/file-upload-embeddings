import pytest
import asyncio
from pathlib import Path
import tempfile
import json
import numpy as np
from datetime import datetime

from app.database.vector_store import VectorStore
from app.document_processing.batch_processor import BatchProcessor
from app.rag.hybrid_searcher import HybridSearcher
from app.rag.document_graph import DocumentGraph
from app.rag.context_manager import ContextManager
from app.rag.query_expander import QueryExpander
from app.cache.redis_cache import RedisCache

class TestScenario:
    """Base class for test scenarios."""
    
    def __init__(self, db_session):
        """Initialize test scenario."""
        self.db_session = db_session
        self.vector_store = VectorStore(db_session)
        self.cache = RedisCache()
        
        # Initialize components
        self.batch_processor = BatchProcessor(self.vector_store)
        self.hybrid_searcher = HybridSearcher(self.vector_store, self.cache)
        self.document_graph = DocumentGraph(self.vector_store, self.cache)
        self.context_manager = ContextManager(self.cache)
        self.query_expander = QueryExpander(self.cache)
        
        # Test data
        self.test_docs = []
        self.test_queries = []
        self.results = {}

@pytest.fixture
async def scenario(db_session):
    """Create test scenario."""
    return TestScenario(db_session)

@pytest.mark.asyncio
async def test_document_processing(scenario):
    """Test complete document processing pipeline."""
    # Create test documents
    test_docs = [
        {
            "content": "The quick brown fox jumps over the lazy dog. See figure 1.",
            "filename": "doc1.txt",
            "content_type": "text/plain",
            "file_size": 100,
            "metadata": {"source": "test"}
        },
        {
            "content": "As shown in doc1.txt, the fox is quick and brown.",
            "filename": "doc2.txt",
            "content_type": "text/plain",
            "file_size": 100,
            "metadata": {"source": "test"}
        },
        {
            "content": """
            def test_function():
                print("Hello, World!")
            """,
            "filename": "test.py",
            "content_type": "text/x-python",
            "file_size": 100,
            "metadata": {"source": "test"}
        }
    ]
    
    # Process documents
    processed_docs = await scenario.batch_processor.process_documents(test_docs)
    
    # Verify processing
    assert len(processed_docs) == len(test_docs)
    for doc in processed_docs:
        assert "id" in doc
        assert "chunk_count" in doc
        assert "metadata" in doc

@pytest.mark.asyncio
async def test_hybrid_search(scenario):
    """Test hybrid search functionality."""
    # Index test documents
    await scenario.hybrid_searcher.index_documents(scenario.test_docs)
    
    # Perform search
    query = "quick brown fox"
    results = await scenario.hybrid_searcher.search(
        query,
        top_k=5,
        threshold=0.5
    )
    
    # Verify results
    assert len(results) > 0
    assert all("score" in r for r in results)
    assert all("document_id" in r for r in results)
    assert all(r["score"] >= 0.5 for r in results)

@pytest.mark.asyncio
async def test_document_graph(scenario):
    """Test document graph functionality."""
    # Build document graph
    await scenario.document_graph.build_graph(scenario.test_docs)
    
    # Get related documents
    related = await scenario.document_graph.get_related_documents(
        document_id=1,
        max_depth=2
    )
    
    # Verify relationships
    assert len(related) > 0
    assert all("document_id" in r for r in related)
    assert all("relationship" in r for r in related)

@pytest.mark.asyncio
async def test_context_management(scenario):
    """Test context window management."""
    # Create test chunks
    chunks = [
        {
            "content": f"Chunk {i}",
            "embedding": np.random.rand(1536).tolist()
        }
        for i in range(10)
    ]
    
    # Create context windows
    query_embedding = np.random.rand(1536).tolist()
    windows = await scenario.context_manager.create_context_windows(
        chunks,
        query_embedding
    )
    
    # Verify windows
    assert len(windows) > 0
    assert all(hasattr(w, 'chunks') for w in windows)
    assert all(hasattr(w, 'relevance_score') for w in windows)

@pytest.mark.asyncio
async def test_query_expansion(scenario):
    """Test query expansion functionality."""
    # Create test context
    context = [
        {
            "content": "The system implements efficient database optimization",
            "metadata": {"type": "technical"}
        }
    ]
    
    # Expand query
    query = "database performance"
    expanded = await scenario.query_expander.expand_query(
        query,
        context
    )
    
    # Verify expansion
    assert expanded["original_query"] == query
    assert len(expanded["expanded_query"]) > len(query)
    assert len(expanded["semantic_terms"]) > 0

@pytest.mark.asyncio
async def test_complete_search_flow(scenario):
    """Test complete search flow with all features."""
    # Process test documents
    processed_docs = await scenario.batch_processor.process_documents(
        scenario.test_docs
    )
    
    # Build document graph
    await scenario.document_graph.build_graph(processed_docs)
    
    # Index for hybrid search
    await scenario.hybrid_searcher.index_documents(processed_docs)
    
    # Perform search with query expansion
    query = "quick fox database"
    expanded = await scenario.query_expander.expand_query(query)
    
    # Search with expanded query
    results = await scenario.hybrid_searcher.search(
        expanded["expanded_query"],
        top_k=5
    )
    
    # Get related documents
    if results:
        related = await scenario.document_graph.get_related_documents(
            results[0]["document_id"]
        )
        
        # Create context windows
        chunks = [doc["chunks"] for doc in processed_docs]
        flat_chunks = [chunk for doc_chunks in chunks for chunk in doc_chunks]
        windows = await scenario.context_manager.create_context_windows(
            flat_chunks,
            np.random.rand(1536).tolist()  # Mock query embedding
        )
        
        # Verify complete flow
        assert len(results) > 0
        assert len(related) > 0
        assert len(windows) > 0

@pytest.mark.asyncio
async def test_error_handling(scenario):
    """Test error handling in integration scenarios."""
    # Test with invalid document
    invalid_docs = [
        {
            "content": None,
            "filename": "invalid.txt",
            "content_type": "text/plain",
            "file_size": 0
        }
    ]
    
    # Process should not fail but return empty results
    processed = await scenario.batch_processor.process_documents(invalid_docs)
    assert len(processed) == 0
    
    # Test with invalid query
    results = await scenario.hybrid_searcher.search(
        "",  # Empty query
        top_k=5
    )
    assert len(results) == 0

@pytest.mark.asyncio
async def test_cache_integration(scenario):
    """Test caching across components."""
    # Perform operations that should use cache
    query = "test query"
    
    # First call should miss cache
    result1 = await scenario.query_expander.expand_query(query)
    
    # Second call should hit cache
    result2 = await scenario.query_expander.expand_query(query)
    
    assert result1 == result2

@pytest.mark.asyncio
async def test_concurrent_operations(scenario):
    """Test concurrent operations."""
    # Create multiple concurrent tasks
    tasks = []
    for i in range(5):
        tasks.append(
            scenario.hybrid_searcher.search(
                f"query {i}",
                top_k=5
            )
        )
    
    # Execute concurrently
    results = await asyncio.gather(*tasks)
    
    # Verify all tasks completed
    assert len(results) == 5
    assert all(isinstance(r, list) for r in results)
