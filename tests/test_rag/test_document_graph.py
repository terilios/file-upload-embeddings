import pytest
from unittest.mock import Mock, patch
import networkx as nx
from datetime import datetime

from app.rag.document_graph import DocumentGraph
from app.database.vector_store import VectorStore
from app.cache.redis_cache import RedisCache

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    return Mock(spec=VectorStore)

@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = Mock(spec=RedisCache)
    cache.get.return_value = None
    return cache

@pytest.fixture
def graph(mock_vector_store, mock_cache):
    """Create document graph instance."""
    return DocumentGraph(
        vector_store=mock_vector_store,
        cache=mock_cache
    )

@pytest.fixture
def test_documents():
    """Create test documents with references."""
    return [
        {
            "id": 1,
            "filename": "doc1.txt",
            "content": "See 'doc2.txt' for more details. Also check https://example.com",
            "content_type": "text/plain",
            "metadata": {"url": "https://example.com"}
        },
        {
            "id": 2,
            "filename": "doc2.txt",
            "content": "As shown in section 1.2 of doc1.txt. [1]",
            "content_type": "text/plain",
            "metadata": {"section": "1.2"}
        },
        {
            "id": 3,
            "filename": "doc3.txt",
            "content": "Reference (Smith et al., 2023)",
            "content_type": "text/plain",
            "metadata": {"citation": "Smith et al., 2023"}
        }
    ]

@pytest.mark.asyncio
async def test_build_graph(graph, test_documents):
    """Test building document reference graph."""
    await graph.build_graph(test_documents)
    
    assert graph.graph.number_of_nodes() == 3
    assert graph.graph.number_of_nodes() > 0
    
    # Check node attributes
    for doc in test_documents:
        assert doc["id"] in graph.graph
        node_data = graph.graph.nodes[doc["id"]]
        assert "title" in node_data
        assert "type" in node_data
        assert "metadata" in node_data

@pytest.mark.asyncio
async def test_find_references(graph, test_documents):
    """Test reference finding in documents."""
    await graph.build_graph(test_documents)
    
    references = await graph._find_references(test_documents[0])
    
    assert "url" in references
    assert "file" in references
    assert "section" in references
    assert "citation" in references
    assert len(references["file"]) > 0  # Should find doc2.txt reference

@pytest.mark.asyncio
async def test_get_related_documents(graph, test_documents):
    """Test getting related documents."""
    await graph.build_graph(test_documents)
    
    related = await graph.get_related_documents(1, max_depth=2)
    
    assert len(related) > 0
    for doc in related:
        assert "document_id" in doc
        assert "title" in doc
        assert "type" in doc
        assert "distance" in doc
        assert "relationship" in doc

@pytest.mark.asyncio
async def test_find_common_references(graph, test_documents):
    """Test finding common references."""
    await graph.build_graph(test_documents)
    
    common_refs = await graph.find_common_references([1, 2])
    
    assert isinstance(common_refs, list)
    for ref in common_refs:
        assert "document_id" in ref
        assert "reference_count" in ref
        assert "reference_types" in ref

@pytest.mark.asyncio
async def test_get_citation_graph(graph, test_documents):
    """Test getting citation graph for visualization."""
    await graph.build_graph(test_documents)
    
    citation_graph = await graph.get_citation_graph(1)
    
    assert "nodes" in citation_graph
    assert "edges" in citation_graph
    assert len(citation_graph["nodes"]) > 0
    assert all("id" in node for node in citation_graph["nodes"])
    assert all("from" in edge for edge in citation_graph["edges"])

@pytest.mark.asyncio
async def test_cache_usage(graph, mock_cache, test_documents):
    """Test cache usage."""
    # Test cache miss
    mock_cache.get.return_value = None
    await graph.get_related_documents(1)
    mock_cache.set.assert_called()
    
    # Test cache hit
    cached_result = [{"document_id": 2}]
    mock_cache.get.return_value = cached_result
    result = await graph.get_related_documents(1)
    assert result == cached_result

def test_reference_patterns(graph):
    """Test reference pattern matching."""
    # Test URL pattern
    assert re.match(graph.patterns["url"], "https://example.com")
    assert re.match(graph.patterns["url"], "www.example.com")
    
    # Test file pattern
    assert re.search(graph.patterns["file"], 'see "document.pdf"')
    assert re.search(graph.patterns["file"], "refer to 'test.txt'")
    
    # Test section pattern
    assert re.search(graph.patterns["section"], "see section 1.2.3")
    assert re.search(graph.patterns["section"], "described in chapter 2")
    
    # Test citation pattern
    assert re.search(graph.patterns["citation"], "[1]")
    assert re.search(graph.patterns["citation"], "(Smith et al., 2023)")

@pytest.mark.asyncio
async def test_find_document_by_filename(graph, test_documents):
    """Test finding document by filename."""
    await graph.build_graph(test_documents)
    
    doc_id = await graph._find_document_by_filename("doc1.txt")
    assert doc_id == 1
    
    doc_id = await graph._find_document_by_filename("nonexistent.txt")
    assert doc_id is None

@pytest.mark.asyncio
async def test_find_document_by_metadata(graph, test_documents):
    """Test finding document by metadata."""
    await graph.build_graph(test_documents)
    
    doc_id = await graph._find_document_by_metadata(
        "url",
        "https://example.com"
    )
    assert doc_id == 1
    
    doc_id = await graph._find_document_by_metadata(
        "nonexistent",
        "value"
    )
    assert doc_id is None

@pytest.mark.asyncio
async def test_error_handling(graph):
    """Test error handling."""
    # Test with invalid document ID
    result = await graph.get_related_documents(999)
    assert result == []
    
    # Test with empty graph
    result = await graph.find_common_references([1, 2])
    assert result == []
    
    # Test with invalid document in common references
    result = await graph.find_common_references([999])
    assert result == []

@pytest.mark.asyncio
async def test_graph_persistence(graph, test_documents, mock_cache):
    """Test graph structure caching."""
    await graph.build_graph(test_documents)
    
    # Verify graph was cached
    mock_cache.set.assert_called_with(
        "document_graph",
        pytest.approx(any_dict()),
        ttl=any_int()
    )

def test_relationship_metadata(graph, test_documents):
    """Test relationship metadata in graph."""
    graph.build_graph(test_documents)
    
    for _, _, data in graph.graph.edges(data=True):
        assert "type" in data
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

@pytest.mark.asyncio
async def test_max_depth_limit(graph, test_documents):
    """Test max depth limit in related documents."""
    await graph.build_graph(test_documents)
    
    # Test with different depths
    depth1 = await graph.get_related_documents(1, max_depth=1)
    depth2 = await graph.get_related_documents(1, max_depth=2)
    
    # Deeper search should find same or more documents
    assert len(depth1) <= len(depth2)

@pytest.mark.asyncio
async def test_reference_type_filtering(graph, test_documents):
    """Test filtering by reference types."""
    await graph.build_graph(test_documents)
    
    # Get related documents with specific reference types
    related = await graph.get_related_documents(
        1,
        ref_types=["file"]
    )
    
    # Verify only specified reference types are included
    for doc in related:
        assert all(
            rel["type"] == "file"
            for rel in doc["relationship"]
        )
