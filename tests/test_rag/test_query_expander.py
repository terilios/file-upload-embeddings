import pytest
from unittest.mock import Mock, patch
import spacy
import numpy as np

from app.rag.query_expander import QueryExpander
from app.cache.redis_cache import RedisCache

@pytest.fixture
def mock_cache():
    """Create mock Redis cache."""
    cache = Mock(spec=RedisCache)
    cache.get.return_value = None
    return cache

@pytest.fixture
def expander(mock_cache):
    """Create query expander instance."""
    return QueryExpander(
        cache=mock_cache,
        max_synonyms=3,
        max_semantic_terms=5,
        similarity_threshold=0.7
    )

@pytest.fixture
def test_context():
    """Create test context documents."""
    return [
        {
            "content": "The algorithm implements efficient database optimization",
            "metadata": {"type": "technical"}
        },
        {
            "content": "System architecture and deployment strategy",
            "metadata": {"type": "technical"}
        }
    ]

@pytest.mark.asyncio
async def test_expand_query(expander):
    """Test complete query expansion."""
    query = "database optimization techniques"
    
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = np.random.rand(1536).tolist()
        
        result = await expander.expand_query(query)
        
        assert "original_query" in result
        assert "expanded_query" in result
        assert "terms" in result
        assert "synonyms" in result
        assert "semantic_terms" in result
        assert "metadata" in result
        assert result["original_query"] == query
        assert len(result["expanded_query"]) >= len(query)

@pytest.mark.asyncio
async def test_extract_terms(expander):
    """Test term extraction from query."""
    doc = expander.nlp("optimize database performance")
    terms = await expander._extract_terms(doc)
    
    assert isinstance(terms, list)
    assert all(isinstance(term, str) for term in terms)
    assert "optimize" in terms or "database" in terms
    assert not any(term in expander.custom_stopwords for term in terms)

@pytest.mark.asyncio
async def test_get_synonyms(expander):
    """Test synonym generation."""
    terms = ["optimize", "database"]
    synonyms = await expander._get_synonyms(terms)
    
    assert isinstance(synonyms, dict)
    for term, term_synonyms in synonyms.items():
        assert isinstance(term_synonyms, list)
        assert len(term_synonyms) <= expander.max_synonyms
        assert term not in term_synonyms  # Original term not in synonyms

@pytest.mark.asyncio
async def test_get_semantic_terms(expander, test_context):
    """Test semantic term generation."""
    query = "database optimization"
    terms = ["database", "optimization"]
    
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = np.random.rand(1536).tolist()
        
        semantic_terms = await expander._get_semantic_terms(
            query,
            terms,
            test_context
        )
        
        assert isinstance(semantic_terms, list)
        assert len(semantic_terms) <= expander.max_semantic_terms
        assert all(term not in terms for term in semantic_terms)

@pytest.mark.asyncio
async def test_combine_expansions(expander):
    """Test combining different types of expansions."""
    query = "optimize database"
    terms = ["optimize", "database"]
    synonyms = {"optimize": ["improve", "enhance"]}
    semantic_terms = ["performance", "efficiency"]
    
    result = await expander._combine_expansions(
        query,
        terms,
        synonyms,
        semantic_terms
    )
    
    assert result["original_query"] == query
    assert len(result["expanded_query"]) > len(query)
    assert "metadata" in result
    assert "expansion_count" in result["metadata"]

@pytest.mark.asyncio
async def test_cache_usage(expander, mock_cache):
    """Test cache usage."""
    query = "test query"
    
    # Test cache miss
    mock_cache.get.return_value = None
    with patch('app.document_processing.embeddings.generate_embeddings') as mock_embed:
        mock_embed.return_value = np.random.rand(1536).tolist()
        result = await expander.expand_query(query)
        mock_cache.set.assert_called()
    
    # Test cache hit
    cached_result = {
        "original_query": query,
        "expanded_query": "test query enhanced",
        "terms": ["test", "query"],
        "synonyms": {},
        "semantic_terms": ["enhanced"],
        "metadata": {}
    }
    mock_cache.get.return_value = cached_result
    result = await expander.expand_query(query)
    assert result == cached_result

def test_domain_terms(expander):
    """Test domain-specific terms."""
    domain_terms = expander._get_domain_terms()
    
    assert isinstance(domain_terms, set)
    assert len(domain_terms) > 0
    assert "algorithm" in domain_terms
    assert "database" in domain_terms
    assert not any(term in expander.custom_stopwords for term in domain_terms)

@pytest.mark.asyncio
async def test_error_handling(expander):
    """Test error handling in query expansion."""
    # Test with empty query
    result = await expander.expand_query("")
    assert result["expanded_query"] == ""
    assert len(result["terms"]) == 0
    
    # Test with None query
    result = await expander.expand_query(None)
    assert result["expanded_query"] == None
    assert len(result["terms"]) == 0

@pytest.mark.asyncio
async def test_context_influence(expander, test_context):
    """Test influence of context on expansion."""
    query = "system optimization"
    
    # Expansion without context
    result1 = await expander.expand_query(query)
    
    # Expansion with context
    result2 = await expander.expand_query(query, test_context)
    
    # Context should influence semantic terms
    assert result1["semantic_terms"] != result2["semantic_terms"]

@pytest.mark.asyncio
async def test_expansion_limits(expander):
    """Test expansion limits are respected."""
    query = "optimize database system architecture deployment"
    
    result = await expander.expand_query(query)
    
    # Check synonym limits
    for term_synonyms in result["synonyms"].values():
        assert len(term_synonyms) <= expander.max_synonyms
    
    # Check semantic term limits
    assert len(result["semantic_terms"]) <= expander.max_semantic_terms

@pytest.mark.asyncio
async def test_metadata_generation(expander):
    """Test metadata generation in expansions."""
    query = "database optimization"
    
    result = await expander.expand_query(query)
    
    assert "metadata" in result
    assert "expansion_count" in result["metadata"]
    assert "synonym_count" in result["metadata"]
    assert "semantic_term_count" in result["metadata"]
    assert "timestamp" in result["metadata"]

@pytest.mark.asyncio
async def test_stopword_handling(expander):
    """Test handling of stopwords in expansion."""
    query = "the database and the system"
    
    result = await expander.expand_query(query)
    
    # Stopwords should not be in extracted terms
    assert "the" not in result["terms"]
    assert "and" not in result["terms"]
    
    # Stopwords should not have synonyms
    assert not any(term in expander.custom_stopwords for term in result["synonyms"])
