import pytest
from datetime import datetime
import re

from .test_data_generator import TestDataGenerator

@pytest.fixture
def generator():
    """Create test data generator instance."""
    return TestDataGenerator()

def test_generate_documents(generator):
    """Test document generation."""
    # Generate documents with default settings
    docs = generator.generate_documents()
    
    assert len(docs) == 10  # Default count
    for doc in docs:
        assert "id" in doc
        assert "content" in doc
        assert "filename" in doc
        assert "content_type" in doc
        assert "file_size" in doc
        assert "metadata" in doc
        
        # Verify metadata
        assert "type" in doc["metadata"]
        assert "created_at" in doc["metadata"]
        assert "version" in doc["metadata"]
        
        # Verify content is non-empty
        assert len(doc["content"]) > 0
        assert doc["file_size"] == len(doc["content"])

def test_generate_code_documents(generator):
    """Test code document generation."""
    # Generate documents with high code probability
    docs = generator.generate_documents(count=10, include_code=True)
    
    code_docs = [
        doc for doc in docs
        if doc["metadata"]["type"] == "code"
    ]
    
    assert len(code_docs) > 0
    for doc in code_docs:
        assert doc["content_type"] == "text/x-python"
        assert doc["filename"].endswith(".py")
        assert "def " in doc["content"] or "class " in doc["content"]
        assert doc["metadata"]["language"] == "python"

def test_generate_technical_documents(generator):
    """Test technical document generation."""
    # Generate only technical documents
    docs = generator.generate_documents(count=5, include_code=False)
    
    assert len(docs) == 5
    for doc in docs:
        assert doc["metadata"]["type"] == "technical"
        assert doc["content_type"] == "text/plain"
        assert doc["filename"].endswith(".txt")

def test_document_references(generator):
    """Test document cross-references."""
    # Generate documents with references
    docs = generator.generate_documents(
        count=5,
        include_references=True
    )
    
    # Check for references in content
    reference_pattern = r'(See document_\d+|Reference: \[doc_\d+\]|As described in doc_\d+)'
    references_found = False
    
    for doc in docs:
        if re.search(reference_pattern, doc["content"]):
            references_found = True
            break
    
    assert references_found

def test_generate_queries(generator):
    """Test query generation."""
    # Test different complexity levels
    simple_queries = generator.generate_queries(
        count=3,
        complexity="simple"
    )
    medium_queries = generator.generate_queries(
        count=3,
        complexity="medium"
    )
    complex_queries = generator.generate_queries(
        count=3,
        complexity="complex"
    )
    
    # Verify simple queries
    for query in simple_queries:
        assert query["type"] == "simple"
        assert len(query["expected_terms"]) == 1
        assert query["metadata"]["complexity"] == "simple"
    
    # Verify medium queries
    for query in medium_queries:
        assert query["type"] == "medium"
        assert len(query["expected_terms"]) == 2
        assert query["metadata"]["complexity"] == "medium"
    
    # Verify complex queries
    for query in complex_queries:
        assert query["type"] == "complex"
        assert len(query["expected_terms"]) >= 3
        assert query["metadata"]["complexity"] == "complex"

def test_query_content(generator):
    """Test query content generation."""
    queries = generator.generate_queries(count=10)
    
    for query in queries:
        # Verify query structure
        assert "query" in query
        assert "type" in query
        assert "expected_terms" in query
        assert "metadata" in query
        
        # Verify query content
        assert len(query["query"]) > 0
        assert all(
            term in query["query"].lower()
            for term in query["expected_terms"]
        )

def test_date_generation(generator):
    """Test random date generation."""
    date_str = generator._random_date()
    
    # Verify date format
    date = datetime.fromisoformat(date_str)
    assert isinstance(date, datetime)
    
    # Verify date is within last year
    days_old = (datetime.now() - date).days
    assert 0 <= days_old <= 365

def test_content_templates(generator):
    """Test content template usage."""
    # Verify technical templates
    assert len(generator.technical_templates) > 0
    for template in generator.technical_templates:
        assert "{" in template and "}" in template
    
    # Verify code templates
    assert len(generator.code_templates) > 0
    for template in generator.code_templates:
        assert "{" in template and "}" in template

def test_domain_terms(generator):
    """Test domain-specific terms."""
    # Verify term categories
    assert len(generator.technologies) > 0
    assert len(generator.algorithms) > 0
    assert len(generator.components) > 0
    assert len(generator.metrics) > 0
    
    # Verify term uniqueness
    assert len(set(generator.technologies)) == len(generator.technologies)
    assert len(set(generator.algorithms)) == len(generator.algorithms)
    assert len(set(generator.components)) == len(generator.components)
    assert len(set(generator.metrics)) == len(generator.metrics)

def test_document_uniqueness(generator):
    """Test document uniqueness."""
    docs = generator.generate_documents(count=10)
    
    # Verify unique IDs
    ids = [doc["id"] for doc in docs]
    assert len(set(ids)) == len(docs)
    
    # Verify unique content
    contents = [doc["content"] for doc in docs]
    assert len(set(contents)) == len(docs)

def test_query_uniqueness(generator):
    """Test query uniqueness."""
    queries = generator.generate_queries(count=10)
    
    # Verify unique queries
    query_texts = [q["query"] for q in queries]
    assert len(set(query_texts)) == len(queries)

def test_file_size_accuracy(generator):
    """Test file size calculation."""
    docs = generator.generate_documents(count=5)
    
    for doc in docs:
        assert doc["file_size"] == len(doc["content"].encode('utf-8'))
