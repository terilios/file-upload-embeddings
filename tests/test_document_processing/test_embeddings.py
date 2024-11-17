import pytest
import numpy as np
from typing import List
import openai

from app.document_processing.embeddings import (
    count_tokens,
    truncate_text,
    generate_embeddings,
    batch_generate_embeddings,
    compute_similarity,
    average_embeddings,
    generate_hybrid_embedding
)
from config.settings import settings

def test_count_tokens():
    """Test token counting functionality."""
    # Test simple text
    text = "This is a test."
    tokens = count_tokens(text)
    assert tokens > 0
    
    # Test empty text
    assert count_tokens("") == 0
    
    # Test long text
    long_text = "Test " * 1000
    tokens = count_tokens(long_text)
    assert tokens > 100
    
    # Test special characters
    special_text = "Special chars: !@#$%^&*()\nNew lines\tTabs"
    tokens = count_tokens(special_text)
    assert tokens > 0
    
    # Test Unicode
    unicode_text = "Hello 你好 World 世界"
    tokens = count_tokens(unicode_text)
    assert tokens > 0

def test_truncate_text():
    """Test text truncation functionality."""
    # Test text within limit
    short_text = "This is a short text."
    truncated = truncate_text(short_text, max_tokens=100)
    assert truncated == short_text
    
    # Test text exceeding limit
    long_text = "Test " * 1000
    truncated = truncate_text(long_text, max_tokens=100)
    assert len(truncated) < len(long_text)
    assert count_tokens(truncated) <= 100
    
    # Test empty text
    assert truncate_text("", max_tokens=100) == ""
    
    # Test exact limit
    text = "This is a test."
    tokens = count_tokens(text)
    truncated = truncate_text(text, max_tokens=tokens)
    assert truncated == text

def test_generate_embeddings(mock_openai):
    """Test embedding generation."""
    # Test simple text
    text = "This is a test document."
    embedding = generate_embeddings(text)
    assert len(embedding) == settings.VECTOR_DIMENSION
    assert all(isinstance(x, float) for x in embedding)
    
    # Test empty text
    with pytest.raises(Exception):
        generate_embeddings("")
    
    # Test long text
    long_text = "Test " * 1000
    embedding = generate_embeddings(long_text)
    assert len(embedding) == settings.VECTOR_DIMENSION

def test_batch_generate_embeddings(mock_openai):
    """Test batch embedding generation."""
    texts = [
        "First document",
        "Second document",
        "Third document"
    ]
    
    embeddings = batch_generate_embeddings(texts)
    
    assert len(embeddings) == len(texts)
    assert all(len(emb) == settings.VECTOR_DIMENSION for emb in embeddings)
    
    # Test empty batch
    assert batch_generate_embeddings([]) == []
    
    # Test batch with one item
    single_embedding = batch_generate_embeddings(["Test"])
    assert len(single_embedding) == 1
    assert len(single_embedding[0]) == settings.VECTOR_DIMENSION

def test_compute_similarity():
    """Test similarity computation between embeddings."""
    # Test identical vectors
    vec1 = [1.0] * settings.VECTOR_DIMENSION
    similarity = compute_similarity(vec1, vec1)
    assert pytest.approx(similarity) == 1.0
    
    # Test orthogonal vectors
    vec2 = [0.0] * settings.VECTOR_DIMENSION
    vec2[0] = 1.0
    vec3 = [0.0] * settings.VECTOR_DIMENSION
    vec3[1] = 1.0
    similarity = compute_similarity(vec2, vec3)
    assert pytest.approx(similarity) == 0.0
    
    # Test opposite vectors
    vec4 = [-x for x in vec1]
    similarity = compute_similarity(vec1, vec4)
    assert pytest.approx(similarity) == -1.0

def test_average_embeddings():
    """Test embedding averaging functionality."""
    # Create test embeddings
    emb1 = [1.0] * settings.VECTOR_DIMENSION
    emb2 = [2.0] * settings.VECTOR_DIMENSION
    emb3 = [3.0] * settings.VECTOR_DIMENSION
    
    embeddings = [emb1, emb2, emb3]
    avg_embedding = average_embeddings(embeddings)
    
    assert len(avg_embedding) == settings.VECTOR_DIMENSION
    assert all(x == 2.0 for x in avg_embedding)
    
    # Test single embedding
    single_result = average_embeddings([emb1])
    assert single_result == emb1
    
    # Test empty list
    with pytest.raises(ValueError):
        average_embeddings([])

def test_generate_hybrid_embedding(mock_openai):
    """Test hybrid embedding generation."""
    # Test short text
    short_text = "This is a short text."
    hybrid_emb = generate_hybrid_embedding(short_text)
    assert len(hybrid_emb) == settings.VECTOR_DIMENSION
    
    # Test long text
    long_text = "Test " * 1000
    hybrid_emb = generate_hybrid_embedding(
        long_text,
        chunk_size=100,
        overlap=20
    )
    assert len(hybrid_emb) == settings.VECTOR_DIMENSION
    
    # Test with different chunk sizes
    hybrid_emb1 = generate_hybrid_embedding(
        long_text,
        chunk_size=50,
        overlap=10
    )
    hybrid_emb2 = generate_hybrid_embedding(
        long_text,
        chunk_size=200,
        overlap=40
    )
    
    # Results should be different with different chunk sizes
    similarity = compute_similarity(hybrid_emb1, hybrid_emb2)
    assert similarity < 1.0

def test_embedding_consistency(mock_openai):
    """Test consistency of embedding generation."""
    text = "This is a test document."
    
    # Generate embeddings multiple times
    emb1 = generate_embeddings(text)
    emb2 = generate_embeddings(text)
    
    # Should be identical for same input
    assert np.allclose(emb1, emb2)
    
    # Different texts should have different embeddings
    other_text = "This is a different document."
    other_emb = generate_embeddings(other_text)
    
    similarity = compute_similarity(emb1, other_emb)
    assert similarity < 1.0

def test_error_handling():
    """Test error handling in embedding operations."""
    # Test invalid input types
    with pytest.raises(Exception):
        generate_embeddings(None)
    
    with pytest.raises(Exception):
        generate_embeddings(123)
    
    # Test invalid dimensions in similarity computation
    vec1 = [1.0] * settings.VECTOR_DIMENSION
    vec2 = [1.0] * (settings.VECTOR_DIMENSION - 1)
    
    with pytest.raises(Exception):
        compute_similarity(vec1, vec2)
