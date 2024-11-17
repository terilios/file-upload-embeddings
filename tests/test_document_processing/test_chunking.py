import pytest
from typing import List, Dict
import numpy as np

from app.document_processing.chunking import (
    determine_chunk_params,
    split_text_into_chunks,
    process_document,
    extract_metadata
)
from config.settings import settings

def test_determine_chunk_params():
    """Test chunk parameter determination based on content type and file size."""
    # Test email type
    size_small = 30000  # 30KB
    chunk_size, overlap = determine_chunk_params("email", size_small)
    assert chunk_size == settings.CHUNK_SIZE_MAPPING["email"]
    assert overlap == settings.CHUNK_OVERLAP_MAPPING["email"]
    
    # Test report type
    size_medium = 500000  # 500KB
    chunk_size, overlap = determine_chunk_params("application/pdf", size_medium)
    assert chunk_size == settings.CHUNK_SIZE_MAPPING["report"]
    assert overlap == settings.CHUNK_OVERLAP_MAPPING["report"]
    
    # Test technical type
    size_large = 2000000  # 2MB
    chunk_size, overlap = determine_chunk_params("technical", size_large)
    assert chunk_size == settings.CHUNK_SIZE_MAPPING["technical"]
    assert overlap == settings.CHUNK_OVERLAP_MAPPING["technical"]
    
    # Test default type
    chunk_size, overlap = determine_chunk_params("unknown", size_medium)
    assert chunk_size == settings.CHUNK_SIZE_MAPPING["default"]
    assert overlap == settings.CHUNK_OVERLAP_MAPPING["default"]

def test_split_text_into_chunks():
    """Test text splitting into chunks with overlap."""
    # Test simple text
    text = "This is a test. Another sentence. And a third one."
    chunk_size = 20
    overlap = 5
    chunks = split_text_into_chunks(text, chunk_size, overlap)
    
    assert len(chunks) > 0
    assert all(len(chunk) <= chunk_size for chunk in chunks)
    
    # Test long text
    long_text = " ".join(["Sentence number {}".format(i) for i in range(100)])
    chunks = split_text_into_chunks(long_text, chunk_size=50, overlap=10)
    
    assert len(chunks) > 1
    assert all(len(chunk) <= 50 for chunk in chunks)
    
    # Test overlap
    chunks = split_text_into_chunks(
        "First sentence. Second sentence. Third sentence.",
        chunk_size=20,
        overlap=10
    )
    if len(chunks) > 1:
        # Check if there's some overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            assert any(
                word in chunks[i+1] 
                for word in chunks[i].split()
            )

def test_process_document(mock_openai):
    """Test document processing with chunking and embeddings."""
    content = "This is a test document. It has multiple sentences. Testing chunk processing."
    filename = "test.txt"
    content_type = "text/plain"
    file_size = len(content.encode())
    
    processed_chunks = process_document(
        content=content,
        filename=filename,
        content_type=content_type,
        file_size=file_size
    )
    
    assert len(processed_chunks) > 0
    
    for chunk in processed_chunks:
        assert "content" in chunk
        assert "embedding" in chunk
        assert "token_count" in chunk
        assert "metadata" in chunk
        
        # Check embedding dimension
        assert len(chunk["embedding"]) == settings.VECTOR_DIMENSION
        
        # Check metadata
        assert chunk["metadata"]["original_file"] == filename
        assert "index" in chunk["metadata"]
        assert "total_chunks" in chunk["metadata"]
        assert "token_count" in chunk["metadata"]

def test_extract_metadata():
    """Test metadata extraction from file information."""
    filename = "test_document.pdf"
    content_type = "application/pdf"
    file_size = 1024
    
    metadata = extract_metadata(filename, content_type, file_size)
    
    assert metadata["filename"] == filename
    assert metadata["content_type"] == content_type
    assert metadata["file_size"] == file_size
    assert metadata["file_extension"] == ".pdf"
    assert "processing_timestamp" in metadata

def test_chunk_size_limits():
    """Test that chunks don't exceed maximum token limits."""
    # Create a very long document
    long_text = " ".join(["Test content"] * 1000)
    
    processed_chunks = process_document(
        content=long_text,
        filename="long_test.txt",
        content_type="text/plain",
        file_size=len(long_text.encode())
    )
    
    # Check that each chunk's token count is within limits
    for chunk in processed_chunks:
        assert chunk["token_count"] <= 8191  # OpenAI's token limit

def test_empty_document():
    """Test handling of empty documents."""
    processed_chunks = process_document(
        content="",
        filename="empty.txt",
        content_type="text/plain",
        file_size=0
    )
    
    assert len(processed_chunks) == 0

def test_special_characters():
    """Test handling of special characters in text."""
    special_text = """
    Special chars: !@#$%^&*()
    New lines: \n\n
    Tabs: \t\t
    Unicode: 你好世界
    """
    
    processed_chunks = process_document(
        content=special_text,
        filename="special.txt",
        content_type="text/plain",
        file_size=len(special_text.encode())
    )
    
    assert len(processed_chunks) > 0
    # Verify content is preserved
    assert any("Special chars" in chunk["content"] for chunk in processed_chunks)
    assert any("Unicode" in chunk["content"] for chunk in processed_chunks)

def test_metadata_consistency():
    """Test consistency of metadata across chunks."""
    content = "Multiple sentence document. " * 10
    
    processed_chunks = process_document(
        content=content,
        filename="test.txt",
        content_type="text/plain",
        file_size=len(content.encode())
    )
    
    # All chunks should have the same total_chunks value
    total_chunks = processed_chunks[0]["metadata"]["total_chunks"]
    assert all(
        chunk["metadata"]["total_chunks"] == total_chunks 
        for chunk in processed_chunks
    )
    
    # Indexes should be sequential
    indexes = [chunk["metadata"]["index"] for chunk in processed_chunks]
    assert indexes == list(range(len(processed_chunks)))
