import pytest
from unittest.mock import Mock, patch
import asyncio
from pathlib import Path
import json

from app.document_processing.batch_processor import BatchProcessor
from app.database.vector_store import VectorStore

@pytest.fixture
def mock_vector_store():
    """Create mock vector store."""
    store = Mock(spec=VectorStore)
    store.store_document.return_value = Mock(
        id=1,
        filename="test.txt",
        metadata={"test": "metadata"}
    )
    return store

@pytest.fixture
def processor(mock_vector_store):
    """Create batch processor instance."""
    return BatchProcessor(mock_vector_store)

@pytest.fixture
def test_documents():
    """Create test document data."""
    return [
        {
            "content": "Test content",
            "filename": "test.txt",
            "content_type": "text/plain",
            "file_size": 100,
            "metadata": {"test": "metadata"}
        },
        {
            "content": "class TestClass:\n    pass",
            "filename": "test.py",
            "content_type": "text/x-python",
            "file_size": 50,
            "metadata": {"test": "metadata"}
        },
        {
            "content": bytes([0xFF, 0xD8, 0xFF, 0xE0]),  # JPEG header
            "filename": "test.jpg",
            "content_type": "image/jpeg",
            "file_size": 1000,
            "metadata": {"test": "metadata"}
        }
    ]

@pytest.mark.asyncio
async def test_process_documents(processor, test_documents):
    """Test processing multiple documents."""
    with patch('app.document_processing.code_parser.code_parser.parse_code_file') as mock_code_parse, \
         patch('app.document_processing.ocr_processor.ocr_processor.process_image') as mock_ocr, \
         patch('app.document_processing.chunking.process_document') as mock_process:
        
        # Mock code parser
        mock_code_parse.return_value = {
            "chunks": [{"content": "test", "embedding": [0.1] * 1536}],
            "language": "Python",
            "metadata": {},
            "highlighted_code": "<pre>test</pre>"
        }
        
        # Mock OCR processor
        mock_ocr.return_value = {
            "text_blocks": [{"text": "test", "confidence": 0.9}],
            "metadata": {}
        }
        
        # Mock standard chunking
        mock_process.return_value = [
            {
                "content": "test",
                "embedding": [0.1] * 1536,
                "metadata": {}
            }
        ]
        
        results = await processor.process_documents(test_documents)
        
        assert len(results) == len(test_documents)
        assert all(isinstance(r, dict) for r in results)
        assert all("id" in r for r in results)

def test_determine_document_type(processor):
    """Test document type determination."""
    assert processor._determine_document_type("test.py", "text/x-python") == "code"
    assert processor._determine_document_type("test.jpg", "image/jpeg") == "image"
    assert processor._determine_document_type("test.pdf", "application/pdf") == "pdf"
    assert processor._determine_document_type("test.txt", "text/plain") == "text"

@pytest.mark.asyncio
async def test_process_code_file(processor):
    """Test code file processing."""
    document = {
        "content": "def test():\n    pass",
        "filename": "test.py",
        "content_type": "text/x-python",
        "file_size": 50
    }
    
    with patch('app.document_processing.code_parser.code_parser.parse_code_file') as mock_parse:
        mock_parse.return_value = {
            "chunks": [
                {
                    "type": "function",
                    "name": "test",
                    "content": "def test():\n    pass",
                    "embedding": [0.1] * 1536
                }
            ],
            "language": "Python",
            "metadata": {"complexity": {"cyclomatic": 1}},
            "highlighted_code": "<pre>code</pre>"
        }
        
        result = await processor._process_code_file(document)
        
        assert "chunks" in result
        assert "metadata" in result
        assert "highlighted_code" in result["metadata"]
        assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_process_image_file(processor):
    """Test image file processing."""
    document = {
        "content": b"image_data",
        "filename": "test.jpg",
        "content_type": "image/jpeg",
        "file_size": 1000
    }
    
    with patch('app.document_processing.ocr_processor.ocr_processor.process_image') as mock_ocr:
        mock_ocr.return_value = {
            "text_blocks": [
                {
                    "text": "test",
                    "confidence": 0.9,
                    "position": {"left": 0, "top": 0}
                }
            ],
            "metadata": {"image_size": (100, 100)}
        }
        
        result = await processor._process_image_file(document)
        
        assert "chunks" in result
        assert "metadata" in result
        assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_process_pdf_file(processor):
    """Test PDF file processing."""
    document = {
        "content": b"pdf_data",
        "filename": "test.pdf",
        "content_type": "application/pdf",
        "file_size": 1000
    }
    
    with patch('app.document_processing.table_extractor.table_extractor.extract_tables_from_pdf') as mock_tables, \
         patch('app.document_processing.ocr_processor.ocr_processor.process_pdf_images') as mock_ocr:
        
        mock_tables.return_value = [{
            "text_content": "table data",
            "embedding": [0.1] * 1536,
            "table_id": 1,
            "data": {"headers": ["col1"], "rows": [["data"]]}
        }]
        
        mock_ocr.return_value = [{
            "text_blocks": [
                {
                    "text": "test",
                    "confidence": 0.9,
                    "position": {"left": 0, "top": 0}
                }
            ],
            "page_number": 1
        }]
        
        result = await processor._process_pdf_file(document)
        
        assert "chunks" in result
        assert "metadata" in result
        assert result["metadata"]["table_count"] > 0
        assert result["metadata"]["page_count"] > 0

@pytest.mark.asyncio
async def test_process_text_file(processor):
    """Test text file processing."""
    document = {
        "content": "Test content",
        "filename": "test.txt",
        "content_type": "text/plain",
        "file_size": 100,
        "metadata": {"test": "metadata"}
    }
    
    with patch('app.document_processing.chunking.process_document') as mock_process:
        mock_process.return_value = [{
            "content": "Test content",
            "embedding": [0.1] * 1536,
            "metadata": {}
        }]
        
        result = await processor._process_text_file(document)
        
        assert "chunks" in result
        assert "metadata" in result
        assert len(result["chunks"]) > 0

@pytest.mark.asyncio
async def test_error_handling(processor, test_documents):
    """Test error handling in batch processing."""
    # Modify one document to cause an error
    test_documents[0]["content"] = None
    
    results = await processor.process_documents(test_documents)
    
    # Should still process other documents
    assert len(results) < len(test_documents)
    assert all(isinstance(r, dict) for r in results)

@pytest.mark.asyncio
async def test_concurrent_processing(processor, test_documents):
    """Test concurrent document processing."""
    # Create multiple copies of test documents
    many_documents = test_documents * 3
    
    with patch('app.document_processing.code_parser.code_parser.parse_code_file') as mock_code_parse, \
         patch('app.document_processing.ocr_processor.ocr_processor.process_image') as mock_ocr, \
         patch('app.document_processing.chunking.process_document') as mock_process:
        
        # Set up mocks
        mock_code_parse.return_value = {
            "chunks": [], "language": "Python",
            "metadata": {}, "highlighted_code": ""
        }
        mock_ocr.return_value = {"text_blocks": [], "metadata": {}}
        mock_process.return_value = []
        
        results = await processor.process_documents(many_documents)
        
        assert len(results) == len(many_documents)
