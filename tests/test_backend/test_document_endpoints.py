import pytest
from fastapi.testclient import TestClient
import io
from pathlib import Path
import json
from datetime import datetime

from config.settings import settings

def create_test_file(content: str, filename: str = "test.txt"):
    """Create a test file-like object."""
    return io.BytesIO(content.encode())

def test_upload_document(client, mock_openai):
    """Test document upload endpoint."""
    content = "This is a test document."
    files = {
        "file": ("test.txt", content, "text/plain")
    }
    
    response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert data["filename"] == "test.txt"
    assert "chunk_count" in data
    assert "metadata" in data

def test_upload_invalid_file_type(client):
    """Test uploading file with invalid extension."""
    content = "Invalid file content"
    files = {
        "file": ("test.invalid", content, "application/octet-stream")
    }
    
    response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]

def test_upload_large_file(client):
    """Test uploading file exceeding size limit."""
    # Create large content
    content = "x" * (settings.MAX_CONTENT_LENGTH + 1)
    files = {
        "file": ("large.txt", content, "text/plain")
    }
    
    response = client.post("/api/v1/documents/upload", files=files)
    
    assert response.status_code == 400
    assert "File too large" in response.json()["detail"]

def test_list_documents(client, uploaded_document):
    """Test listing documents endpoint."""
    response = client.get("/api/v1/documents/list")
    
    assert response.status_code == 200
    documents = response.json()
    assert len(documents) > 0
    
    # Verify document fields
    document = documents[0]
    assert "id" in document
    assert "filename" in document
    assert "content_type" in document
    assert "created_at" in document
    assert "metadata" in document

def test_list_documents_pagination(client, mock_openai):
    """Test document listing with pagination."""
    # Upload multiple documents
    for i in range(3):
        files = {
            "file": (f"test_{i}.txt", f"Content {i}", "text/plain")
        }
        client.post("/api/v1/documents/upload", files=files)
    
    # Test pagination
    response = client.get("/api/v1/documents/list?skip=0&limit=2")
    assert response.status_code == 200
    page1 = response.json()
    assert len(page1) == 2
    
    response = client.get("/api/v1/documents/list?skip=2&limit=2")
    assert response.status_code == 200
    page2 = response.json()
    assert len(page2) > 0

def test_get_document(client, uploaded_document):
    """Test getting single document endpoint."""
    response = client.get(f"/api/v1/documents/{uploaded_document['id']}")
    
    assert response.status_code == 200
    document = response.json()
    assert document["id"] == uploaded_document["id"]
    assert document["filename"] == uploaded_document["filename"]
    assert "content_type" in document
    assert "file_size" in document
    assert "created_at" in document
    assert "metadata" in document
    assert "chunk_count" in document

def test_get_nonexistent_document(client):
    """Test getting document that doesn't exist."""
    response = client.get("/api/v1/documents/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_delete_document(client, uploaded_document):
    """Test document deletion endpoint."""
    response = client.delete(f"/api/v1/documents/{uploaded_document['id']}")
    
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"].lower()
    
    # Verify document is deleted
    response = client.get(f"/api/v1/documents/{uploaded_document['id']}")
    assert response.status_code == 404

def test_delete_nonexistent_document(client):
    """Test deleting document that doesn't exist."""
    response = client.delete("/api/v1/documents/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_upload_different_file_types(client, mock_openai, test_files_dir):
    """Test uploading different types of files."""
    # Test txt file
    with open(test_files_dir / "test.txt", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", f, "text/plain")}
        )
        assert response.status_code == 200
    
    # Test long document
    with open(test_files_dir / "long_test.txt", "rb") as f:
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("long_test.txt", f, "text/plain")}
        )
        assert response.status_code == 200

def test_concurrent_uploads(client, mock_openai):
    """Test handling multiple uploads concurrently."""
    import concurrent.futures
    
    def upload_file(i):
        content = f"Test content {i}"
        files = {
            "file": (f"test_{i}.txt", content, "text/plain")
        }
        return client.post("/api/v1/documents/upload", files=files)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(upload_file, i) for i in range(3)]
        responses = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in responses)
    
    # Verify all documents were stored
    response = client.get("/api/v1/documents/list")
    documents = response.json()
    assert len(documents) >= 3

def test_error_handling(client):
    """Test API error handling."""
    # Test missing file
    response = client.post("/api/v1/documents/upload")
    assert response.status_code == 422
    
    # Test empty file
    files = {
        "file": ("empty.txt", "", "text/plain")
    }
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    
    # Test invalid pagination parameters
    response = client.get("/api/v1/documents/list?skip=-1")
    assert response.status_code == 422
    
    response = client.get("/api/v1/documents/list?limit=0")
    assert response.status_code == 422
