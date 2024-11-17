import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import json

def test_query_without_document(client):
    """Test querying without specifying a document."""
    query_data = {
        "query": "What is this about?"
    }
    
    response = client.post("/api/v1/chat/query", json=query_data)
    
    # Should still work but might indicate no context available
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert "session_id" in data

def test_query_with_document(client, uploaded_document):
    """Test querying with a specific document."""
    query_data = {
        "query": "What is this document about?",
        "document_id": uploaded_document["id"]
    }
    
    response = client.post("/api/v1/chat/query", json=query_data)
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert "session_id" in data
    
    # Verify source structure
    source = data["sources"][0]
    assert "content" in source
    assert "score" in source
    assert "document_id" in source
    assert "chunk_index" in source

def test_query_with_session(client, uploaded_document):
    """Test querying with session continuity."""
    # First query to create session
    query1_data = {
        "query": "What is this document about?",
        "document_id": uploaded_document["id"]
    }
    
    response1 = client.post("/api/v1/chat/query", json=query1_data)
    assert response1.status_code == 200
    session_id = response1.json()["session_id"]
    
    # Second query using same session
    query2_data = {
        "query": "Tell me more about that",
        "document_id": uploaded_document["id"],
        "session_id": session_id
    }
    
    response2 = client.post("/api/v1/chat/query", json=query2_data)
    assert response2.status_code == 200
    assert response2.json()["session_id"] == session_id

def test_query_nonexistent_document(client):
    """Test querying with non-existent document ID."""
    query_data = {
        "query": "What is this about?",
        "document_id": 999
    }
    
    response = client.post("/api/v1/chat/query", json=query_data)
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_get_chat_session(client, chat_session):
    """Test retrieving chat session details."""
    response = client.get(f"/api/v1/chat/sessions/{chat_session['session_id']}")
    
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "created_at" in data
    assert "messages" in data
    assert len(data["messages"]) > 0
    
    # Verify message structure
    message = data["messages"][0]
    assert "id" in message
    assert "role" in message
    assert "content" in message
    assert "created_at" in message
    assert "metadata" in message

def test_get_nonexistent_session(client):
    """Test retrieving non-existent chat session."""
    response = client.get("/api/v1/chat/sessions/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_delete_chat_session(client, chat_session):
    """Test deleting a chat session."""
    response = client.delete(f"/api/v1/chat/sessions/{chat_session['session_id']}")
    
    assert response.status_code == 200
    assert "deleted successfully" in response.json()["message"].lower()
    
    # Verify session is deleted
    response = client.get(f"/api/v1/chat/sessions/{chat_session['session_id']}")
    assert response.status_code == 404

def test_delete_nonexistent_session(client):
    """Test deleting non-existent chat session."""
    response = client.delete("/api/v1/chat/sessions/999")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()

def test_conversation_context(client, uploaded_document):
    """Test that conversation context is maintained."""
    # First query
    query1_data = {
        "query": "What is this document about?",
        "document_id": uploaded_document["id"]
    }
    
    response1 = client.post("/api/v1/chat/query", json=query1_data)
    assert response1.status_code == 200
    session_id = response1.json()["session_id"]
    
    # Follow-up query
    query2_data = {
        "query": "Can you elaborate on that?",
        "document_id": uploaded_document["id"],
        "session_id": session_id
    }
    
    response2 = client.post("/api/v1/chat/query", json=query2_data)
    assert response2.status_code == 200
    
    # Get session to verify context
    response = client.get(f"/api/v1/chat/sessions/{session_id}")
    assert response.status_code == 200
    messages = response.json()["messages"]
    assert len(messages) >= 4  # Should have user and assistant messages for both queries

def test_multiple_documents(client, mock_openai):
    """Test querying across multiple documents."""
    # Upload two documents
    files1 = {
        "file": ("test1.txt", "This is the first test document.", "text/plain")
    }
    response1 = client.post("/api/v1/documents/upload", files=files1)
    doc1 = response1.json()
    
    files2 = {
        "file": ("test2.txt", "This is the second test document.", "text/plain")
    }
    response2 = client.post("/api/v1/documents/upload", files=files2)
    doc2 = response2.json()
    
    # Query referencing both documents
    query_data = {
        "query": "Compare these documents",
        "document_id": doc1["id"]  # Primary document
    }
    
    response = client.post("/api/v1/chat/query", json=query_data)
    assert response.status_code == 200

def test_error_handling(client):
    """Test chat endpoint error handling."""
    # Test empty query
    query_data = {
        "query": ""
    }
    response = client.post("/api/v1/chat/query", json=query_data)
    assert response.status_code == 422
    
    # Test invalid session ID
    query_data = {
        "query": "Test query",
        "session_id": "invalid"
    }
    response = client.post("/api/v1/chat/query", json=query_data)
    assert response.status_code == 422
    
    # Test missing required fields
    response = client.post("/api/v1/chat/query", json={})
    assert response.status_code == 422

def test_concurrent_queries(client, uploaded_document):
    """Test handling concurrent chat queries."""
    import concurrent.futures
    
    def make_query(i):
        query_data = {
            "query": f"Query number {i}",
            "document_id": uploaded_document["id"]
        }
        return client.post("/api/v1/chat/query", json=query_data)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_query, i) for i in range(3)]
        responses = [f.result() for f in futures]
    
    assert all(r.status_code == 200 for r in responses)
    session_ids = [r.json()["session_id"] for r in responses]
    assert len(set(session_ids)) == 3  # Each query should get its own session
