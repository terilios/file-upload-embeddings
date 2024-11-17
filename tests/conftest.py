import pytest
import os
from typing import Generator, Dict
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from fastapi.testclient import TestClient
import numpy as np
from pathlib import Path
import sys

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.database.models import Base
from app.backend.main import app, get_db
from config.settings import settings

# Test database URL
TEST_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/test_file_upload_embeddings"

@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    engine = create_engine(TEST_DATABASE_URL)
    
    # Create test database tables
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Drop all tables after tests
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """Create a fresh database session for each test."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    
    try:
        yield session
    finally:
        session.rollback()
        session.close()

@pytest.fixture(scope="function")
def client(db_session) -> Generator:
    """Create a test client with a test database session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client
    
    app.dependency_overrides.clear()

@pytest.fixture(scope="session")
def test_documents() -> Dict:
    """Provide test document data."""
    return {
        "simple": {
            "content": "This is a test document for unit testing.",
            "filename": "test.txt",
            "content_type": "text/plain",
            "file_size": 37
        },
        "long": {
            "content": " ".join(["Test content"] * 100),  # Long document
            "filename": "long_test.txt",
            "content_type": "text/plain",
            "file_size": 1200
        }
    }

@pytest.fixture(scope="session")
def mock_embeddings() -> Dict:
    """Provide consistent mock embeddings for testing."""
    return {
        "simple": np.random.rand(settings.VECTOR_DIMENSION).tolist(),
        "long": np.random.rand(settings.VECTOR_DIMENSION).tolist()
    }

@pytest.fixture(scope="function")
def mock_openai(monkeypatch):
    """Mock OpenAI API calls."""
    class MockResponse:
        def __init__(self, embedding):
            self.data = [type('obj', (object,), {'embedding': embedding})]
    
    def mock_create(*args, **kwargs):
        return MockResponse(mock_embeddings()["simple"])
    
    # Mock both standard and Azure OpenAI
    monkeypatch.setattr("openai.Embedding.create", mock_create)
    monkeypatch.setattr("openai.AzureOpenAI.Embedding.create", mock_create)

@pytest.fixture(scope="function")
def uploaded_document(client, test_documents) -> Dict:
    """Create a test document in the database."""
    files = {
        "file": (
            test_documents["simple"]["filename"],
            test_documents["simple"]["content"],
            test_documents["simple"]["content_type"]
        )
    }
    response = client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 200
    return response.json()

@pytest.fixture(scope="function")
def chat_session(client, uploaded_document) -> Dict:
    """Create a test chat session."""
    query = "What is this document about?"
    response = client.post(
        "/api/v1/chat/query",
        json={
            "query": query,
            "document_id": uploaded_document["id"]
        }
    )
    assert response.status_code == 200
    return response.json()

@pytest.fixture(scope="session")
def test_files_dir(tmp_path_factory) -> Path:
    """Create a temporary directory with test files."""
    test_files = tmp_path_factory.mktemp("test_files")
    
    # Create a text file
    with open(test_files / "test.txt", "w") as f:
        f.write("This is a test document for unit testing.")
    
    # Create a long text file
    with open(test_files / "long_test.txt", "w") as f:
        f.write(" ".join(["Test content"] * 100))
    
    return test_files

@pytest.fixture(scope="function")
def cleanup_files():
    """Clean up any test files after tests."""
    yield
    # Clean up any files created during tests
    test_files = Path("test_files")
    if test_files.exists():
        for file in test_files.glob("*"):
            file.unlink()
        test_files.rmdir()
