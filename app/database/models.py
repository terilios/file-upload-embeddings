from datetime import datetime
from typing import List, Optional
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON, Text
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.types import TypeDecorator, UserDefinedType

Base = declarative_base()

class Vector(UserDefinedType):
    def __init__(self, dimensions=None):
        self.dimensions = dimensions

    def get_col_spec(self):
        if self.dimensions is not None:
            return f"vector({self.dimensions})"
        return "vector"

    def bind_expression(self, bindvalue):
        return bindvalue

    def column_expression(self, col):
        return col

class Document(Base):
    """Model for storing document metadata and content."""
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    content_type = Column(String, nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    doc_metadata = Column(JSON, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Document(id={self.id}, filename={self.filename})>"

class DocumentChunk(Base):
    """Model for storing document chunks with their embeddings."""
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within document
    embedding = Column(Vector(1536), nullable=True)  # Vector embedding with OpenAI dimensions
    token_count = Column(Integer, nullable=False)
    chunk_metadata = Column(JSON, nullable=True)  # Chunk-specific metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"

class ChatSession(Base):
    """Model for storing chat sessions."""
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    session_metadata = Column(JSON, nullable=True)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<ChatSession(id={self.id})>"

class ChatMessage(Base):
    """Model for storing chat messages."""
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    message_metadata = Column(JSON, nullable=True)  # For storing citations, sources, etc.
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self):
        return f"<ChatMessage(id={self.id}, session_id={self.session_id}, role={self.role})>"
