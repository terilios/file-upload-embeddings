from typing import List, Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
import io
import PyPDF2
import docx
from datetime import datetime

from app.database.vector_store import VectorStore
from app.document_processing.chunking import process_document, extract_metadata
from app.database.models import Document
from config.settings import settings
from app.backend.main import get_db

router = APIRouter()

async def read_file_content(file: UploadFile) -> str:
    """Extract text content from various file types."""
    content = ""
    file_extension = file.filename.lower().split('.')[-1]
    
    try:
        if file_extension == 'txt':
            content = (await file.read()).decode('utf-8')
        
        elif file_extension == 'pdf':
            pdf_bytes = await file.read()
            pdf_file = io.BytesIO(pdf_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            content = ' '.join(page.extract_text() for page in pdf_reader.pages)
        
        elif file_extension in ['doc', 'docx']:
            doc_bytes = await file.read()
            doc_file = io.BytesIO(doc_bytes)
            doc = docx.Document(doc_file)
            content = ' '.join(paragraph.text for paragraph in doc.paragraphs)
        
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        return content.strip()
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload and process a document for vector search.
    """
    # Validate file extension
    file_extension = file.filename.lower().split('.')[-1]
    if file_extension not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(settings.ALLOWED_EXTENSIONS)}"
        )
    
    try:
        # Read file content
        content = await read_file_content(file)
        
        # Get file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset position
        
        # Extract metadata
        metadata = extract_metadata(
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size
        )
        
        # Process document into chunks with embeddings
        processed_chunks = process_document(
            content=content,
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            metadata=metadata
        )
        
        # Store document and chunks in vector store
        vector_store = VectorStore(db)
        document = await vector_store.store_document(
            filename=file.filename,
            content_type=file.content_type,
            file_size=file_size,
            chunks=processed_chunks,
            metadata=metadata
        )
        
        return {
            "id": document.id,
            "filename": document.filename,
            "chunk_count": len(processed_chunks),
            "metadata": document.metadata
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}"
        )

@router.get("/list")
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    List all uploaded documents with pagination.
    """
    vector_store = VectorStore(db)
    documents = await vector_store.get_all_documents(skip=skip, limit=limit)
    
    return [{
        "id": doc.id,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "created_at": doc.created_at,
        "metadata": doc.metadata
    } for doc in documents]

@router.get("/{document_id}")
async def get_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Get document details by ID.
    """
    vector_store = VectorStore(db)
    document = await vector_store.get_document_by_id(document_id)
    
    if not document:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found"
        )
    
    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "file_size": document.file_size,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "metadata": document.metadata,
        "chunk_count": len(document.chunks)
    }

@router.delete("/{document_id}")
async def delete_document(
    document_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete a document and its chunks.
    """
    vector_store = VectorStore(db)
    success = await vector_store.delete_document(document_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Document with ID {document_id} not found"
        )
    
    return {"message": "Document deleted successfully"}
