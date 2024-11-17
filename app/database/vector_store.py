from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.sql import select
from datetime import datetime

from config.settings import settings
from .models import Document, DocumentChunk
from .connection_manager import connection_manager
from .query_optimizer import query_optimizer
from app.document_processing.embeddings import compute_similarity, generate_embeddings
from app.cache.redis_cache import RedisCache, cache_embedding

class VectorStore:
    def __init__(self, session: Session):
        """Initialize vector store with database session."""
        self.session = session
        self.dimension = settings.VECTOR_DIMENSION
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.cache = RedisCache()
    
    async def store_document(
        self,
        filename: str,
        content_type: str,
        file_size: int,
        chunks: List[dict],
        metadata: Optional[dict] = None
    ) -> Document:
        """Store document and chunks with optimized batch processing."""
        try:
            # Create document record
            document = Document(
                filename=filename,
                content_type=content_type,
                file_size=file_size,
                metadata=metadata,
                created_at=datetime.utcnow()
            )
            self.session.add(document)
            self.session.flush()
            
            # Prepare chunks for batch insertion
            chunk_records = []
            for idx, chunk_data in enumerate(chunks):
                # Cache embedding
                await self.cache.set_embedding(
                    chunk_data["content"],
                    chunk_data["embedding"]
                )
                
                chunk = DocumentChunk(
                    document_id=document.id,
                    content=chunk_data["content"],
                    chunk_index=idx,
                    embedding=chunk_data["embedding"],
                    token_count=chunk_data["token_count"],
                    metadata=chunk_data.get("metadata")
                )
                chunk_records.append(chunk)
            
            # Batch insert chunks
            self.session.bulk_save_objects(chunk_records)
            
            # Cache document metadata
            await self.cache.set_document_metadata(
                document.id,
                {
                    "filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "metadata": metadata,
                    "chunk_count": len(chunks),
                    "created_at": document.created_at.isoformat()
                }
            )
            
            self.session.commit()
            return document
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error storing document: {str(e)}")
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = settings.TOP_K_RESULTS,
        threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Tuple[DocumentChunk, float]]:
        """Perform optimized similarity search."""
        try:
            # Execute optimized query
            results = await query_optimizer.execute_similarity_query(
                query_embedding,
                self.session,
                top_k=top_k,
                threshold=threshold or self.similarity_threshold,
                filters=filters
            )
            
            # Convert results to DocumentChunk objects with scores
            chunk_scores = []
            for result in results:
                chunk = self.session.get(DocumentChunk, result["id"])
                if chunk:
                    chunk_scores.append((chunk, result["similarity"]))
            
            return chunk_scores
            
        except Exception as e:
            raise Exception(f"Error performing similarity search: {str(e)}")
    
    @cache_embedding(ttl=settings.CACHE_DEFAULT_TIMEOUT)
    async def update_document_embeddings(
        self,
        document_id: int,
        force: bool = False
    ) -> Document:
        """Update document embeddings with caching."""
        try:
            document = self.session.get(Document, document_id)
            if not document:
                raise ValueError(f"Document with ID {document_id} not found")
            
            # Update embeddings in batches
            batch_size = 100
            for i in range(0, len(document.chunks), batch_size):
                batch = document.chunks[i:i + batch_size]
                for chunk in batch:
                    if force or chunk.embedding is None:
                        # Try cache first
                        cached_embedding = await self.cache.get_embedding(chunk.content)
                        if cached_embedding and not force:
                            chunk.embedding = cached_embedding
                        else:
                            embedding = generate_embeddings(chunk.content)
                            chunk.embedding = embedding
                            # Cache new embedding
                            await self.cache.set_embedding(chunk.content, embedding)
                
                self.session.bulk_save_objects(batch)
            
            # Update document metadata
            document.updated_at = datetime.utcnow()
            
            # Invalidate document cache
            await self.cache.invalidate_document(document_id)
            
            self.session.commit()
            return document
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error updating embeddings: {str(e)}")
    
    async def delete_document(self, document_id: int) -> bool:
        """Delete document with cache invalidation."""
        try:
            document = self.session.get(Document, document_id)
            if not document:
                return False
            
            # Delete document and related chunks
            self.session.delete(document)
            
            # Invalidate cache
            await self.cache.invalidate_document(document_id)
            
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error deleting document: {str(e)}")
    
    async def get_document_by_id(self, document_id: int) -> Optional[Document]:
        """Get document with cached metadata."""
        try:
            # Try cache first
            cached_metadata = await self.cache.get_document_metadata(document_id)
            document = self.session.get(Document, document_id)
            
            if document and cached_metadata:
                document.metadata = cached_metadata.get("metadata")
            
            return document
            
        except Exception as e:
            raise Exception(f"Error retrieving document: {str(e)}")
    
    async def get_all_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict] = None
    ) -> List[Document]:
        """Get documents with optimized query and caching."""
        try:
            query = select(Document)
            
            # Apply filters
            if filters:
                if "content_types" in filters:
                    query = query.filter(Document.content_type.in_(filters["content_types"]))
                if "date_range" in filters:
                    query = query.filter(
                        Document.created_at.between(
                            filters["date_range"]["from"],
                            filters["date_range"]["to"]
                        )
                    )
            
            # Apply pagination
            query = query.offset(skip).limit(limit)
            
            result = self.session.execute(query)
            documents = result.scalars().all()
            
            # Cache metadata for retrieved documents
            for doc in documents:
                await self.cache.set_document_metadata(
                    doc.id,
                    {
                        "filename": doc.filename,
                        "content_type": doc.content_type,
                        "file_size": doc.file_size,
                        "metadata": doc.metadata,
                        "chunk_count": len(doc.chunks),
                        "created_at": doc.created_at.isoformat()
                    }
                )
            
            return documents
            
        except Exception as e:
            raise Exception(f"Error retrieving documents: {str(e)}")
    
    async def add_chunks(
        self,
        document_id: int,
        chunks: List[Dict[str, Any]]
    ) -> bool:
        """Add chunks to existing document with batch processing."""
        try:
            document = self.session.get(Document, document_id)
            if not document:
                return False
            
            # Get current max chunk index
            max_index = max(
                (c.chunk_index for c in document.chunks),
                default=-1
            )
            
            # Prepare chunks for batch insertion
            chunk_records = []
            for idx, chunk_data in enumerate(chunks, start=max_index + 1):
                # Cache embedding
                await self.cache.set_embedding(
                    chunk_data["content"],
                    chunk_data["embedding"]
                )
                
                chunk = DocumentChunk(
                    document_id=document_id,
                    content=chunk_data["content"],
                    chunk_index=idx,
                    embedding=chunk_data["embedding"],
                    token_count=chunk_data["token_count"],
                    metadata=chunk_data.get("metadata")
                )
                chunk_records.append(chunk)
            
            # Batch insert chunks
            self.session.bulk_save_objects(chunk_records)
            
            # Update document metadata
            document.updated_at = datetime.utcnow()
            
            # Invalidate cache
            await self.cache.invalidate_document(document_id)
            
            self.session.commit()
            return True
            
        except Exception as e:
            self.session.rollback()
            raise Exception(f"Error adding chunks: {str(e)}")
