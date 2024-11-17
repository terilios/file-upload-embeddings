from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import mimetypes
import magic
import logging
from datetime import datetime

from app.document_processing.chunking import process_document
from app.document_processing.table_extractor import table_extractor
from app.document_processing.ocr_processor import ocr_processor
from app.document_processing.code_parser import code_parser
from app.database.vector_store import VectorStore
from config.settings import settings

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Process documents in batches with advanced feature support."""
    
    def __init__(self, vector_store: VectorStore):
        """Initialize batch processor."""
        self.vector_store = vector_store
        self.supported_code_extensions = code_parser.supported_extensions.keys()
    
    async def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel.
        
        Args:
            documents: List of document dictionaries containing:
                - content: Document content or file path
                - filename: Name of the file
                - content_type: MIME type
                - file_size: Size in bytes
                - metadata: Optional metadata
        
        Returns:
            List of processed document results
        """
        tasks = [
            self._process_single_document(doc)
            for doc in documents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out failed documents
        successful_results = [
            result for result in results
            if not isinstance(result, Exception)
        ]
        
        # Log failures
        failures = [
            result for result in results
            if isinstance(result, Exception)
        ]
        if failures:
            logger.error(f"Failed to process {len(failures)} documents")
            for error in failures:
                logger.error(str(error))
        
        return successful_results
    
    async def _process_single_document(
        self,
        document: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single document with advanced features."""
        try:
            filename = document["filename"]
            content_type = document["content_type"]
            file_size = document["file_size"]
            metadata = document.get("metadata", {})
            
            # Determine document type and processing strategy
            doc_type = self._determine_document_type(filename, content_type)
            
            # Process based on document type
            if doc_type == "code":
                result = await self._process_code_file(document)
            elif doc_type == "image":
                result = await self._process_image_file(document)
            elif doc_type == "pdf":
                result = await self._process_pdf_file(document)
            else:
                result = await self._process_text_file(document)
            
            # Store in vector database
            if result:
                doc = await self.vector_store.store_document(
                    filename=filename,
                    content_type=content_type,
                    file_size=file_size,
                    chunks=result["chunks"],
                    metadata={
                        **metadata,
                        **result.get("metadata", {}),
                        "document_type": doc_type,
                        "processing_timestamp": datetime.utcnow().isoformat()
                    }
                )
                
                return {
                    "id": doc.id,
                    "filename": filename,
                    "content_type": content_type,
                    "file_size": file_size,
                    "chunk_count": len(result["chunks"]),
                    "metadata": doc.metadata
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error processing document {filename}: {str(e)}")
            raise
    
    def _determine_document_type(
        self,
        filename: str,
        content_type: str
    ) -> str:
        """Determine document type for processing strategy."""
        ext = Path(filename).suffix.lower()
        
        if ext in self.supported_code_extensions:
            return "code"
        elif content_type.startswith("image/"):
            return "image"
        elif content_type == "application/pdf":
            return "pdf"
        else:
            return "text"
    
    async def _process_code_file(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process code file with syntax highlighting and structure analysis."""
        content = document["content"]
        filename = document["filename"]
        
        # Parse code file
        code_data = await code_parser.parse_code_file(
            content if isinstance(content, str) else content.decode()
        )
        
        # Convert code chunks to vector store format
        chunks = []
        for chunk in code_data["chunks"]:
            chunks.append({
                "content": chunk["content"],
                "embedding": chunk["embedding"],
                "metadata": {
                    "type": chunk["type"],
                    "name": chunk["name"],
                    "language": code_data["language"]
                }
            })
        
        return {
            "chunks": chunks,
            "metadata": {
                **code_data["metadata"],
                "highlighted_code": code_data["highlighted_code"]
            }
        }
    
    async def _process_image_file(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process image file with OCR."""
        content = document["content"]
        
        # Process image with OCR
        ocr_data = await ocr_processor.process_image(content)
        
        # Convert OCR blocks to vector store format
        chunks = []
        for block in ocr_data["text_blocks"]:
            chunks.append({
                "content": block["text"],
                "embedding": await generate_embeddings(block["text"]),
                "metadata": {
                    "confidence": block["confidence"],
                    "position": block["position"]
                }
            })
        
        return {
            "chunks": chunks,
            "metadata": ocr_data["metadata"]
        }
    
    async def _process_pdf_file(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process PDF file with table extraction and OCR."""
        content = document["content"]
        
        # Extract tables
        tables = await table_extractor.extract_tables_from_pdf(content)
        
        # Process PDF images
        ocr_results = await ocr_processor.process_pdf_images(content)
        
        # Combine chunks from tables and OCR
        chunks = []
        
        # Add table chunks
        for table in tables:
            chunks.append({
                "content": table["text_content"],
                "embedding": table["embedding"],
                "metadata": {
                    "type": "table",
                    "table_id": table["table_id"],
                    "table_data": table["data"]
                }
            })
        
        # Add OCR chunks
        for page in ocr_results:
            for block in page["text_blocks"]:
                chunks.append({
                    "content": block["text"],
                    "embedding": await generate_embeddings(block["text"]),
                    "metadata": {
                        "type": "ocr",
                        "page": page["page_number"],
                        "confidence": block["confidence"],
                        "position": block["position"]
                    }
                })
        
        return {
            "chunks": chunks,
            "metadata": {
                "table_count": len(tables),
                "page_count": len(ocr_results)
            }
        }
    
    async def _process_text_file(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text file with standard chunking."""
        content = document["content"]
        filename = document["filename"]
        content_type = document["content_type"]
        file_size = document["file_size"]
        metadata = document.get("metadata", {})
        
        # Process with standard chunking
        chunks = process_document(
            content=content,
            filename=filename,
            content_type=content_type,
            file_size=file_size,
            metadata=metadata
        )
        
        return {
            "chunks": chunks,
            "metadata": {}
        }

# Global batch processor instance will be initialized with vector store
batch_processor = None
