from typing import List, Dict, Any, Optional, Union
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
        self.semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_TASKS)
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1,
            'max_delay': 10
        }

    async def _process_with_retries(self, coroutine, doc_info: str):
        """Execute coroutine with exponential backoff retry logic."""
        retries = 0
        while retries < self.retry_config['max_retries']:
            try:
                return await coroutine
            except Exception as e:
                retries += 1
                if retries == self.retry_config['max_retries']:
                    logger.error(f"Failed processing {doc_info} after {retries} retries: {str(e)}")
                    raise
                delay = min(
                    self.retry_config['base_delay'] * (2 ** retries),
                    self.retry_config['max_delay']
                )
                logger.warning(f"Retry {retries} for {doc_info} after {delay}s: {str(e)}")
                await asyncio.sleep(delay)

    async def process_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process multiple documents in parallel with improved error handling and metadata extraction.
        
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
        async def process_with_semaphore(doc):
            async with self.semaphore:
                return await self._process_with_retries(
                    self._process_single_document(doc),
                    f"document {doc.get('filename', 'unknown')}"
                )

        tasks = [process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        processed_results = []
        for doc, result in zip(documents, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to process {doc.get('filename', 'unknown')}: {str(result)}")
                processed_results.append({
                    'status': 'error',
                    'filename': doc.get('filename', 'unknown'),
                    'error': str(result),
                    'timestamp': datetime.utcnow().isoformat()
                })
            else:
                processed_results.append(result)
        
        return processed_results

    async def _process_single_document(
        self,
        document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single document with enhanced metadata extraction and validation."""
        try:
            # Basic validation
            if not document.get('content'):
                raise ValueError("Document content is missing")

            # Enhanced metadata extraction
            metadata = document.get('metadata', {})
            metadata.update({
                'filename': document.get('filename'),
                'content_type': document.get('content_type'),
                'file_size': document.get('file_size'),
                'processing_timestamp': datetime.utcnow().isoformat(),
                'processing_version': '2.0'
            })

            # Detect file type and validate
            content_type = document.get('content_type')
            if not content_type:
                content_type = magic.from_buffer(document['content'][:2048], mime=True)
                metadata['detected_content_type'] = content_type

            # Process based on content type
            result = await self._route_processing(document['content'], content_type, metadata)
            
            # Enrich metadata with processing results
            result['metadata'].update({
                'processing_status': 'success',
                'processing_duration': (
                    datetime.utcnow() - 
                    datetime.fromisoformat(metadata['processing_timestamp'])
                ).total_seconds()
            })
            
            return result

        except Exception as e:
            logger.error(f"Error processing document {document.get('filename', 'unknown')}: {str(e)}")
            raise

    async def _route_processing(
        self,
        content: Union[str, bytes],
        content_type: str,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route document to appropriate processor based on content type."""
        try:
            if content_type.startswith('text/'):
                return await self._process_text_document(content, metadata)
            elif content_type.startswith('image/'):
                return await self._process_image_document(content, metadata)
            elif content_type.startswith('application/pdf'):
                return await self._process_pdf_document(content, metadata)
            else:
                raise ValueError(f"Unsupported content type: {content_type}")

        except Exception as e:
            logger.error(f"Error in content type specific processing: {str(e)}")
            raise

    async def _process_text_document(
        self,
        content: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process text document with standard chunking."""
        filename = metadata.get('filename')
        content_type = metadata.get('content_type')
        file_size = metadata.get('file_size')
        
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

    async def _process_image_document(
        self,
        content: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process image document with OCR."""
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

    async def _process_pdf_document(
        self,
        content: Union[str, bytes],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process PDF document with table extraction and OCR."""
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

# Global batch processor instance will be initialized with vector store
batch_processor = None
