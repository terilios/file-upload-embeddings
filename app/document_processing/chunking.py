from typing import List, Dict, Optional
import re
import logging
from pathlib import Path
from datetime import datetime

from config.settings import settings
from .embeddings import count_tokens, generate_embeddings

logger = logging.getLogger(__name__)

def determine_chunk_params(
    content_type: str,
    file_size: int
) -> tuple[int, int]:
    """
    Determine chunk size and overlap based on document type and file size.
    
    Args:
        content_type: MIME type of the document
        file_size: Size of the file in bytes
    
    Returns:
        Tuple of (chunk_size, overlap)
    """
    # Logging added for debugging
    print(f"Determining chunk params - Content Type: {content_type}, File Size: {file_size}")
    print(f"Available chunk size mapping: {settings.CHUNK_SIZE_MAPPING}")
    print(f"Available chunk overlap mapping: {settings.CHUNK_OVERLAP_MAPPING}")
    
    # Map content type to document type
    doc_type = "default"
    if "email" in content_type or file_size < 50000:  # 50KB
        doc_type = "email"
    elif "pdf" in content_type or "document" in content_type:
        doc_type = "report"
    elif "technical" in content_type or file_size > 1000000:  # 1MB
        doc_type = "technical"
    
    print(f"Determined document type: {doc_type}")
    
    chunk_size = settings.CHUNK_SIZE_MAPPING.get(doc_type, settings.CHUNK_SIZE_MAPPING["default"])
    chunk_overlap = settings.CHUNK_OVERLAP_MAPPING.get(doc_type, settings.CHUNK_OVERLAP_MAPPING["default"])
    
    print(f"Chunk parameters - Size: {chunk_size}, Overlap: {chunk_overlap}")
    
    return chunk_size, chunk_overlap

def split_text_into_chunks(
    text: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[str]:
    """
    Split text into overlapping chunks while trying to maintain semantic boundaries.
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in characters
        chunk_overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        if current_size + sentence_size <= chunk_size:
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if chunks:
                # Find sentences from previous chunk to include as overlap
                overlap_text = chunks[-1]
                overlap_sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
                overlap_size = 0
                overlap_chunk = []
                
                # Add sentences from the end until we reach desired overlap
                for sent in reversed(overlap_sentences):
                    if overlap_size + len(sent) > chunk_overlap:
                        break
                    overlap_chunk.insert(0, sent)
                    overlap_size += len(sent) + 1
                
                current_chunk = overlap_chunk + [sentence]
                current_size = sum(len(s) + 1 for s in current_chunk)
            else:
                current_chunk = [sentence]
                current_size = sentence_size + 1
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def truncate_text(text: str, max_tokens: int) -> str:
    """
    Truncate text to a maximum number of tokens.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens
    
    Returns:
        Truncated text
    """
    words = text.split()
    truncated_text = ' '.join(words[:max_tokens])
    return truncated_text

def process_document(
    content: str,
    filename: str,
    content_type: str,
    file_size: int,
    metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Process a document into chunks with embeddings and metadata.
    
    Args:
        content: Document content
        filename: Name of the file
        content_type: MIME type of the file
        file_size: Size of the file in bytes
        metadata: Optional document metadata
    
    Returns:
        List of dictionaries containing chunk data
    
    Raises:
        ValueError: If document content is invalid or processing fails
        RuntimeError: If embedding generation fails repeatedly
    """
    # Enhanced input validation
    if not isinstance(content, str):
        raise ValueError(f"Document content must be string, got {type(content)}")
    
    content = content.strip()
    if not content:
        raise ValueError("Document content is empty")
        
    if file_size > settings.MAX_CONTENT_LENGTH:
        raise ValueError(f"File size {file_size} exceeds maximum allowed size {settings.MAX_CONTENT_LENGTH}")
    
    # Determine chunk parameters with validation
    try:
        chunk_size, chunk_overlap = determine_chunk_params(content_type, file_size)
        if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError(f"Invalid chunk parameters: size={chunk_size}, overlap={chunk_overlap}")
    except Exception as e:
        raise ValueError(f"Failed to determine chunk parameters: {str(e)}")
    
    # Split content into chunks with validation
    try:
        text_chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
        if not text_chunks:
            raise ValueError("Failed to split document into chunks")
    except Exception as e:
        raise ValueError(f"Error during document chunking: {str(e)}")
    
    # Process each chunk with retries
    processed_chunks = []
    max_retries = 3
    
    for idx, chunk_text in enumerate(text_chunks):
        # Skip empty chunks
        if not chunk_text.strip():
            continue
            
        # Count tokens with validation
        try:
            token_count = count_tokens(chunk_text)
            if token_count > settings.MAX_TOKENS:
                chunk_text = truncate_text(chunk_text, settings.MAX_TOKENS)
                token_count = count_tokens(chunk_text)
        except Exception as e:
            raise ValueError(f"Error counting tokens for chunk {idx}: {str(e)}")
        
        # Generate embedding with retries
        embedding = None
        last_error = None
        
        for retry in range(max_retries):
            try:
                # Try OpenAI first
                embedding = generate_embeddings(chunk_text, use_azure=False)
                if embedding and len(embedding) == settings.VECTOR_DIMENSION:
                    break
            except Exception as e:
                try:
                    # Fallback to Azure OpenAI
                    embedding = generate_embeddings(chunk_text, use_azure=True)
                    if embedding and len(embedding) == settings.VECTOR_DIMENSION:
                        break
                except Exception as e2:
                    last_error = f"OpenAI: {str(e)}, Azure: {str(e2)}"
                    if retry < max_retries - 1:
                        continue
                    raise RuntimeError(f"Failed to generate embedding for chunk {idx} after {max_retries} retries: {last_error}")
        
        # Create chunk data with validation
        chunk_data = {
            "content": chunk_text,
            "embedding": embedding,
            "token_count": token_count,
            "metadata": {
                "chunk_index": idx,
                "source_file": filename,
                "content_type": content_type,
                **(metadata or {})
            }
        }
        processed_chunks.append(chunk_data)
    
    if not processed_chunks:
        raise ValueError("No valid chunks were processed")
    
    return processed_chunks

def extract_metadata(
    filename: str,
    content_type: str,
    file_size: int
) -> Dict:
    """
    Extract basic metadata from file information.
    
    Args:
        filename: Name of the file
        content_type: MIME type of the file
        file_size: Size of the file in bytes
    
    Returns:
        Dictionary containing metadata
    """
    return {
        "filename": filename,
        "content_type": content_type,
        "file_size": file_size,
        "file_extension": Path(filename).suffix.lower(),
        "processing_timestamp": str(datetime.utcnow())
    }
