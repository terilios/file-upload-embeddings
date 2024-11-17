from typing import List, Dict, Optional
import re
from pathlib import Path
from datetime import datetime

from config.settings import settings
from .embeddings import count_tokens, generate_embeddings

def determine_chunk_params(
    content_type: str,
    file_size: int
) -> tuple[int, int]:
    """
    Determine appropriate chunk size and overlap based on content type and file size.
    
    Args:
        content_type: MIME type or document type
        file_size: Size of the file in bytes
    
    Returns:
        Tuple of (chunk_size, overlap)
    """
    # Map content type to document type
    doc_type = "default"
    if "email" in content_type or file_size < 50000:  # 50KB
        doc_type = "email"
    elif "pdf" in content_type or "document" in content_type:
        doc_type = "report"
    elif "technical" in content_type or file_size > 1000000:  # 1MB
        doc_type = "technical"
    
    chunk_size = settings.CHUNK_SIZE_MAPPING.get(doc_type, settings.CHUNK_SIZE_MAPPING["default"])
    chunk_overlap = settings.CHUNK_OVERLAP_MAPPING.get(doc_type, settings.CHUNK_OVERLAP_MAPPING["default"])
    
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
    """
    # Determine chunk parameters
    chunk_size, chunk_overlap = determine_chunk_params(content_type, file_size)
    
    # Split content into chunks
    text_chunks = split_text_into_chunks(content, chunk_size, chunk_overlap)
    
    # Process each chunk
    processed_chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        # Count tokens
        token_count = count_tokens(chunk_text)
        
        # Generate embedding
        embedding = generate_embeddings(chunk_text)
        
        # Create chunk metadata
        chunk_metadata = {
            "index": idx,
            "total_chunks": len(text_chunks),
            "token_count": token_count,
            "original_file": filename,
            **(metadata or {})
        }
        
        processed_chunks.append({
            "content": chunk_text,
            "embedding": embedding,
            "token_count": token_count,
            "metadata": chunk_metadata
        })
    
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
