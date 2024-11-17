from typing import List, Optional, Union
import numpy as np
import openai
from openai import AzureOpenAI
import tiktoken
from config.settings import settings
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize OpenAI clients
openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)

azure_client = None
if settings.AZURE_OPENAI_API_KEY:
    azure_client = AzureOpenAI(
        api_key=settings.AZURE_OPENAI_API_KEY,
        api_version=settings.AZURE_OPENAI_API_VERSION,
        azure_endpoint=settings.AZURE_OPENAI_API_BASE
    )

def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
    return len(encoding.encode(text))

def truncate_text(text: str, max_tokens: int = 8191) -> str:
    """Truncate text to fit within token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)

def generate_embeddings(
    text: str,
    model: str = "text-embedding-ada-002",
    max_tokens: int = 8191,
    use_azure: bool = False
) -> List[float]:
    """
    Generate embeddings for text using OpenAI's API.
    
    Args:
        text: The text to generate embeddings for
        model: The model to use for embedding generation
        max_tokens: Maximum number of tokens to process
        use_azure: Whether to use Azure OpenAI endpoint
    
    Returns:
        List of embedding values
    """
    # Validate inputs
    if not text or not isinstance(text, str):
        raise ValueError("Text must be a non-empty string")
    
    # Validate clients
    if use_azure and not azure_client:
        logger.warning("Azure client not configured. Falling back to OpenAI.")
        use_azure = False
    
    if not openai_client:
        raise RuntimeError("OpenAI client not configured. Please set OPENAI_API_KEY.")
    
    # Truncate text if necessary
    if count_tokens(text) > max_tokens:
        text = truncate_text(text, max_tokens)
    
    # Embedding generation with enhanced error handling
    error_messages = []
    
    # Try OpenAI first
    try:
        response = openai_client.embeddings.create(
            model=model,
            input=text
        )
        embedding = response.data[0].embedding
        
        # Validate embedding
        if not embedding or len(embedding) != settings.VECTOR_DIMENSION:
            error_messages.append(f"Invalid OpenAI embedding: length {len(embedding)}")
        else:
            return embedding
    except Exception as e:
        error_messages.append(f"OpenAI embedding error: {str(e)}")
        logger.error(f"OpenAI embedding error: {str(e)}")
    
    # Fallback to Azure if configured
    if use_azure and azure_client:
        try:
            response = azure_client.embeddings.create(
                model=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
                input=text
            )
            embedding = response.data[0].embedding
            
            # Validate embedding
            if not embedding or len(embedding) != settings.VECTOR_DIMENSION:
                error_messages.append(f"Invalid Azure embedding: length {len(embedding)}")
            else:
                return embedding
        except Exception as e:
            error_messages.append(f"Azure embedding error: {str(e)}")
            logger.error(f"Azure embedding error: {str(e)}")
    
    # If all attempts fail
    raise RuntimeError(f"Embedding generation failed. Errors: {'; '.join(error_messages)}")

def batch_generate_embeddings(
    texts: List[str],
    batch_size: int = 100,
    **kwargs
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to generate embeddings for
        batch_size: Number of texts to process in each batch
        **kwargs: Additional arguments to pass to generate_embeddings
    
    Returns:
        List of embedding vectors
    """
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = [generate_embeddings(text, **kwargs) for text in batch]
        embeddings.extend(batch_embeddings)
    
    return embeddings

def compute_similarity(
    embedding1: List[float],
    embedding2: List[float]
) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
    
    Returns:
        Cosine similarity score between 0 and 1
    """
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Compute cosine similarity
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    return float(similarity)

def average_embeddings(embeddings: List[List[float]]) -> List[float]:
    """
    Compute the average of multiple embeddings.
    
    Args:
        embeddings: List of embedding vectors
    
    Returns:
        Average embedding vector
    """
    if not embeddings:
        raise ValueError("Empty embeddings list")
    
    # Convert to numpy array and compute mean
    avg_embedding = np.mean(np.array(embeddings), axis=0)
    
    return avg_embedding.tolist()

def generate_hybrid_embedding(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
    **kwargs
) -> List[float]:
    """
    Generate a hybrid embedding by averaging embeddings of overlapping chunks.
    Useful for long texts that exceed token limits.
    
    Args:
        text: Text to generate hybrid embedding for
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        **kwargs: Additional arguments to pass to generate_embeddings
    
    Returns:
        Hybrid embedding vector
    """
    # Split text into overlapping chunks
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    # Generate embeddings for each chunk
    chunk_embeddings = batch_generate_embeddings(chunks, **kwargs)
    
    # Average the embeddings
    return average_embeddings(chunk_embeddings)
