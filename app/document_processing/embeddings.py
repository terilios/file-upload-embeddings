from typing import List, Optional, Union
import numpy as np
import openai
from openai import AzureOpenAI
import tiktoken
from config.settings import settings
import logging
import asyncio

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

class EmbeddingsGenerator:
    """Enhanced embeddings generation with advanced chunking and API handling."""
    
    def __init__(self):
        self.openai_client = openai_client
        self.azure_client = azure_client
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.retry_config = {
            'max_retries': 3,
            'base_delay': 1,
            'max_delay': 10
        }
        
    async def generate_with_retry(
        self,
        text: str,
        model: str = "text-embedding-ada-002",
        use_azure: bool = False
    ) -> List[float]:
        """Generate embeddings with retry logic."""
        retries = 0
        while retries < self.retry_config['max_retries']:
            try:
                client = self.azure_client if use_azure else self.openai_client
                response = await self._make_embedding_request(client, text, model)
                return response.data[0].embedding
            except Exception as e:
                retries += 1
                if retries == self.retry_config['max_retries']:
                    logger.error(f"Failed to generate embeddings after {retries} retries: {str(e)}")
                    raise
                delay = min(
                    self.retry_config['base_delay'] * (2 ** retries),
                    self.retry_config['max_delay']
                )
                logger.warning(f"Retry {retries} for embedding generation after {delay}s: {str(e)}")
                await asyncio.sleep(delay)

    async def _make_embedding_request(
        self,
        client: Union[openai.OpenAI, AzureOpenAI],
        text: str,
        model: str
    ):
        """Make the actual API request."""
        return await client.embeddings.create(
            input=text,
            model=model
        )

    def smart_chunk_text(
        self,
        text: str,
        max_tokens: int = 8191,
        overlap_ratio: float = 0.1
    ) -> List[str]:
        """
        Intelligently chunk text based on natural boundaries and token limits.
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_ratio: Ratio of overlap between chunks
        
        Returns:
            List of text chunks
        """
        # First split on natural boundaries
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        overlap_tokens = int(max_tokens * overlap_ratio)
        
        for para in paragraphs:
            para_tokens = self.encoding.encode(para)
            para_token_count = len(para_tokens)
            
            if current_tokens + para_token_count > max_tokens:
                # Create chunk from current paragraphs
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if chunks:
                    overlap_text = chunks[-1].split('\n\n')[-2:]
                    current_chunk = overlap_text
                    current_tokens = len(self.encoding.encode('\n\n'.join(current_chunk)))
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_token_count
        
        # Add final chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks

    async def generate_chunked_embeddings(
        self,
        text: str,
        max_tokens: int = 8191,
        overlap_ratio: float = 0.1,
        **kwargs
    ) -> List[List[float]]:
        """
        Generate embeddings for chunked text with parallel processing.
        
        Args:
            text: Text to generate embeddings for
            max_tokens: Maximum tokens per chunk
            overlap_ratio: Ratio of overlap between chunks
            **kwargs: Additional arguments for embedding generation
        
        Returns:
            List of embedding vectors for each chunk
        """
        chunks = self.smart_chunk_text(text, max_tokens, overlap_ratio)
        
        tasks = [
            self.generate_with_retry(chunk, **kwargs)
            for chunk in chunks
        ]
        
        return await asyncio.gather(*tasks)

    def aggregate_embeddings(
        self,
        embeddings: List[List[float]],
        weights: Optional[List[float]] = None
    ) -> List[float]:
        """
        Aggregate multiple embeddings with optional weighting.
        
        Args:
            embeddings: List of embedding vectors
            weights: Optional weights for each embedding
        
        Returns:
            Aggregated embedding vector
        """
        if not embeddings:
            raise ValueError("No embeddings to aggregate")
        
        if weights is None:
            weights = [1.0] * len(embeddings)
        elif len(weights) != len(embeddings):
            raise ValueError("Number of weights must match number of embeddings")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Convert to numpy for efficient computation
        embeddings_array = np.array(embeddings)
        weights_array = np.array(weights).reshape(-1, 1)
        
        # Weighted average
        aggregated = np.sum(embeddings_array * weights_array, axis=0)
        
        # Normalize
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        
        return aggregated.tolist()

# Global embeddings generator instance
embeddings_generator = EmbeddingsGenerator()

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

async def generate_embeddings(
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
        response = await embeddings_generator.generate_with_retry(text, model, use_azure)
        return response
    except Exception as e:
        error_messages.append(f"OpenAI embedding error: {str(e)}")
        logger.error(f"OpenAI embedding error: {str(e)}")
    
    # Fallback to Azure if configured
    if use_azure and azure_client:
        try:
            response = await embeddings_generator.generate_with_retry(text, model, use_azure)
            return response
        except Exception as e:
            error_messages.append(f"Azure embedding error: {str(e)}")
            logger.error(f"Azure embedding error: {str(e)}")
    
    # If all attempts fail
    raise RuntimeError(f"Embedding generation failed. Errors: {'; '.join(error_messages)}")

async def batch_generate_embeddings(
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
        batch_embeddings = [await generate_embeddings(text, **kwargs) for text in batch]
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

async def generate_hybrid_embedding(
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
    chunk_embeddings = await batch_generate_embeddings(chunks, **kwargs)
    
    # Average the embeddings
    return average_embeddings(chunk_embeddings)
