from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import logging
from datetime import datetime

from app.document_processing.embeddings import generate_embeddings
from app.cache.redis_cache import RedisCache
from config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ContextWindow:
    """Represents a context window with relevance scoring."""
    chunks: List[Dict[str, Any]]
    start_idx: int
    end_idx: int
    relevance_score: float
    metadata: Dict[str, Any]

class ContextManager:
    """Manage dynamic context windows for retrieval."""
    
    def __init__(
        self,
        cache: Optional[RedisCache] = None,
        min_window_size: int = 2,
        max_window_size: int = 8,
        overlap_size: int = 1,
        relevance_threshold: float = 0.6
    ):
        """
        Initialize context manager.
        
        Args:
            cache: Optional Redis cache instance
            min_window_size: Minimum window size in chunks
            max_window_size: Maximum window size in chunks
            overlap_size: Number of chunks to overlap between windows
            relevance_threshold: Minimum relevance score threshold
        """
        self.cache = cache or RedisCache()
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.overlap_size = overlap_size
        self.relevance_threshold = relevance_threshold
    
    async def create_context_windows(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: List[float]
    ) -> List[ContextWindow]:
        """
        Create dynamic context windows from chunks.
        
        Args:
            chunks: List of document chunks
            query_embedding: Query embedding vector
        
        Returns:
            List of context windows with relevance scores
        """
        try:
            # Check cache first
            cache_key = f"context_windows:{hash(str(chunks))}:{hash(str(query_embedding))}"
            cached_windows = await self.cache.get(cache_key)
            if cached_windows:
                return [ContextWindow(**w) for w in cached_windows]
            
            windows = []
            chunk_count = len(chunks)
            
            # Calculate chunk relevance scores
            chunk_scores = await self._calculate_chunk_scores(
                chunks,
                query_embedding
            )
            
            # Create initial windows with minimum size
            for i in range(0, chunk_count - self.min_window_size + 1):
                window = await self._create_window(
                    chunks[i:i + self.min_window_size],
                    chunk_scores[i:i + self.min_window_size],
                    i,
                    i + self.min_window_size - 1
                )
                if window.relevance_score >= self.relevance_threshold:
                    windows.append(window)
            
            # Expand promising windows
            expanded_windows = []
            for window in windows:
                expanded = await self._expand_window(
                    chunks,
                    chunk_scores,
                    window
                )
                expanded_windows.append(expanded)
            
            # Merge overlapping windows
            merged_windows = await self._merge_windows(expanded_windows)
            
            # Sort by relevance score
            merged_windows.sort(
                key=lambda w: w.relevance_score,
                reverse=True
            )
            
            # Cache results
            await self.cache.set(
                cache_key,
                [
                    {
                        "chunks": w.chunks,
                        "start_idx": w.start_idx,
                        "end_idx": w.end_idx,
                        "relevance_score": w.relevance_score,
                        "metadata": w.metadata
                    }
                    for w in merged_windows
                ],
                ttl=settings.CACHE_DEFAULT_TIMEOUT
            )
            
            return merged_windows
            
        except Exception as e:
            logger.error(f"Error creating context windows: {str(e)}")
            return []
    
    async def _calculate_chunk_scores(
        self,
        chunks: List[Dict[str, Any]],
        query_embedding: List[float]
    ) -> List[float]:
        """Calculate relevance scores for individual chunks."""
        try:
            scores = []
            query_embedding = np.array(query_embedding)
            
            for chunk in chunks:
                chunk_embedding = np.array(chunk["embedding"])
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) *
                    np.linalg.norm(chunk_embedding)
                )
                scores.append(float(similarity))
            
            return scores
            
        except Exception as e:
            logger.error(f"Error calculating chunk scores: {str(e)}")
            return [0.0] * len(chunks)
    
    async def _create_window(
        self,
        window_chunks: List[Dict[str, Any]],
        chunk_scores: List[float],
        start_idx: int,
        end_idx: int
    ) -> ContextWindow:
        """Create context window with metadata."""
        try:
            # Calculate window relevance score
            # Use weighted average based on position
            weights = np.linspace(0.8, 1.0, len(chunk_scores))
            relevance_score = np.average(chunk_scores, weights=weights)
            
            # Generate window metadata
            metadata = {
                "chunk_count": len(window_chunks),
                "avg_chunk_length": np.mean([
                    len(chunk["content"]) for chunk in window_chunks
                ]),
                "source_documents": list(set(
                    chunk.get("metadata", {}).get("source_id")
                    for chunk in window_chunks
                    if chunk.get("metadata")
                )),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return ContextWindow(
                chunks=window_chunks,
                start_idx=start_idx,
                end_idx=end_idx,
                relevance_score=relevance_score,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating window: {str(e)}")
            return ContextWindow(
                chunks=window_chunks,
                start_idx=start_idx,
                end_idx=end_idx,
                relevance_score=0.0,
                metadata={}
            )
    
    async def _expand_window(
        self,
        all_chunks: List[Dict[str, Any]],
        chunk_scores: List[float],
        window: ContextWindow
    ) -> ContextWindow:
        """Expand window if beneficial."""
        try:
            current_score = window.relevance_score
            best_window = window
            
            # Try expanding in both directions
            for direction in ["left", "right"]:
                expanded_window = await self._try_expand(
                    all_chunks,
                    chunk_scores,
                    window,
                    direction
                )
                
                if expanded_window and expanded_window.relevance_score > current_score:
                    best_window = expanded_window
                    current_score = expanded_window.relevance_score
            
            return best_window
            
        except Exception as e:
            logger.error(f"Error expanding window: {str(e)}")
            return window
    
    async def _try_expand(
        self,
        all_chunks: List[Dict[str, Any]],
        chunk_scores: List[float],
        window: ContextWindow,
        direction: str
    ) -> Optional[ContextWindow]:
        """Try expanding window in specified direction."""
        try:
            if direction == "left" and window.start_idx > 0:
                new_start = max(0, window.start_idx - self.overlap_size)
                new_chunks = all_chunks[new_start:window.end_idx + 1]
                new_scores = chunk_scores[new_start:window.end_idx + 1]
                
                if len(new_chunks) <= self.max_window_size:
                    return await self._create_window(
                        new_chunks,
                        new_scores,
                        new_start,
                        window.end_idx
                    )
                    
            elif direction == "right" and window.end_idx < len(all_chunks) - 1:
                new_end = min(
                    len(all_chunks) - 1,
                    window.end_idx + self.overlap_size
                )
                new_chunks = all_chunks[window.start_idx:new_end + 1]
                new_scores = chunk_scores[window.start_idx:new_end + 1]
                
                if len(new_chunks) <= self.max_window_size:
                    return await self._create_window(
                        new_chunks,
                        new_scores,
                        window.start_idx,
                        new_end
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error trying window expansion: {str(e)}")
            return None
    
    async def _merge_windows(
        self,
        windows: List[ContextWindow]
    ) -> List[ContextWindow]:
        """Merge overlapping windows."""
        try:
            if not windows:
                return []
            
            # Sort windows by start index
            sorted_windows = sorted(windows, key=lambda w: w.start_idx)
            merged = [sorted_windows[0]]
            
            for current in sorted_windows[1:]:
                previous = merged[-1]
                
                # Check for overlap
                if current.start_idx <= previous.end_idx + self.overlap_size:
                    # Merge windows
                    merged_chunks = previous.chunks + current.chunks[
                        previous.end_idx - current.start_idx + 1:
                    ]
                    
                    # Create new merged window
                    merged_window = await self._create_window(
                        merged_chunks,
                        [chunk.get("score", 0.0) for chunk in merged_chunks],
                        previous.start_idx,
                        current.end_idx
                    )
                    
                    merged[-1] = merged_window
                else:
                    merged.append(current)
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging windows: {str(e)}")
            return windows

# Global context manager instance
context_manager = ContextManager()
