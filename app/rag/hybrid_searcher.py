from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
import logging
from datetime import datetime

from app.database.vector_store import VectorStore
from app.document_processing.embeddings import generate_embeddings
from app.cache.redis_cache import RedisCache
from config.settings import settings

logger = logging.getLogger(__name__)

class HybridSearcher:
    """Combine BM25 and vector similarity search for improved retrieval."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        cache: Optional[RedisCache] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7
    ):
        """
        Initialize hybrid searcher.
        
        Args:
            vector_store: Vector store instance
            cache: Optional Redis cache instance
            bm25_weight: Weight for BM25 scores (default: 0.3)
            vector_weight: Weight for vector similarity scores (default: 0.7)
        """
        self.vector_store = vector_store
        self.cache = cache or RedisCache()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # Initialize BM25 index
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents with content and metadata
        """
        try:
            # Prepare documents for BM25
            self.documents = [doc["content"] for doc in documents]
            self.doc_ids = [doc["id"] for doc in documents]
            
            # Tokenize documents
            tokenized_docs = [doc.split() for doc in self.documents]
            
            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_docs)
            
            logger.info(f"Indexed {len(documents)} documents for BM25 search")
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    async def search(
        self,
        query: str,
        top_k: int = settings.TOP_K_RESULTS,
        threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            threshold: Minimum similarity threshold
            filters: Optional filters for vector search
        
        Returns:
            List of search results with combined scores
        """
        try:
            # Check cache first
            cache_key = f"hybrid_search:{query}:{top_k}:{threshold}:{str(filters)}"
            cached_results = await self.cache.get(cache_key)
            if cached_results:
                return cached_results
            
            # Get BM25 scores
            bm25_scores = await self._get_bm25_scores(query, top_k)
            
            # Get vector similarity scores
            vector_scores = await self._get_vector_scores(
                query,
                top_k,
                threshold,
                filters
            )
            
            # Combine scores
            combined_results = self._combine_scores(
                bm25_scores,
                vector_scores,
                top_k
            )
            
            # Cache results
            await self.cache.set(
                cache_key,
                combined_results,
                ttl=settings.CACHE_DEFAULT_TIMEOUT
            )
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error performing hybrid search: {str(e)}")
            raise
    
    async def _get_bm25_scores(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Get BM25 scores for query."""
        if not self.bm25:
            return []
        
        try:
            # Tokenize query
            tokenized_query = query.split()
            
            # Get BM25 scores
            scores = self.bm25.get_scores(tokenized_query)
            
            # Get top-k document indices and scores
            top_indices = np.argsort(scores)[-top_k:][::-1]
            
            return [
                (self.doc_ids[idx], float(scores[idx]))
                for idx in top_indices
                if scores[idx] > 0
            ]
            
        except Exception as e:
            logger.error(f"Error getting BM25 scores: {str(e)}")
            return []
    
    async def _get_vector_scores(
        self,
        query: str,
        top_k: int,
        threshold: Optional[float],
        filters: Optional[Dict]
    ) -> List[Tuple[int, float]]:
        """Get vector similarity scores for query."""
        try:
            # Generate query embedding
            query_embedding = await generate_embeddings(query)
            
            # Perform vector similarity search
            results = await self.vector_store.similarity_search(
                query_embedding,
                top_k=top_k,
                threshold=threshold,
                filters=filters
            )
            
            return [
                (result[0].document_id, float(result[1]))
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting vector scores: {str(e)}")
            return []
    
    def _combine_scores(
        self,
        bm25_scores: List[Tuple[int, float]],
        vector_scores: List[Tuple[int, float]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Combine BM25 and vector similarity scores.
        
        Uses weighted sum of normalized scores.
        """
        try:
            # Create score dictionaries
            bm25_dict = dict(bm25_scores)
            vector_dict = dict(vector_scores)
            
            # Get all document IDs
            doc_ids = set(bm25_dict.keys()) | set(vector_dict.keys())
            
            # Normalize scores
            if bm25_scores:
                max_bm25 = max(bm25_dict.values())
                min_bm25 = min(bm25_dict.values())
            if vector_scores:
                max_vector = max(vector_dict.values())
                min_vector = min(vector_dict.values())
            
            # Calculate combined scores
            combined_scores = []
            for doc_id in doc_ids:
                # Get normalized scores
                bm25_score = 0.0
                if doc_id in bm25_dict:
                    bm25_score = (bm25_dict[doc_id] - min_bm25) / (max_bm25 - min_bm25) if bm25_scores else 0.0
                
                vector_score = 0.0
                if doc_id in vector_dict:
                    vector_score = (vector_dict[doc_id] - min_vector) / (max_vector - min_vector) if vector_scores else 0.0
                
                # Calculate weighted sum
                combined_score = (
                    self.bm25_weight * bm25_score +
                    self.vector_weight * vector_score
                )
                
                combined_scores.append({
                    "document_id": doc_id,
                    "score": combined_score,
                    "bm25_score": bm25_score,
                    "vector_score": vector_score
                })
            
            # Sort by combined score and get top-k
            combined_scores.sort(key=lambda x: x["score"], reverse=True)
            return combined_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Error combining scores: {str(e)}")
            return []
    
    async def update_weights(
        self,
        bm25_weight: float,
        vector_weight: float
    ) -> None:
        """
        Update scoring weights.
        
        Args:
            bm25_weight: New weight for BM25 scores
            vector_weight: New weight for vector similarity scores
        """
        if not (0 <= bm25_weight <= 1 and 0 <= vector_weight <= 1):
            raise ValueError("Weights must be between 0 and 1")
        
        if abs(bm25_weight + vector_weight - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")
        
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        logger.info(
            f"Updated weights: BM25={bm25_weight}, Vector={vector_weight}"
        )

# Global hybrid searcher instance will be initialized with vector store
hybrid_searcher = None
