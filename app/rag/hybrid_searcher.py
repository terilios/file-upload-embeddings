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
    """Advanced hybrid search with machine learning-based query classification."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        cache: Optional[RedisCache] = None,
        bm25_weight: float = 0.3,
        vector_weight: float = 0.7,
        reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        """
        Initialize hybrid searcher with enhanced capabilities.
        
        Args:
            vector_store: Vector store instance
            cache: Optional Redis cache instance
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector similarity scores
            reranking_model: Model to use for reranking
        """
        self.vector_store = vector_store
        self.cache = cache or RedisCache()
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        
        # Initialize components
        self.bm25 = None
        self.documents = []
        self.doc_ids = []
        
        # Load reranking model
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranking_model)
        except Exception as e:
            logger.warning(f"Failed to load reranking model: {str(e)}")
            self.reranker = None
        
        # Initialize query classifier
        self.query_classifier = self._initialize_query_classifier()
    
    def _initialize_query_classifier(self):
        """Initialize the query classifier model."""
        try:
            from transformers import pipeline
            return pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1  # CPU
            )
        except Exception as e:
            logger.warning(f"Failed to load query classifier: {str(e)}")
            return None
    
    async def classify_query(self, query: str) -> Dict[str, float]:
        """
        Classify query using zero-shot learning.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary of query type probabilities
        """
        if not self.query_classifier:
            return {"general": 1.0}
        
        candidate_labels = [
            "factual",
            "conceptual",
            "procedural",
            "comparative"
        ]
        
        try:
            result = self.query_classifier(
                query,
                candidate_labels,
                multi_label=True
            )
            return dict(zip(result['labels'], result['scores']))
        except Exception as e:
            logger.error(f"Query classification failed: {str(e)}")
            return {"general": 1.0}
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Enhanced search with query classification and reranking.
        
        Args:
            query: Search query
            limit: Maximum number of results
            min_score: Minimum similarity score
        
        Returns:
            List of search results with scores
        """
        # Classify query
        query_types = await self.classify_query(query)
        logger.info(f"Query classification: {query_types}")
        
        # Adjust weights based on query type
        weights = self._compute_weights(query_types)
        
        # Get initial results
        vector_results = await self._vector_search(query, limit * 2)
        bm25_results = self._bm25_search(query, limit * 2)
        
        # Combine results with dynamic weights
        combined_results = self._combine_results(
            vector_results,
            bm25_results,
            weights['vector'],
            weights['bm25']
        )
        
        # Filter by minimum score
        filtered_results = [
            r for r in combined_results
            if r['score'] >= min_score
        ]
        
        # Rerank if available
        if self.reranker and filtered_results:
            reranked_results = await self._rerank_results(
                query,
                filtered_results[:limit * 2]
            )
            filtered_results = reranked_results
        
        return filtered_results[:limit]
    
    def _compute_weights(self, query_types: Dict[str, float]) -> Dict[str, float]:
        """Compute search weights based on query classification."""
        weights = {
            'vector': self.vector_weight,
            'bm25': self.bm25_weight
        }
        
        # Adjust weights based on query type probabilities
        if 'factual' in query_types:
            # Favor exact matches for factual queries
            factor = query_types['factual']
            weights['bm25'] += 0.1 * factor
            weights['vector'] -= 0.1 * factor
        
        if 'conceptual' in query_types:
            # Favor semantic search for conceptual queries
            factor = query_types['conceptual']
            weights['vector'] += 0.1 * factor
            weights['bm25'] -= 0.1 * factor
        
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    
    async def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder model."""
        if not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, r['content']) for r in results]
        
        try:
            # Get cross-encoder scores
            scores = self.reranker.predict(pairs)
            
            # Combine with original scores
            for result, new_score in zip(results, scores):
                result['score'] = 0.7 * new_score + 0.3 * result['score']
            
            # Sort by new scores
            results.sort(key=lambda x: x['score'], reverse=True)
            
        except Exception as e:
            logger.error(f"Reranking failed: {str(e)}")
        
        return results
    
    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict[str, Any]]:
        """Combine and normalize search results."""
        combined = {}
        
        # Process vector results
        for result in vector_results:
            doc_id = result['id']
            combined[doc_id] = {
                **result,
                'score': result['score'] * vector_weight
            }
        
        # Process BM25 results
        for result in bm25_results:
            doc_id = result['id']
            if doc_id in combined:
                combined[doc_id]['score'] += result['score'] * bm25_weight
            else:
                combined[doc_id] = {
                    **result,
                    'score': result['score'] * bm25_weight
                }
        
        # Convert to list and sort
        results = list(combined.values())
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results
    
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
    
    def _bm25_search(
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
    
    async def _vector_search(
        self,
        query: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Get vector similarity scores for query."""
        try:
            # Generate query embedding
            query_embedding = await generate_embeddings(query)
            
            # Perform vector similarity search
            results = await self.vector_store.similarity_search(
                query_embedding,
                top_k=top_k
            )
            
            return [
                (result[0].document_id, float(result[1]))
                for result in results
            ]
            
        except Exception as e:
            logger.error(f"Error getting vector scores: {str(e)}")
            return []

# Global hybrid searcher instance will be initialized with vector store
hybrid_searcher = None
