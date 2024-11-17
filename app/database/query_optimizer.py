from typing import List, Dict, Any, Optional
from sqlalchemy import text
import numpy as np
import logging
from datetime import datetime

from config.settings import settings
from .connection_manager import connection_manager
from app.cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)

class QueryOptimizer:
    """Optimizes vector similarity queries for better performance."""
    
    def __init__(self):
        """Initialize query optimizer."""
        self.cache = RedisCache()
        self._stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "avg_query_time": 0
        }
    
    async def optimize_similarity_query(
        self,
        query_embedding: List[float],
        top_k: int = settings.TOP_K_RESULTS,
        threshold: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> str:
        """
        Optimize similarity search query.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            threshold: Similarity threshold
            filters: Optional query filters
        
        Returns:
            Optimized SQL query
        """
        # Base query with vector similarity
        base_query = """
            WITH similarity_scores AS (
                SELECT 
                    dc.id,
                    dc.document_id,
                    dc.content,
                    dc.chunk_index,
                    dc.metadata,
                    1 - (dc.embedding <=> :query_embedding::vector) as similarity
                FROM document_chunks dc
                WHERE 1 - (dc.embedding <=> :query_embedding::vector) > :threshold
                {filter_clause}
            )
            SELECT 
                ss.*,
                d.filename,
                d.content_type
            FROM similarity_scores ss
            JOIN documents d ON ss.document_id = d.id
            ORDER BY ss.similarity DESC
            LIMIT :limit
        """
        
        # Add filters if provided
        filter_clause = ""
        if filters:
            conditions = []
            if "document_ids" in filters:
                conditions.append("dc.document_id = ANY(:document_ids)")
            if "content_types" in filters:
                conditions.append("d.content_type = ANY(:content_types)")
            if "date_range" in filters:
                conditions.append(
                    "d.created_at BETWEEN :date_from AND :date_to"
                )
            if conditions:
                filter_clause = "AND " + " AND ".join(conditions)
        
        return base_query.format(filter_clause=filter_clause)
    
    async def execute_similarity_query(
        self,
        query_embedding: List[float],
        session,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute optimized similarity query.
        
        Args:
            query_embedding: Query embedding vector
            session: Database session
            **kwargs: Additional query parameters
        
        Returns:
            List of results with similarity scores
        """
        start_time = datetime.now()
        
        try:
            # Get optimized query
            query = await self.optimize_similarity_query(
                query_embedding,
                **kwargs
            )
            
            # Execute query
            params = {
                "query_embedding": query_embedding,
                "threshold": kwargs.get("threshold", settings.SIMILARITY_THRESHOLD),
                "limit": kwargs.get("top_k", settings.TOP_K_RESULTS)
            }
            
            # Add filter parameters if provided
            filters = kwargs.get("filters", {})
            params.update(filters)
            
            result = session.execute(text(query), params)
            rows = result.fetchall()
            
            # Update statistics
            self._stats["queries_executed"] += 1
            query_time = (datetime.now() - start_time).total_seconds()
            self._update_avg_query_time(query_time)
            
            # Format results
            return [
                {
                    "id": row.id,
                    "document_id": row.document_id,
                    "content": row.content,
                    "chunk_index": row.chunk_index,
                    "metadata": row.metadata,
                    "similarity": float(row.similarity),
                    "filename": row.filename,
                    "content_type": row.content_type
                }
                for row in rows
            ]
            
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}")
            raise
    
    async def analyze_query_performance(
        self,
        query_embedding: List[float],
        session,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze query performance.
        
        Args:
            query_embedding: Query embedding vector
            session: Database session
            **kwargs: Query parameters
        
        Returns:
            Performance analysis results
        """
        # Get query plan
        query = await self.optimize_similarity_query(
            query_embedding,
            **kwargs
        )
        
        explain_query = f"EXPLAIN ANALYZE {query}"
        result = session.execute(
            text(explain_query),
            {
                "query_embedding": query_embedding,
                "threshold": kwargs.get("threshold", settings.SIMILARITY_THRESHOLD),
                "limit": kwargs.get("top_k", settings.TOP_K_RESULTS),
                **(kwargs.get("filters", {}))
            }
        )
        
        plan = result.fetchall()
        
        return {
            "query_plan": [row[0] for row in plan],
            "statistics": self._stats,
            "recommendations": self._generate_recommendations(plan)
        }
    
    def _update_avg_query_time(self, query_time: float):
        """Update average query time statistics."""
        current_avg = self._stats["avg_query_time"]
        queries = self._stats["queries_executed"]
        self._stats["avg_query_time"] = (
            (current_avg * (queries - 1) + query_time) / queries
        )
    
    def _generate_recommendations(self, query_plan: List) -> List[str]:
        """Generate query optimization recommendations."""
        recommendations = []
        
        # Analyze query plan
        plan_text = "\n".join(row[0] for row in query_plan)
        
        if "Seq Scan" in plan_text:
            recommendations.append(
                "Consider adding an index to avoid sequential scans"
            )
        
        if "Bitmap Heap Scan" in plan_text:
            recommendations.append(
                "Consider increasing work_mem for better bitmap heap scan performance"
            )
        
        if self._stats["avg_query_time"] > 1.0:
            recommendations.append(
                "Consider implementing query result caching"
            )
        
        return recommendations
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """
        Get query optimization statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self._stats,
            "cache_hit_ratio": (
                self._stats["cache_hits"] / self._stats["queries_executed"]
                if self._stats["queries_executed"] > 0 else 0
            )
        }
    
    async def reset_stats(self):
        """Reset optimization statistics."""
        self._stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "avg_query_time": 0
        }

# Global query optimizer instance
query_optimizer = QueryOptimizer()
