from typing import Optional, Any, List
import json
import pickle
from datetime import timedelta, datetime
import redis
import numpy as np
from functools import wraps

from config.settings import settings

class RedisCache:
    """Redis cache implementation for storing embeddings and query results."""
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis = redis.Redis.from_url(
            settings.CACHE_REDIS_URL,
            decode_responses=True,
            encoding='utf-8',
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30
        )
        # Separate connection for binary data (embeddings)
        self.binary_redis = redis.Redis.from_url(
            settings.CACHE_REDIS_URL,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
            health_check_interval=30
        )
        self.default_ttl = settings.CACHE_DEFAULT_TIMEOUT
        self._init_cache_metrics()
    
    def _init_cache_metrics(self):
        """Initialize cache metrics for monitoring."""
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self.last_error = None
        self.last_error_time = None
    
    def _get_embedding_key(self, text: str) -> str:
        """Generate cache key for embeddings with namespace."""
        return f"emb:{hash(text)}"
    
    def _get_query_key(self, query: str, doc_id: Optional[int] = None) -> str:
        """Generate cache key for query results with namespace."""
        return f"q:{hash(query)}:d:{doc_id}" if doc_id else f"q:{hash(query)}"
    
    def _get_document_key(self, doc_id: int) -> str:
        """Generate cache key for document metadata with namespace."""
        return f"doc:{doc_id}"
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache with error handling and metrics.
        
        Args:
            text: Text to get embedding for
        
        Returns:
            Embedding vector if cached, None otherwise
        """
        key = self._get_embedding_key(text)
        try:
            data = self.binary_redis.get(key)
            if data:
                self.hits += 1
                return pickle.loads(data)
            self.misses += 1
            return None
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return None
    
    async def set_embedding(
        self,
        text: str,
        embedding: List[float],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store embedding in cache with compression and validation.
        
        Args:
            text: Text the embedding is for
            embedding: Embedding vector
            ttl: Time to live in seconds
        
        Returns:
            True if successful, False otherwise
        """
        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            return False
            
        key = self._get_embedding_key(text)
        try:
            data = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
            return self.binary_redis.set(
                key,
                data,
                ex=ttl or self.default_ttl,
                nx=True  # Only set if not exists
            )
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return False
    
    async def get_query_result(
        self,
        query: str,
        doc_id: Optional[int] = None
    ) -> Optional[dict]:
        """
        Get cached query result with compression and metrics.
        
        Args:
            query: Query string
            doc_id: Optional document ID
        
        Returns:
            Cached result if exists, None otherwise
        """
        key = self._get_query_key(query, doc_id)
        try:
            data = self.redis.get(key)
            if data:
                self.hits += 1
                return json.loads(data)
            self.misses += 1
            return None
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return None
    
    async def set_query_result(
        self,
        query: str,
        result: dict,
        doc_id: Optional[int] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store query result in cache with compression.
        
        Args:
            query: Query string
            result: Query result to cache
            doc_id: Optional document ID
            ttl: Time to live in seconds
        
        Returns:
            True if successful, False otherwise
        """
        key = self._get_query_key(query, doc_id)
        try:
            data = json.dumps(result, ensure_ascii=False)
            return self.redis.set(
                key,
                data,
                ex=ttl or self.default_ttl,
                nx=True  # Only set if not exists
            )
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return False
    
    async def get_document_metadata(self, doc_id: int) -> Optional[dict]:
        """
        Get cached document metadata.
        
        Args:
            doc_id: Document ID
        
        Returns:
            Document metadata if cached, None otherwise
        """
        key = self._get_document_key(doc_id)
        try:
            data = self.redis.get(key)
            if data:
                self.hits += 1
                return json.loads(data)
            self.misses += 1
            return None
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return None
    
    async def set_document_metadata(
        self,
        doc_id: int,
        metadata: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Store document metadata in cache.
        
        Args:
            doc_id: Document ID
            metadata: Document metadata
            ttl: Time to live in seconds
        
        Returns:
            True if successful, False otherwise
        """
        key = self._get_document_key(doc_id)
        try:
            data = json.dumps(metadata, ensure_ascii=False)
            return self.redis.set(
                key,
                data,
                ex=ttl or self.default_ttl,
                nx=True  # Only set if not exists
            )
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return False
    
    async def invalidate_document(self, doc_id: int) -> bool:
        """
        Invalidate all cache entries for a document.
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete document metadata
            self.redis.delete(self._get_document_key(doc_id))
            
            # Delete all query results for this document
            pattern = f"q:*:d:{doc_id}"
            keys = self.redis.keys(pattern)
            if keys:
                self.redis.delete(*keys)
            
            return True
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return False
    
    async def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.redis.flushdb()
            self.binary_redis.flushdb()
            return True
        except Exception as e:
            self.errors += 1
            self.last_error = str(e)
            self.last_error_time = datetime.utcnow()
            return False
    
    def get_metrics(self) -> dict:
        """Get cache performance metrics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": hit_rate,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None
        }

def cache_embedding(ttl: Optional[int] = None):
    """
    Decorator for caching embeddings.
    
    Args:
        ttl: Optional time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(text: str, *args, **kwargs):
            cache = RedisCache()
            
            # Try to get from cache
            cached = await cache.get_embedding(text)
            if cached is not None:
                return cached
            
            # Generate embedding
            embedding = await func(text, *args, **kwargs)
            
            # Cache result
            await cache.set_embedding(text, embedding, ttl)
            
            return embedding
        return wrapper
    return decorator

def cache_query(ttl: Optional[int] = None):
    """
    Decorator for caching query results.
    
    Args:
        ttl: Optional time to live in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(query: str, doc_id: Optional[int] = None, *args, **kwargs):
            cache = RedisCache()
            
            # Try to get from cache
            cached = await cache.get_query_result(query, doc_id)
            if cached is not None:
                return cached
            
            # Execute query
            result = await func(query, doc_id, *args, **kwargs)
            
            # Cache result
            await cache.set_query_result(query, result, doc_id, ttl)
            
            return result
        return wrapper
    return decorator
