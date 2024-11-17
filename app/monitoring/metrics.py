from typing import Dict, Any, Optional
import time
import psutil
import threading
from datetime import datetime
import numpy as np
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    CollectorRegistry, generate_latest
)

from config.settings import settings
from .logger import logger

class MetricsCollector:
    """Collects and manages system and application metrics."""
    
    def __init__(self):
        """Initialize metrics collector with Prometheus metrics."""
        self.registry = CollectorRegistry()
        
        # Request metrics
        self.request_count = Counter(
            'app_request_total',
            'Total request count',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            'app_request_latency_seconds',
            'Request latency in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Database metrics
        self.db_query_count = Counter(
            'app_db_query_total',
            'Total database query count',
            ['operation_type'],
            registry=self.registry
        )
        
        self.db_query_latency = Histogram(
            'app_db_query_latency_seconds',
            'Database query latency in seconds',
            ['operation_type'],
            registry=self.registry
        )
        
        # Vector search metrics
        self.vector_search_count = Counter(
            'app_vector_search_total',
            'Total vector similarity searches',
            registry=self.registry
        )
        
        self.vector_search_latency = Histogram(
            'app_vector_search_latency_seconds',
            'Vector search latency in seconds',
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hit_count = Counter(
            'app_cache_hit_total',
            'Total cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_miss_count = Counter(
            'app_cache_miss_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'app_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'app_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'app_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
        
        # Document processing metrics
        self.document_process_count = Counter(
            'app_document_process_total',
            'Total documents processed',
            ['status'],
            registry=self.registry
        )
        
        self.document_process_latency = Histogram(
            'app_document_process_latency_seconds',
            'Document processing latency in seconds',
            registry=self.registry
        )
        
        # Start system metrics collection
        self._start_system_metrics_collection()
    
    def _start_system_metrics_collection(self):
        """Start background thread for collecting system metrics."""
        def collect_metrics():
            while True:
                try:
                    # Collect CPU usage
                    self.cpu_usage.set(psutil.cpu_percent())
                    
                    # Collect memory usage
                    memory = psutil.virtual_memory()
                    self.memory_usage.set(memory.used)
                    
                    # Collect disk usage
                    disk = psutil.disk_usage('/')
                    self.disk_usage.set(disk.percent)
                    
                    time.sleep(settings.METRICS_COLLECTION_INTERVAL)
                    
                except Exception as e:
                    logger.error(
                        "Error collecting system metrics",
                        exc_info=e
                    )
        
        thread = threading.Thread(
            target=collect_metrics,
            daemon=True,
            name="MetricsCollector"
        )
        thread.start()
    
    def track_request(
        self,
        method: str,
        endpoint: str,
        status: int,
        duration: float
    ) -> None:
        """
        Track HTTP request metrics.
        
        Args:
            method: HTTP method
            endpoint: Request endpoint
            status: Response status code
            duration: Request duration in seconds
        """
        try:
            self.request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
            
            self.request_latency.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
        except Exception as e:
            logger.error(
                "Error tracking request metrics",
                extra={
                    "method": method,
                    "endpoint": endpoint,
                    "status": status
                },
                exc_info=e
            )
    
    def track_db_query(
        self,
        operation_type: str,
        duration: float
    ) -> None:
        """
        Track database query metrics.
        
        Args:
            operation_type: Type of database operation
            duration: Query duration in seconds
        """
        try:
            self.db_query_count.labels(
                operation_type=operation_type
            ).inc()
            
            self.db_query_latency.labels(
                operation_type=operation_type
            ).observe(duration)
            
        except Exception as e:
            logger.error(
                "Error tracking database metrics",
                extra={"operation_type": operation_type},
                exc_info=e
            )
    
    def track_vector_search(self, duration: float) -> None:
        """
        Track vector similarity search metrics.
        
        Args:
            duration: Search duration in seconds
        """
        try:
            self.vector_search_count.inc()
            self.vector_search_latency.observe(duration)
            
        except Exception as e:
            logger.error(
                "Error tracking vector search metrics",
                exc_info=e
            )
    
    def track_cache(
        self,
        cache_type: str,
        hit: bool
    ) -> None:
        """
        Track cache hit/miss metrics.
        
        Args:
            cache_type: Type of cache
            hit: Whether cache hit or miss
        """
        try:
            if hit:
                self.cache_hit_count.labels(
                    cache_type=cache_type
                ).inc()
            else:
                self.cache_miss_count.labels(
                    cache_type=cache_type
                ).inc()
                
        except Exception as e:
            logger.error(
                "Error tracking cache metrics",
                extra={"cache_type": cache_type},
                exc_info=e
            )
    
    def track_document_processing(
        self,
        duration: float,
        success: bool
    ) -> None:
        """
        Track document processing metrics.
        
        Args:
            duration: Processing duration in seconds
            success: Whether processing succeeded
        """
        try:
            status = "success" if success else "failure"
            self.document_process_count.labels(
                status=status
            ).inc()
            
            self.document_process_latency.observe(duration)
            
        except Exception as e:
            logger.error(
                "Error tracking document processing metrics",
                exc_info=e
            )
    
    def get_metrics(self) -> bytes:
        """
        Get current metrics in Prometheus format.
        
        Returns:
            Metrics data as bytes
        """
        return generate_latest(self.registry)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get human-readable metrics summary.
        
        Returns:
            Dictionary of metric summaries
        """
        try:
            return {
                "requests": {
                    "total": sum(
                        self.request_count._value.values()
                    ),
                    "latency_avg": np.mean(
                        list(self.request_latency._sum.values())
                    )
                },
                "database": {
                    "queries": sum(
                        self.db_query_count._value.values()
                    ),
                    "latency_avg": np.mean(
                        list(self.db_query_latency._sum.values())
                    )
                },
                "vector_search": {
                    "total": self.vector_search_count._value,
                    "latency_avg": np.mean(
                        list(self.vector_search_latency._sum.values())
                    )
                },
                "cache": {
                    "hits": sum(
                        self.cache_hit_count._value.values()
                    ),
                    "misses": sum(
                        self.cache_miss_count._value.values()
                    )
                },
                "system": {
                    "cpu_usage": self.cpu_usage._value,
                    "memory_usage": self.memory_usage._value,
                    "disk_usage": self.disk_usage._value
                },
                "document_processing": {
                    "total": sum(
                        self.document_process_count._value.values()
                    ),
                    "latency_avg": np.mean(
                        list(self.document_process_latency._sum.values())
                    )
                }
            }
            
        except Exception as e:
            logger.error(
                "Error generating metrics summary",
                exc_info=e
            )
            return {}

# Global metrics collector instance
metrics_collector = MetricsCollector()
