import pytest
import time
import threading
from prometheus_client.parser import text_string_to_metric_families

from app.monitoring.metrics import MetricsCollector, metrics_collector

@pytest.fixture
def collector():
    """Create a fresh metrics collector for testing."""
    return MetricsCollector()

def test_request_metrics(collector):
    """Test tracking of HTTP request metrics."""
    # Track some test requests
    collector.track_request("GET", "/api/test", 200, 0.1)
    collector.track_request("POST", "/api/test", 201, 0.2)
    collector.track_request("GET", "/api/test", 404, 0.05)
    
    # Get metrics in Prometheus format
    metrics_data = collector.get_metrics().decode()
    
    # Parse metrics
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify request count
    assert "app_request_total" in metrics
    request_count = metrics["app_request_total"]
    assert sum(s.value for s in request_count.samples) == 3
    
    # Verify request latency
    assert "app_request_latency_seconds" in metrics
    latency = metrics["app_request_latency_seconds"]
    assert any(s.value > 0 for s in latency.samples)

def test_database_metrics(collector):
    """Test tracking of database metrics."""
    # Track some database operations
    collector.track_db_query("select", 0.1)
    collector.track_db_query("insert", 0.2)
    collector.track_db_query("update", 0.15)
    
    metrics_data = collector.get_metrics().decode()
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify query count
    assert "app_db_query_total" in metrics
    query_count = metrics["app_db_query_total"]
    assert sum(s.value for s in query_count.samples) == 3
    
    # Verify query latency
    assert "app_db_query_latency_seconds" in metrics
    latency = metrics["app_db_query_latency_seconds"]
    assert any(s.value > 0 for s in latency.samples)

def test_vector_search_metrics(collector):
    """Test tracking of vector search metrics."""
    # Track vector searches
    collector.track_vector_search(0.3)
    collector.track_vector_search(0.4)
    
    metrics_data = collector.get_metrics().decode()
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify search count
    assert "app_vector_search_total" in metrics
    search_count = metrics["app_vector_search_total"]
    assert sum(s.value for s in search_count.samples) == 2
    
    # Verify search latency
    assert "app_vector_search_latency_seconds" in metrics
    latency = metrics["app_vector_search_latency_seconds"]
    assert any(s.value > 0 for s in latency.samples)

def test_cache_metrics(collector):
    """Test tracking of cache metrics."""
    # Track cache hits and misses
    collector.track_cache("embedding", True)  # Hit
    collector.track_cache("embedding", False)  # Miss
    collector.track_cache("query", True)  # Hit
    
    metrics_data = collector.get_metrics().decode()
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify cache hits
    assert "app_cache_hit_total" in metrics
    hits = metrics["app_cache_hit_total"]
    assert sum(s.value for s in hits.samples) == 2
    
    # Verify cache misses
    assert "app_cache_miss_total" in metrics
    misses = metrics["app_cache_miss_total"]
    assert sum(s.value for s in misses.samples) == 1

def test_system_metrics(collector):
    """Test collection of system metrics."""
    # Wait for system metrics to be collected
    time.sleep(settings.METRICS_COLLECTION_INTERVAL + 0.1)
    
    metrics_data = collector.get_metrics().decode()
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify system metrics
    assert "app_cpu_usage_percent" in metrics
    assert "app_memory_usage_bytes" in metrics
    assert "app_disk_usage_percent" in metrics
    
    # Verify values are reasonable
    cpu = metrics["app_cpu_usage_percent"]
    assert 0 <= next(iter(cpu.samples)).value <= 100
    
    memory = metrics["app_memory_usage_bytes"]
    assert next(iter(memory.samples)).value > 0
    
    disk = metrics["app_disk_usage_percent"]
    assert 0 <= next(iter(disk.samples)).value <= 100

def test_document_processing_metrics(collector):
    """Test tracking of document processing metrics."""
    # Track document processing
    collector.track_document_processing(1.0, True)  # Success
    collector.track_document_processing(0.5, False)  # Failure
    
    metrics_data = collector.get_metrics().decode()
    metrics = {
        m.name: m for m in text_string_to_metric_families(metrics_data)
    }
    
    # Verify processing count
    assert "app_document_process_total" in metrics
    process_count = metrics["app_document_process_total"]
    assert sum(s.value for s in process_count.samples) == 2
    
    # Verify processing latency
    assert "app_document_process_latency_seconds" in metrics
    latency = metrics["app_document_process_latency_seconds"]
    assert any(s.value > 0 for s in latency.samples)

def test_metrics_summary(collector):
    """Test generation of metrics summary."""
    # Generate some test metrics
    collector.track_request("GET", "/api/test", 200, 0.1)
    collector.track_db_query("select", 0.2)
    collector.track_vector_search(0.3)
    collector.track_cache("embedding", True)
    collector.track_document_processing(0.4, True)
    
    # Get summary
    summary = collector.get_metrics_summary()
    
    # Verify summary structure
    assert "requests" in summary
    assert "database" in summary
    assert "vector_search" in summary
    assert "cache" in summary
    assert "system" in summary
    assert "document_processing" in summary
    
    # Verify some values
    assert summary["requests"]["total"] == 1
    assert summary["database"]["queries"] == 1
    assert summary["vector_search"]["total"] == 1
    assert summary["cache"]["hits"] == 1

def test_concurrent_metric_tracking(collector):
    """Test concurrent metric tracking."""
    def track_metrics():
        for _ in range(100):
            collector.track_request("GET", "/api/test", 200, 0.1)
            collector.track_db_query("select", 0.1)
    
    # Create multiple threads
    threads = [
        threading.Thread(target=track_metrics)
        for _ in range(3)
    ]
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    # Verify metrics
    summary = collector.get_metrics_summary()
    assert summary["requests"]["total"] == 300
    assert summary["database"]["queries"] == 300

def test_global_metrics_collector():
    """Test global metrics collector instance."""
    assert metrics_collector is not None
    
    # Track a metric
    metrics_collector.track_request("GET", "/api/test", 200, 0.1)
    
    # Verify metric was recorded
    summary = metrics_collector.get_metrics_summary()
    assert summary["requests"]["total"] > 0

def test_error_handling(collector):
    """Test error handling in metrics collection."""
    # Test with invalid method
    collector.track_request(None, "/api/test", 200, 0.1)
    
    # Test with invalid duration
    collector.track_db_query("select", -1)
    
    # Should not raise exceptions and still return metrics
    metrics_data = collector.get_metrics().decode()
    assert metrics_data
    
    summary = collector.get_metrics_summary()
    assert isinstance(summary, dict)
