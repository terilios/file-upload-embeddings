import pytest
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
import json
import time
from datetime import datetime

from app.monitoring.middleware import (
    MonitoringMiddleware,
    ResponseHeaderMiddleware,
    ErrorHandlingMiddleware
)
from app.monitoring.logger import logger
from app.monitoring.metrics import metrics_collector

@pytest.fixture
def app():
    """Create test FastAPI application with monitoring middleware."""
    app = FastAPI()
    
    # Add monitoring middleware
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(ResponseHeaderMiddleware)
    app.add_middleware(MonitoringMiddleware)
    
    # Test endpoints
    @app.get("/test")
    async def test_endpoint():
        return {"message": "success"}
    
    @app.post("/test")
    async def test_post_endpoint(data: dict):
        return {"received": data}
    
    @app.get("/error")
    async def error_endpoint():
        raise ValueError("Test error")
    
    @app.get("/slow")
    async def slow_endpoint():
        time.sleep(0.1)
        return {"message": "slow response"}
    
    return app

@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)

def test_successful_request(client, caplog):
    """Test monitoring of successful requests."""
    response = client.get("/test")
    
    assert response.status_code == 200
    assert "X-Request-ID" in response.headers
    assert "X-Response-Time" in response.headers
    
    # Verify logging
    assert any(
        "Request received" in record.message
        for record in caplog.records
    )
    assert any(
        "Request completed" in record.message
        for record in caplog.records
    )
    
    # Verify metrics
    metrics = metrics_collector.get_metrics_summary()
    assert metrics["requests"]["total"] > 0

def test_request_with_body(client, caplog):
    """Test monitoring of requests with body."""
    test_data = {"key": "value"}
    response = client.post("/test", json=test_data)
    
    assert response.status_code == 200
    
    # Verify request body was logged
    log_records = [r for r in caplog.records if "Request received" in r.message]
    assert len(log_records) > 0
    assert "body" in log_records[0].extra
    assert json.loads(log_records[0].extra["body"]) == test_data

def test_error_handling(client, caplog):
    """Test monitoring of requests with errors."""
    response = client.get("/error")
    
    assert response.status_code == 500
    assert "error" in response.json()
    assert "trace_id" in response.json()
    
    # Verify error logging
    assert any(
        "Request failed" in record.message
        for record in caplog.records
    )
    
    # Verify error metrics
    metrics = metrics_collector.get_metrics_summary()
    assert metrics["requests"]["total"] > 0

def test_response_time_tracking(client):
    """Test response time tracking."""
    response = client.get("/slow")
    
    assert response.status_code == 200
    assert "X-Response-Time" in response.headers
    
    response_time = float(response.headers["X-Response-Time"].rstrip("s"))
    assert response_time >= 0.1

def test_trace_id_propagation(client):
    """Test trace ID propagation through requests."""
    response = client.get("/test")
    
    assert "X-Request-ID" in response.headers
    trace_id = response.headers["X-Request-ID"]
    assert trace_id != "unknown"
    
    # Make another request
    response2 = client.get("/test")
    trace_id2 = response2.headers["X-Request-ID"]
    
    # Trace IDs should be different
    assert trace_id != trace_id2

def test_concurrent_requests(client):
    """Test handling of concurrent requests."""
    import threading
    
    def make_request():
        response = client.get("/test")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
    
    # Create multiple threads
    threads = [
        threading.Thread(target=make_request)
        for _ in range(3)
    ]
    
    # Start threads
    for thread in threads:
        thread.start()
    
    # Wait for threads to complete
    for thread in threads:
        thread.join()
    
    # Verify metrics
    metrics = metrics_collector.get_metrics_summary()
    assert metrics["requests"]["total"] >= 3

def test_large_request_handling(client, caplog):
    """Test handling of large requests."""
    large_data = {"data": "x" * 1000000}  # 1MB of data
    response = client.post("/test", json=large_data)
    
    assert response.status_code == 200
    
    # Verify request was logged without full body
    log_records = [r for r in caplog.records if "Request received" in r.message]
    assert len(log_records) > 0
    assert "body" in log_records[0].extra

def test_invalid_json_handling(client):
    """Test handling of invalid JSON requests."""
    response = client.post(
        "/test",
        headers={"Content-Type": "application/json"},
        data="invalid json"
    )
    
    assert response.status_code == 422  # FastAPI validation error
    assert "X-Request-ID" in response.headers

def test_metrics_collection(client):
    """Test comprehensive metrics collection."""
    # Make various types of requests
    client.get("/test")
    client.post("/test", json={"test": "data"})
    client.get("/error")
    client.get("/slow")
    
    # Get metrics summary
    metrics = metrics_collector.get_metrics_summary()
    
    # Verify request metrics
    assert metrics["requests"]["total"] >= 4
    assert metrics["requests"]["latency_avg"] > 0
    
    # Get raw metrics
    raw_metrics = metrics_collector.get_metrics().decode()
    
    # Verify different status codes were tracked
    assert "status=\"200\"" in raw_metrics
    assert "status=\"500\"" in raw_metrics

def test_error_response_format(client):
    """Test format of error responses."""
    response = client.get("/error")
    
    assert response.status_code == 500
    data = response.json()
    
    assert "error" in data
    assert "type" in data
    assert "trace_id" in data
    assert data["type"] == "ValueError"
    assert "Test error" in data["error"]

def test_middleware_order(app):
    """Test middleware execution order."""
    middleware_order = [
        type(m.cls) for m in app.user_middleware
    ]
    
    # ErrorHandling should be last (first to handle)
    assert middleware_order[0] == ErrorHandlingMiddleware
    # Monitoring should be first (last to handle)
    assert middleware_order[-1] == MonitoringMiddleware
