import pytest
import json
import logging
import time
from pathlib import Path
import threading
from datetime import datetime

from app.monitoring.logger import logger, log_execution_time, log_context

@pytest.fixture
def log_files():
    """Create and clean up log files for testing."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Clear existing log files
    for file in log_dir.glob("*.log"):
        file.write_text("")
    
    yield {
        "app": log_dir / "app.log",
        "error": log_dir / "error.log",
        "performance": log_dir / "performance.log"
    }
    
    # Clean up after tests
    for file in log_dir.glob("*.log"):
        file.unlink()
    log_dir.rmdir()

def read_json_log(file_path: Path) -> list:
    """Read and parse JSON log lines."""
    logs = []
    with open(file_path) as f:
        for line in f:
            try:
                logs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return logs

def test_basic_logging(log_files):
    """Test basic logging functionality."""
    test_message = "Test log message"
    logger.info(test_message)
    
    logs = read_json_log(log_files["app"])
    assert len(logs) > 0
    
    log_entry = logs[-1]
    assert log_entry["message"] == test_message
    assert log_entry["level"] == "INFO"
    assert "timestamp" in log_entry
    assert "thread_id" in log_entry

def test_error_logging(log_files):
    """Test error logging with exception information."""
    try:
        raise ValueError("Test error")
    except Exception as e:
        logger.error("Error occurred", exc_info=e)
    
    error_logs = read_json_log(log_files["error"])
    assert len(error_logs) > 0
    
    error_entry = error_logs[-1]
    assert error_entry["level"] == "ERROR"
    assert "error" in error_entry
    assert error_entry["error"]["type"] == "ValueError"
    assert "traceback" in error_entry["error"]

def test_performance_logging(log_files):
    """Test performance metric logging."""
    operation = "test_operation"
    duration = 1.5
    logger.log_performance(operation, duration, True)
    
    perf_logs = read_json_log(log_files["performance"])
    assert len(perf_logs) > 0
    
    perf_entry = perf_logs[-1]
    assert perf_entry["operation"] == operation
    assert perf_entry["duration"] == duration
    assert perf_entry["success"] is True

def test_log_execution_time_decorator(log_files):
    """Test execution time logging decorator."""
    @log_execution_time("test_function")
    def test_function():
        time.sleep(0.1)
        return "result"
    
    result = test_function()
    assert result == "result"
    
    perf_logs = read_json_log(log_files["performance"])
    assert len(perf_logs) > 0
    
    perf_entry = perf_logs[-1]
    assert perf_entry["operation"] == "test_function"
    assert perf_entry["duration"] >= 0.1
    assert perf_entry["success"] is True

def test_log_context_manager(log_files):
    """Test logging context manager."""
    extra_context = {"param": "value"}
    
    try:
        with log_context("test_operation", **extra_context):
            time.sleep(0.1)
            raise ValueError("Test error")
    except ValueError:
        pass
    
    perf_logs = read_json_log(log_files["performance"])
    error_logs = read_json_log(log_files["error"])
    
    assert len(perf_logs) > 0
    assert len(error_logs) > 0
    
    perf_entry = perf_logs[-1]
    assert perf_entry["operation"] == "test_operation"
    assert perf_entry["success"] is False
    assert perf_entry["duration"] >= 0.1
    
    error_entry = error_logs[-1]
    assert "Test error" in str(error_entry["error"]["message"])

def test_thread_information(log_files):
    """Test thread information in logs."""
    def thread_function():
        logger.info("Thread log")
    
    thread = threading.Thread(target=thread_function)
    thread.start()
    thread.join()
    
    logs = read_json_log(log_files["app"])
    assert len(logs) > 0
    
    log_entry = logs[-1]
    assert "thread_id" in log_entry
    assert "thread_name" in log_entry

def test_log_levels(log_files):
    """Test different log levels."""
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")
    
    logs = read_json_log(log_files["app"])
    levels = [log["level"] for log in logs[-5:]]
    
    assert "DEBUG" in levels
    assert "INFO" in levels
    assert "WARNING" in levels
    assert "ERROR" in levels
    assert "CRITICAL" in levels

def test_extra_context(log_files):
    """Test logging with extra context."""
    extra = {
        "user_id": 123,
        "action": "test",
        "metadata": {"key": "value"}
    }
    
    logger.info("Test with context", extra=extra)
    
    logs = read_json_log(log_files["app"])
    log_entry = logs[-1]
    
    assert log_entry["user_id"] == extra["user_id"]
    assert log_entry["action"] == extra["action"]
    assert log_entry["metadata"] == extra["metadata"]

def test_performance_metrics(log_files):
    """Test detailed performance metrics."""
    @log_execution_time("complex_operation")
    def complex_operation():
        with log_context("sub_operation", step="1"):
            time.sleep(0.1)
        with log_context("sub_operation", step="2"):
            time.sleep(0.1)
        return True
    
    complex_operation()
    
    perf_logs = read_json_log(log_files["performance"])
    assert len(perf_logs) >= 3  # Main operation + 2 sub-operations
    
    # Verify main operation
    main_op = [log for log in perf_logs if log["operation"] == "complex_operation"]
    assert len(main_op) == 1
    assert main_op[0]["duration"] >= 0.2
    
    # Verify sub-operations
    sub_ops = [log for log in perf_logs if log["operation"] == "sub_operation"]
    assert len(sub_ops) == 2
    for op in sub_ops:
        assert op["duration"] >= 0.1
        assert "step" in op

def test_error_handling_in_context(log_files):
    """Test error handling in logging context."""
    class CustomError(Exception):
        pass
    
    with pytest.raises(CustomError):
        with log_context("error_operation"):
            raise CustomError("Custom test error")
    
    error_logs = read_json_log(log_files["error"])
    perf_logs = read_json_log(log_files["performance"])
    
    # Verify error log
    error_entry = error_logs[-1]
    assert error_entry["error"]["type"] == "CustomError"
    assert "Custom test error" in error_entry["error"]["message"]
    
    # Verify performance log
    perf_entry = perf_logs[-1]
    assert perf_entry["operation"] == "error_operation"
    assert perf_entry["success"] is False
