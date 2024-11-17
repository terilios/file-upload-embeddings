import logging
import json
from datetime import datetime
from typing import Any, Dict, Optional
import sys
import traceback
from pathlib import Path
from pythonjsonlogger import jsonlogger
from functools import wraps
import time
import threading
from contextlib import contextmanager

from config.settings import settings

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for structured logging."""
    
    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict) -> None:
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add environment
        log_record['environment'] = settings.ENVIRONMENT
        
        # Add thread information
        log_record['thread_id'] = threading.get_ident()
        log_record['thread_name'] = threading.current_thread().name
        
        # Add caller information
        log_record['filename'] = record.filename
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add trace ID if available
        trace_id = getattr(threading.current_thread(), 'trace_id', None)
        if trace_id:
            log_record['trace_id'] = trace_id

class Logger:
    """Centralized logging system with structured logging."""
    
    def __init__(self):
        """Initialize logger with multiple handlers."""
        self.logger = logging.getLogger('app')
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        json_formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'app.log',
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'error.log',
            maxBytes=10485760,
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            log_dir / 'performance.log',
            maxBytes=10485760,
            backupCount=5
        )
        perf_handler.setFormatter(json_formatter)
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.addHandler(perf_handler)
    
    def _log(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: Optional[Exception] = None
    ) -> None:
        """
        Log a message with extra context.
        
        Args:
            level: Log level
            message: Log message
            extra: Additional context
            exc_info: Exception information
        """
        extra = extra or {}
        
        # Add exception details if available
        if exc_info:
            extra['error'] = {
                'type': type(exc_info).__name__,
                'message': str(exc_info),
                'traceback': traceback.format_exc()
            }
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, extra)
    
    def error(self, message: str, extra: Optional[Dict] = None, exc_info: Optional[Exception] = None) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, extra, exc_info)
    
    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, extra)
    
    def debug(self, message: str, extra: Optional[Dict] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict] = None, exc_info: Optional[Exception] = None) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, extra, exc_info)
    
    def log_performance(
        self,
        operation: str,
        duration: float,
        success: bool,
        extra: Optional[Dict] = None
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Operation being measured
            duration: Duration in seconds
            success: Whether operation succeeded
            extra: Additional context
        """
        extra = extra or {}
        extra.update({
            'operation': operation,
            'duration': duration,
            'success': success,
            'timestamp': datetime.utcnow().isoformat()
        })
        self.perf_logger.info('Performance metric', extra=extra)

def log_execution_time(operation: str):
    """
    Decorator to log execution time of functions.
    
    Args:
        operation: Name of the operation being timed
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                success = False
                raise e
            finally:
                duration = time.time() - start_time
                logger.log_performance(
                    operation,
                    duration,
                    success
                )
        return wrapper
    return decorator

@contextmanager
def log_context(operation: str, **kwargs):
    """
    Context manager for logging operations with timing.
    
    Args:
        operation: Name of the operation
        **kwargs: Additional context to log
    """
    start_time = time.time()
    success = True
    try:
        yield
    except Exception as e:
        success = False
        logger.error(
            f"Error in {operation}",
            extra=kwargs,
            exc_info=e
        )
        raise
    finally:
        duration = time.time() - start_time
        logger.log_performance(
            operation,
            duration,
            success,
            extra=kwargs
        )

# Global logger instance
logger = Logger()
