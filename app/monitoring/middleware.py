from typing import Callable, Dict, Any
import time
from datetime import datetime
import json
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import uuid
import threading

from .logger import logger
from .metrics import metrics_collector

class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for request monitoring, logging, and metrics collection."""
    
    def __init__(self, app: ASGIApp):
        """Initialize monitoring middleware."""
        super().__init__(app)
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request with monitoring.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
        
        Returns:
            Response from the endpoint
        """
        # Generate trace ID for request
        trace_id = str(uuid.uuid4())
        threading.current_thread().trace_id = trace_id
        
        # Start timing
        start_time = time.time()
        
        # Initialize response
        response = None
        error = None
        
        try:
            # Log request
            await self._log_request(request, trace_id)
            
            # Process request
            response = await call_next(request)
            
            # Track successful request
            duration = time.time() - start_time
            self._track_request(request, response, duration)
            
            return response
            
        except Exception as e:
            # Track failed request
            duration = time.time() - start_time
            error = e
            self._track_error(request, e, duration)
            raise
            
        finally:
            # Log response
            await self._log_response(
                request,
                response,
                error,
                trace_id,
                time.time() - start_time
            )
            
            # Clear trace ID
            threading.current_thread().trace_id = None
    
    async def _log_request(
        self,
        request: Request,
        trace_id: str
    ) -> None:
        """
        Log incoming request details.
        
        Args:
            request: FastAPI request
            trace_id: Request trace ID
        """
        # Get request body if available
        body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.json()
            except:
                body = await request.body()
        
        logger.info(
            f"Request received: {request.method} {request.url.path}",
            extra={
                "trace_id": trace_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": dict(request.headers),
                "body": body,
                "client_host": request.client.host if request.client else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Response,
        error: Exception,
        trace_id: str,
        duration: float
    ) -> None:
        """
        Log response details.
        
        Args:
            request: FastAPI request
            response: Response or None if error
            error: Exception if any
            trace_id: Request trace ID
            duration: Request duration in seconds
        """
        status_code = 500 if error else response.status_code
        
        log_data = {
            "trace_id": trace_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": status_code,
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if error:
            log_data["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra=log_data,
                exc_info=error
            )
        else:
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra=log_data
            )
    
    def _track_request(
        self,
        request: Request,
        response: Response,
        duration: float
    ) -> None:
        """
        Track request metrics.
        
        Args:
            request: FastAPI request
            response: Response
            duration: Request duration in seconds
        """
        metrics_collector.track_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration
        )
    
    def _track_error(
        self,
        request: Request,
        error: Exception,
        duration: float
    ) -> None:
        """
        Track error metrics.
        
        Args:
            request: FastAPI request
            error: Exception that occurred
            duration: Request duration in seconds
        """
        metrics_collector.track_request(
            method=request.method,
            endpoint=request.url.path,
            status=500,
            duration=duration
        )

class ResponseHeaderMiddleware(BaseHTTPMiddleware):
    """Middleware for adding monitoring headers to responses."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Add monitoring headers to response.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
        
        Returns:
            Response with added headers
        """
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Add monitoring headers
        response.headers["X-Request-ID"] = getattr(
            threading.current_thread(),
            'trace_id',
            'unknown'
        )
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for consistent error handling and logging."""
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Handle errors consistently.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/endpoint
        
        Returns:
            Response with error handling
        """
        try:
            return await call_next(request)
            
        except Exception as e:
            # Log error
            logger.error(
                "Unhandled error in request",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "error_type": type(e).__name__,
                    "trace_id": getattr(
                        threading.current_thread(),
                        'trace_id',
                        'unknown'
                    )
                },
                exc_info=e
            )
            
            # Return error response
            return Response(
                content=json.dumps({
                    "error": str(e),
                    "type": type(e).__name__,
                    "trace_id": getattr(
                        threading.current_thread(),
                        'trace_id',
                        'unknown'
                    )
                }),
                status_code=500,
                media_type="application/json"
            )
