"""
Logging Middleware Module for Agentical

This module provides middleware classes for integrating
structured logging into FastAPI applications.

Features:
- Request ID generation and propagation
- Request/response logging
- Performance timing
- Error tracking
"""

import time
import uuid
from typing import Callable, Dict, Any

import logfire
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from agentical.core.logging import mask_sensitive_data


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging request and response information with
    request ID generation and propagation.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, adding request ID and logging request/response details.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
        """
        # Generate or propagate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Extract relevant request data for logging
        request_data = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("User-Agent", "unknown")
        }
        
        # Log request with context
        with logfire.context(**request_data):
            # Create a span for the entire request lifecycle
            with logfire.span("HTTP Request"):
                start_time = time.time()
                
                # Log incoming request
                logfire.info(
                    "Request received",
                    headers=mask_sensitive_data(dict(request.headers))
                )
                
                try:
                    # Process the request
                    response = await call_next(request)
                    
                    # Calculate processing time
                    process_time = time.time() - start_time
                    process_time_ms = round(process_time * 1000, 2)
                    
                    # Add headers to response
                    response.headers["X-Request-ID"] = request_id
                    response.headers["X-Process-Time-Ms"] = str(process_time_ms)
                    
                    # Log successful response
                    logfire.info(
                        "Request completed",
                        status_code=response.status_code,
                        processing_time_ms=process_time_ms
                    )
                    
                    return response
                    
                except Exception as e:
                    # Calculate processing time for failed request
                    process_time = time.time() - start_time
                    process_time_ms = round(process_time * 1000, 2)
                    
                    # Log exception with details
                    logfire.error(
                        "Request failed with exception",
                        error=str(e),
                        error_type=type(e).__name__,
                        processing_time_ms=process_time_ms
                    )
                    raise


class HealthCheckLoggingFilter(BaseHTTPMiddleware):
    """
    Middleware to filter out health check requests from normal logging
    to prevent log pollution from frequent health checks.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, filtering out health check requests from normal logging.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
        """
        # Check if this is a health check request
        is_health_check = request.url.path in ["/health", "/ready", "/live"]
        
        if is_health_check and request.method == "GET":
            # For health checks, use a minimal span with debug level
            with logfire.context(path=request.url.path, method=request.method):
                with logfire.span("Health Check Request"):
                    # Process health check with minimal logging
                    start_time = time.time()
                    response = await call_next(request)
                    process_time = time.time() - start_time
                    
                    # Add processing time header
                    response.headers["X-Process-Time-Ms"] = str(round(process_time * 1000, 2))
                    
                    # Log at debug level only
                    logfire.debug(
                        "Health check processed",
                        status_code=response.status_code,
                        processing_time_ms=round(process_time * 1000, 2)
                    )
                    
                    return response
        else:
            # For non-health check requests, proceed normally
            return await call_next(request)


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware specifically for handling and logging errors in a structured way.
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request, providing enhanced error logging.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response from the next middleware or route handler
        """
        try:
            # Process the request normally
            return await call_next(request)
            
        except Exception as e:
            # Create detailed error log with request context
            error_data = {
                "error": str(e),
                "error_type": type(e).__name__,
                "method": request.method,
                "path": request.url.path,
                "query_params": str(request.query_params),
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("User-Agent", "unknown"),
                "request_id": request.headers.get("X-Request-ID", "unknown")
            }
            
            # Log the error with full context
            logfire.error("Unhandled exception in request processing", **error_data)
            
            # Re-raise the exception for the exception handlers to process
            raise