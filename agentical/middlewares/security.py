"""
Security Middleware Module for Agentical

Comprehensive security middleware stack implementing:
- Rate limiting with Redis backend
- Security headers (CSP, HSTS, etc.)
- Request validation and sanitization
- Bot protection and detection
- CORS enhancement

Following DevQ.ai security standards and OWASP best practices.
"""

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Union, Any
from urllib.parse import urlparse

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp


# Configuration Models
class RateLimitConfig(BaseModel):
    """Rate limiting configuration"""
    requests_per_minute: int = Field(default=100, ge=1, le=10000)
    burst: int = Field(default=10, ge=1, le=100)
    strategy: str = Field(default="sliding_window", regex="^(fixed|sliding_window)$")
    exclude_paths: List[str] = Field(default_factory=list)
    per_endpoint: bool = Field(default=True)
    redis_url: Optional[str] = Field(default=None)
    fallback_memory: bool = Field(default=True)


class SecurityHeadersConfig(BaseModel):
    """Security headers configuration"""
    csp_directives: Dict[str, List[str]] = Field(default_factory=dict)
    hsts_max_age: int = Field(default=31536000)  # 1 year
    frame_options: str = Field(default="DENY")
    content_type_options: str = Field(default="nosniff")
    referrer_policy: str = Field(default="strict-origin-when-cross-origin")
    permissions_policy: Dict[str, List[str]] = Field(default_factory=dict)
    disable_for_paths: List[str] = Field(default_factory=list)


class RequestValidationConfig(BaseModel):
    """Request validation configuration"""
    max_content_length: int = Field(default=10 * 1024 * 1024)  # 10MB
    max_query_params: int = Field(default=50)
    max_headers: int = Field(default=50)
    validate_query_params: bool = Field(default=True)
    validate_headers: bool = Field(default=True)
    validate_content_type: bool = Field(default=True)
    allowed_content_types: List[str] = Field(
        default=["application/json", "application/x-www-form-urlencoded", 
                 "multipart/form-data", "text/plain"]
    )
    exclude_paths: List[str] = Field(default_factory=list)


class BotDetectionConfig(BaseModel):
    """Bot detection and protection configuration"""
    enabled: bool = Field(default=True)
    challenge_suspicious: bool = Field(default=False)
    block_known_bots: bool = Field(default=False)
    exclude_paths: List[str] = Field(default_factory=list)
    trusted_user_agents: List[str] = Field(default_factory=list)
    suspicious_patterns: List[str] = Field(
        default=[
            r"bot", r"crawler", r"spider", r"scraper", r"wget", r"curl",
            r"python-requests", r"automated", r"scanner"
        ]
    )


# In-Memory Rate Limiting Store
class MemoryRateLimitStore:
    """In-memory rate limiting store as fallback"""
    
    def __init__(self):
        self._store: Dict[str, Dict[str, Union[int, float]]] = {}
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = time.time()
    
    async def get_requests(self, key: str, window: int) -> int:
        """Get request count for key in time window"""
        await self._cleanup_expired()
        now = time.time()
        
        if key not in self._store:
            self._store[key] = {"count": 0, "window_start": now}
            return 0
        
        data = self._store[key]
        if now - data["window_start"] > window:
            # Reset window
            self._store[key] = {"count": 0, "window_start": now}
            return 0
        
        return data["count"]
    
    async def increment(self, key: str, window: int) -> int:
        """Increment request count for key"""
        await self._cleanup_expired()
        now = time.time()
        
        if key not in self._store:
            self._store[key] = {"count": 1, "window_start": now}
            return 1
        
        data = self._store[key]
        if now - data["window_start"] > window:
            # Reset window
            self._store[key] = {"count": 1, "window_start": now}
            return 1
        
        self._store[key]["count"] += 1
        return self._store[key]["count"]
    
    async def _cleanup_expired(self):
        """Clean up expired entries"""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        expired_keys = [
            key for key, data in self._store.items()
            if now - data["window_start"] > 3600  # 1 hour
        ]
        
        for key in expired_keys:
            del self._store[key]
        
        self._last_cleanup = now


# Rate Limiting Middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware with Redis and memory fallback"""
    
    def __init__(self, app: ASGIApp, config: RateLimitConfig):
        super().__init__(app)
        self.config = config
        self.memory_store = MemoryRateLimitStore()
        self.redis_client = None
        
        # Try to initialize Redis if URL provided
        if config.redis_url:
            try:
                import redis.asyncio as redis
                self.redis_client = redis.from_url(config.redis_url, 
                                                   decode_responses=True)
            except ImportError:
                pass  # Fall back to memory store
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.config.exclude_paths):
            return await call_next(request)
        
        # Generate rate limit key
        client_ip = self._get_client_ip(request)
        if self.config.per_endpoint:
            key = f"rate_limit:{client_ip}:{request.url.path}:{request.method}"
        else:
            key = f"rate_limit:{client_ip}"
        
        # Check rate limit
        try:
            current_requests = await self._get_request_count(key)
            
            if current_requests >= self.config.requests_per_minute:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {self.config.requests_per_minute}/min",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Increment counter
            await self._increment_request_count(key)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, self.config.requests_per_minute - current_requests - 1)
            )
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
            
            return response
            
        except Exception:
            # On error, allow request to proceed
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    async def _get_request_count(self, key: str) -> int:
        """Get current request count"""
        if self.redis_client:
            try:
                count = await self.redis_client.get(key)
                return int(count) if count else 0
            except Exception:
                pass
        
        return await self.memory_store.get_requests(key, 60)
    
    async def _increment_request_count(self, key: str) -> int:
        """Increment request count"""
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                pipe.incr(key)
                pipe.expire(key, 60)
                results = await pipe.execute()
                return results[0]
            except Exception:
                pass
        
        return await self.memory_store.increment(key, 60)


# Security Headers Middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    def __init__(self, app: ASGIApp, csp_directives: Optional[Dict[str, List[str]]] = None,
                 disable_for_paths: Optional[List[str]] = None, **kwargs):
        super().__init__(app)
        self.csp_directives = csp_directives or {}
        self.disable_for_paths = disable_for_paths or []
        self.config = SecurityHeadersConfig(
            csp_directives=self.csp_directives,
            disable_for_paths=self.disable_for_paths,
            **kwargs
        )
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Skip security headers for excluded paths
        if any(request.url.path.startswith(path) for path in self.config.disable_for_paths):
            return response
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers"""
        
        # Content Security Policy
        if self.config.csp_directives:
            csp_value = "; ".join([
                f"{directive} {' '.join(sources)}"
                for directive, sources in self.config.csp_directives.items()
            ])
            response.headers["Content-Security-Policy"] = csp_value
        
        # HTTP Strict Transport Security
        response.headers["Strict-Transport-Security"] = (
            f"max-age={self.config.hsts_max_age}; includeSubDomains; preload"
        )
        
        # Frame Options
        response.headers["X-Frame-Options"] = self.config.frame_options
        
        # Content Type Options
        response.headers["X-Content-Type-Options"] = self.config.content_type_options
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Permissions Policy
        if self.config.permissions_policy:
            permissions_value = ", ".join([
                f"{feature}=({' '.join(allowlist)})"
                for feature, allowlist in self.config.permissions_policy.items()
            ])
            response.headers["Permissions-Policy"] = permissions_value
        
        # Additional security headers
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["X-DNS-Prefetch-Control"] = "off"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"


# Request Validation Middleware
class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Request validation and sanitization middleware"""
    
    def __init__(self, app: ASGIApp, config: RequestValidationConfig):
        super().__init__(app)
        self.config = config
        
        # Compile suspicious patterns
        self.sql_injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"(\bunion\b.*\bselect\b)|(\bselect\b.*\bunion\b)",
                r"\b(select|insert|update|delete|drop|create|alter)\b.*\b(from|into|table)\b",
                r"['\";].*(\bor\b|\band\b).*['\";]",
                r"(\bexec\b|\bexecute\b).*\(",
            ]
        ]
        
        self.xss_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*=",
                r"<iframe[^>]*>.*?</iframe>",
            ]
        ]
    
    async def dispatch(self, request: Request, call_next):
        # Skip validation for excluded paths
        if any(request.url.path.startswith(path) for path in self.config.exclude_paths):
            return await call_next(request)
        
        # Validate request
        validation_error = await self._validate_request(request)
        if validation_error:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"error": "Invalid request", "message": validation_error}
            )
        
        return await call_next(request)
    
    async def _validate_request(self, request: Request) -> Optional[str]:
        """Validate request for security issues"""
        
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length and int(content_length) > self.config.max_content_length:
            return f"Content too large. Max: {self.config.max_content_length} bytes"
        
        # Check query parameters
        if self.config.validate_query_params:
            if len(request.query_params) > self.config.max_query_params:
                return f"Too many query parameters. Max: {self.config.max_query_params}"
            
            for key, value in request.query_params.items():
                if self._contains_malicious_content(f"{key}={value}"):
                    return "Malicious content detected in query parameters"
        
        # Check headers
        if self.config.validate_headers:
            if len(request.headers) > self.config.max_headers:
                return f"Too many headers. Max: {self.config.max_headers}"
            
            for key, value in request.headers.items():
                if self._contains_malicious_content(f"{key}: {value}"):
                    return "Malicious content detected in headers"
        
        # Check content type
        if self.config.validate_content_type and request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type and content_type not in self.config.allowed_content_types:
                return f"Invalid content type: {content_type}"
        
        return None
    
    def _contains_malicious_content(self, content: str) -> bool:
        """Check if content contains malicious patterns"""
        
        # Check for SQL injection
        for pattern in self.sql_injection_patterns:
            if pattern.search(content):
                return True
        
        # Check for XSS
        for pattern in self.xss_patterns:
            if pattern.search(content):
                return True
        
        return False


# Bot Protection Middleware
class BotProtectionMiddleware(BaseHTTPMiddleware):
    """Bot detection and protection middleware"""
    
    def __init__(self, app: ASGIApp, config: BotDetectionConfig):
        super().__init__(app)
        self.config = config
        
        # Compile suspicious patterns
        self.suspicious_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in config.suspicious_patterns
        ]
    
    async def dispatch(self, request: Request, call_next):
        if not self.config.enabled:
            return await call_next(request)
        
        # Skip bot protection for excluded paths
        if any(request.url.path.startswith(path) for path in self.config.exclude_paths):
            return await call_next(request)
        
        # Analyze request for bot characteristics
        bot_score = self._analyze_bot_characteristics(request)
        
        if bot_score > 0.7:  # High confidence bot
            if self.config.block_known_bots:
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"error": "Access denied", "message": "Bot traffic detected"}
                )
            elif self.config.challenge_suspicious:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={
                        "error": "Challenge required",
                        "message": "Please complete verification",
                        "challenge_type": "captcha"
                    }
                )
        
        response = await call_next(request)
        
        # Add bot detection headers for monitoring
        response.headers["X-Bot-Score"] = str(round(bot_score, 2))
        
        return response
    
    def _analyze_bot_characteristics(self, request: Request) -> float:
        """Analyze request characteristics to determine bot probability"""
        score = 0.0
        
        user_agent = request.headers.get("User-Agent", "").lower()
        
        # Check if user agent is in trusted list
        if any(trusted in user_agent for trusted in self.config.trusted_user_agents):
            return 0.0
        
        # Check for suspicious patterns in user agent
        for pattern in self.suspicious_patterns:
            if pattern.search(user_agent):
                score += 0.3
        
        # Check for missing or suspicious headers
        common_headers = ["accept", "accept-language", "accept-encoding"]
        missing_headers = sum(1 for header in common_headers if header not in request.headers)
        score += missing_headers * 0.1
        
        # Check for suspicious request patterns
        if not user_agent:
            score += 0.4
        
        if len(user_agent) < 10:
            score += 0.2
        
        # Check for rapid requests (would need Redis/memory store for full implementation)
        # This is a simplified check
        if request.headers.get("Connection", "").lower() == "close":
            score += 0.1
        
        return min(score, 1.0)