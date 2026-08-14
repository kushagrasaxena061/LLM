# api/hardening.py
"""Production API hardening: payload size validation, rate limiting, and safe error handling."""

import time
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

class APIHardeningMiddleware(BaseHTTPMiddleware):
    """Enforces strict security limits on request frequency and payload size."""
    def __init__(self, app, max_tokens_per_min: int = 1000):
        super().__init__(app)
        self.max_tokens_per_min = max_tokens_per_min
        self.request_counts = {}

    async def dispatch(self, request: Request, call_next):
        # 1. Payload Size Guard (Reject payloads > 1MB immediately)
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1_000_000:
            logger.warning("Payload size exceeded limit", size_bytes=content_length)
            return JSONResponse(status_code=413, content={"detail": "Payload too large. Maximum size is 1MB."})

        # 2. Basic IP-based Rate Limiting check
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        window_start = current_time - 60
        history = self.request_counts.get(client_ip, [])
        history = [t for t in history if t > window_start]
        
        if len(history) > 30:
            logger.warning("Rate limit exceeded", client_ip=client_ip)
            return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Please try again later."})
            
        history.append(current_time)
        self.request_counts[client_ip] = history

        # Proceed with request safely
        try:
            return await call_next(request)
        except Exception as e:
            logger.error("Unhandled API exception caught by hardening layer", error=str(e))
            return JSONResponse(status_code=500, content={"detail": "Internal server error. Safe exception wrapper engaged."})
