# observability/middleware.py
"""Asynchronous telemetry and audit logging middleware for FastAPI."""

import time
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from utils.logger import get_logger

logger = get_logger(__name__)

class TelemetryMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """
        Intercepts incoming HTTP requests to inject tracking IDs, 
        measure latency, and log production audit trails.
        """
        # 1. Generate a unique tracing ID for this specific request
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # 2. Attach the ID to the request state so inner functions can use it if needed
        request.state.request_id = request_id
        
        try:
            # 3. Process the actual API request
            response = await call_next(request)
            
            # 4. Calculate end-to-end latency
            process_time = time.time() - start_time
            
            # 5. Inject observability metrics into the HTTP Response Headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Sec"] = str(round(process_time, 4))
            
            # 6. Write to the Audit Log
            logger.info(
                "API Request Completed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                latency_sec=round(process_time, 4)
            )
            
            return response
            
        except Exception as e:
            # Audit log any catastrophic server crashes with the trace ID
            process_time = time.time() - start_time
            logger.error(
                "API Request Failed",
                request_id=request_id,
                method=request.method,
                path=request.url.path,
                error=str(e),
                latency_sec=round(process_time, 4)
            )
            raise e
