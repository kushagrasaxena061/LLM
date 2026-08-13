# tests/test_observability.py
"""Unit tests to verify telemetry and observability middleware injection."""

from fastapi.testclient import TestClient
from api.server import app

def test_telemetry_headers_injection():
    """Verifies that every API response includes Request IDs and latency metrics."""
    with TestClient(app) as client:
        response = client.get("/health")
        
        # Verify standard behavior
        assert response.status_code == 200, "Health check failed!"
        
        # Verify Telemetry Headers
        headers = response.headers
        assert "x-request-id" in headers, "Missing X-Request-ID telemetry header!"
        assert "x-process-time-sec" in headers, "Missing X-Process-Time-Sec telemetry header!"
        
        request_id = headers["x-request-id"]
        latency = float(headers["x-process-time-sec"])
        
        assert len(request_id) > 10, "Invalid Request ID format!"
        assert latency >= 0.0, "Negative latency recorded!"
        
        print(f"\n✅ Production Observability Test Passed!")
        print(f"   - Tracked Request ID: {request_id}")
        print(f"   - Recorded End-to-End Latency: {latency * 1000:.2f} ms")
