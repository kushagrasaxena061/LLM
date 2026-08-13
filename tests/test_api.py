# tests/test_api.py
"""Unit tests for FastAPI endpoints using TestClient."""

from fastapi.testclient import TestClient
from api.server import app

def test_health_endpoint():
    """Verifies that the health check endpoint returns 200 OK."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200, "Health check failed!"
        data = response.json()
        assert data["status"] == "healthy", "Server reported unhealthy status!"
        print(f"\n✅ API Health Check Passed! Status: {data['status']}, Device: {data['device']}")

def test_generate_endpoint():
    """Verifies that the /generate POST endpoint returns valid text."""
    with TestClient(app) as client:
        payload = {
            "prompt": "FastAPI",
            "max_new_tokens": 10,
            "temperature": 0.7
        }
        response = client.post("/generate", json=payload)
        assert response.status_code == 200, f"Generation endpoint failed with: {response.text}"
        data = response.json()
        assert "generated_text" in data, "Missing generated_text in response JSON!"
        print(f"✅ API Generation Endpoint Passed! Output: '{data['generated_text']}'")

def test_rag_endpoint():
    """Verifies that the /rag/query POST endpoint returns augmented generation."""
    with TestClient(app) as client:
        payload = {
            "query": "What is FastAPI?",
            "max_new_tokens": 10
        }
        response = client.post("/rag/query", json=payload)
        assert response.status_code == 200, f"RAG endpoint failed with: {response.text}"
        data = response.json()
        assert "rag_response" in data, "Missing rag_response in response JSON!"
        print(f"✅ API RAG Endpoint Passed! Response:\n{data['rag_response']}")
