# tests/security/test_api_hardening.py
"""Unit tests verifying API hardening rules (payload limits and input validation)."""

import sys
from pathlib import Path

# Force inject the absolute path of the project root
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi.testclient import TestClient
from api.server import app

def test_payload_size_limit():
    """Verifies that requests exceeding 1MB are rejected with status 413."""
    with TestClient(app) as client:
        huge_prompt = "A" * 1_500_000
        response = client.post("/generate", json={"prompt": huge_prompt, "max_new_tokens": 5})
        assert response.status_code == 413, f"API failed to reject oversized payload! Got {response.status_code}"
        print("\n✅ API Hardening Test Passed: Oversized payload successfully blocked (413).")

def test_hardened_health_endpoint():
    """Verifies the hardened health check returns status 200."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["security"] == "hardened"
        print("✅ API Hardening Test Passed: Hardened health check verified.")

if __name__ == "__main__":
    test_payload_size_limit()
    test_hardened_health_endpoint()
