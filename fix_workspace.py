import os
import sys
from pathlib import Path

root = Path(".")

# 1. Create conftest.py to permanently fix Pytest pathing
with open(root / "conftest.py", "w") as f:
    f.write("import sys\nimport os\nsys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))\n")

# 2. Ensure all packages have __init__.py markers
packages = ["api", "evaluation", "security", "rag", "prompt_engineering", "tests", "tests/unit", "tests/performance", "tests/security", "tests/evaluation"]
for p in packages:
    os.makedirs(root / p, exist_ok=True)
    (root / p / "__init__.py").touch(exist_ok=True)

# 3. Rewrite test_api_hardening.py with the execution block
test_code = """import sys
from pathlib import Path

root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi.testclient import TestClient
from api.server import app

def test_payload_size_limit():
    with TestClient(app) as client:
        huge_prompt = "A" * 1_500_000
        response = client.post("/generate", json={"prompt": huge_prompt, "max_new_tokens": 5})
        assert response.status_code == 413, "API failed to reject oversized payload!"
        print("\\n✅ API Hardening Test Passed: Oversized payload successfully blocked (413).")

def test_hardened_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["security"] == "hardened"
        print("✅ API Hardening Test Passed: Hardened health check verified.")

if __name__ == "__main__":
    test_payload_size_limit()
    test_hardened_health_endpoint()
"""
with open(root / "tests/security/test_api_hardening.py", "w") as f:
    f.write(test_code)

print("\n✅ Workspace permanently fixed for Pytest and Native execution!")
