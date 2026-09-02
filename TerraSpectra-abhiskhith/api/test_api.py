"""
VERIFY FASTAPI REST API (ML INFERENCE)
--------------------------------------
Tests the health check and tile-inference endpoints programmatically using FastAPI TestClient.
"""

from pathlib import Path
import sys
from fastapi.testclient import TestClient

# Paths setup
V2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(V2_ROOT))

from api.main import app

client = TestClient(app)


def test_api_endpoints():
    print("=" * 65)
    print("⚡ WEEK 4: FASTAPI REST API VERIFICATION")
    print("=" * 65)
    
    # 1. Health check
    print("\n1. Testing GET /health ...")
    res_health = client.get("/health")
    print(f"   Status Code: {res_health.status_code}")
    print(f"   Response:    {res_health.json()}")
    assert res_health.status_code == 200
    
    # 2. Tile inference (3D-CNN)
    print("\n2. Testing POST /tile-inference (Model: 3D-CNN) ...")
    res_tile = client.post("/tile-inference", json={"height": 128, "width": 128, "model_type": "3dcnn"})
    print(f"   Status Code: {res_tile.status_code}")
    print(f"   Response:    {res_tile.json()}")
    assert res_tile.status_code == 200
    
    # 3. Tile inference (ViT)
    print("\n3. Testing POST /tile-inference (Model: ViT) ...")
    res_vit = client.post("/tile-inference", json={"height": 128, "width": 128, "model_type": "vit"})
    print(f"   Status Code: {res_vit.status_code}")
    print(f"   Response:    {res_vit.json()}")
    assert res_vit.status_code == 200
    
    print("\n" + "=" * 65)
    print("🎉 FASTAPI ML API VERIFIED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    test_api_endpoints()
