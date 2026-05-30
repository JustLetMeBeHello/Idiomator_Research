from fastapi.testclient import TestClient
from agentic_os.backend.main import app

client = TestClient(app)

def test_folders_endpoint():
    r = client.get("/api/folders")
    assert r.status_code == 200
    assert any(f["id"] == "research" for f in r.json())

def test_systems_endpoint():
    r = client.get("/api/research/systems")
    assert r.status_code == 200
    rows = r.json()
    assert any(row["label"] == "D" for row in rows)

def test_ablation_endpoint():
    r = client.get("/api/research/ablation")
    assert r.status_code == 200
    assert any(row["combo"] == "en_es_hi_te" for row in r.json())

def test_experiments_endpoint():
    r = client.get("/api/research/experiments")
    assert r.status_code == 200
    assert any(e["id"] == "02" for e in r.json())

def test_blockers_endpoint():
    r = client.get("/api/research/blockers")
    assert r.status_code == 200
    assert any(b["hard_blocker"] for b in r.json())

def test_activity_endpoint():
    r = client.get("/api/activity")
    assert r.status_code == 200
    assert isinstance(r.json(), list)
