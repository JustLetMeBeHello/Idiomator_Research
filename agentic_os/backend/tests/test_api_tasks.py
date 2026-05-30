from fastapi.testclient import TestClient
from agentic_os.backend.main import app

client = TestClient(app)

def test_next_endpoint():
    r = client.get("/api/next")
    assert r.status_code == 200
    body = r.json()
    assert body is None or ("title" in body and "next_action" in body)

def test_tasks_endpoint():
    r = client.get("/api/tasks")
    assert r.status_code == 200
    assert isinstance(r.json(), list)

def test_done_unknown_id_returns_error():
    r = client.post("/api/tasks/exp-99/done")
    assert r.status_code == 200
    assert r.json().get("error") is True
