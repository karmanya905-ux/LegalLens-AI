from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_documents_empty_or_list():
    response = client.get("/documents")
    assert response.status_code == 200
    assert "documents" in response.json()
