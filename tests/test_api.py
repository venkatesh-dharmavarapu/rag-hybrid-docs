from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_documents():
    response = client.get("/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert "total_count" in data


def test_ask_endpoint():
    payload = {
        "question": "What is the token expiration period?",
        "top_k_retrieval": 5,
        "top_n_rerank": 2,
    }
    response = client.post("/v1/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "confidence" in data
    assert "sources" in data