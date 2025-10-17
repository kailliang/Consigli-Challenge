from app.main import create_app
from fastapi.testclient import TestClient


def test_query_returns_placeholder_response() -> None:
    client = TestClient(create_app())
    response = client.post("/v1/query", json={"prompt": "What is Tesla's revenue?"})
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["message"]["role"] == "assistant"
    assert "RAG pipeline is not yet connected" in body["message"]["content"]
