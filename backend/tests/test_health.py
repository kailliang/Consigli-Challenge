from fastapi.testclient import TestClient

from app.main import create_app


def test_healthcheck_returns_ok() -> None:
    client = TestClient(create_app())
    response = client.get("/v1/health/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
