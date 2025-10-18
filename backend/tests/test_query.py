import json
import os

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import create_app
from app.services import rag


def test_query_returns_chunk_match(tmp_path) -> None:
    chunks = [
        {
            "chunk_id": "sample-1",
            "text": "Tesla reported revenue of USD 80B in 2023.",
            "document_name": "tesla-2023.md",
        }
    ]
    chunk_path = tmp_path / "chunks.json"
    chunk_path.write_text(json.dumps(chunks), encoding="utf-8")

    rag._chunk_cache = None  # reset cached index

    os.environ["CHUNK_INDEX_PATH"] = str(chunk_path)
    try:
        get_settings.cache_clear()
        client = TestClient(create_app())

        response = client.post("/v1/query", json={"prompt": "What was Tesla's 2023 revenue?"})
        assert response.status_code == 200
        body = response.json()

        assert body["session_id"]
        assert body["message"]["role"] == "assistant"
        assert "Top match" in body["message"]["content"]
        assert body["message"]["citations"]
    finally:
        get_settings.cache_clear()
        os.environ.pop("CHUNK_INDEX_PATH", None)
        rag._chunk_cache = None


def test_streaming_endpoint_returns_sse_response(tmp_path) -> None:
    chunks = [
        {
            "chunk_id": "sample-1",
            "text": "BMW grew automotive revenue by 10% in 2022.",
            "document_name": "bmw-2022.md",
        }
    ]
    chunk_path = tmp_path / "chunks.json"
    chunk_path.write_text(json.dumps(chunks), encoding="utf-8")

    rag._chunk_cache = None

    os.environ["CHUNK_INDEX_PATH"] = str(chunk_path)
    try:
        get_settings.cache_clear()
        client = TestClient(create_app())

        response = client.post("/v1/query/stream", json={"prompt": "How fast did BMW grow?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
    finally:
        get_settings.cache_clear()
        os.environ.pop("CHUNK_INDEX_PATH", None)
        rag._chunk_cache = None
