from contextlib import contextmanager

from fastapi.testclient import TestClient

from app.api import deps
from app.core.config import get_settings
from app.main import create_app
from app.models.chat import ChatMessage
from app.services import rag


@contextmanager
def override_dependencies(app):
    original = dict(app.dependency_overrides)
    try:
        app.dependency_overrides[deps.get_rag_context] = lambda: rag.RAGContext(
            retriever=object(),
            chain=object(),
            llm=object(),
            prompt=object(),
            settings=get_settings(),
        )
        yield
    finally:
        app.dependency_overrides = original


def test_query_returns_response(monkeypatch) -> None:
    async def fake_generate_response(*_, **__) -> ChatMessage:
        return ChatMessage(
            id="test-id",
            role="assistant",
            content="Tesla reported revenue of USD 80B in 2023.",
            citations=[],
        )

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    monkeypatch.setattr(rag, "generate_response", fake_generate_response)

    app = create_app()

    with override_dependencies(app):
        client = TestClient(app)
        response = client.post("/v1/query", json={"prompt": "What was Tesla's 2023 revenue?"})
        assert response.status_code == 200
        body = response.json()
        assert body["session_id"]
        assert body["message"]["role"] == "assistant"
        assert "Tesla reported revenue" in body["message"]["content"]
        assert body["message"]["citations"] == []
    get_settings.cache_clear()


def test_streaming_endpoint_returns_sse_response(monkeypatch) -> None:
    async def fake_stream_response(*_, **__):
        yield {"event": "token", "data": {"content": "chunk"}}
        yield {
            "event": "done",
            "data": {
                "session_id": "session",
                "message": ChatMessage(id="stream", role="assistant", content="final", citations=[]).model_dump(),
            },
        }

    async def fake_generate_response(*_, **__) -> ChatMessage:
        return ChatMessage(id="fallback", role="assistant", content="chunk", citations=[])

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    get_settings.cache_clear()
    monkeypatch.setattr(rag, "stream_response", fake_stream_response)
    monkeypatch.setattr(rag, "generate_response", fake_generate_response)

    app = create_app()

    with override_dependencies(app):
        client = TestClient(app)
        response = client.post("/v1/query/stream", json={"prompt": "How fast did BMW grow?"})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = list(response.iter_lines())
        assert any("chunk" in line for line in body)
    get_settings.cache_clear()
