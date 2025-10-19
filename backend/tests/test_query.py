import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.core.config import get_settings


def _print_header(title: str) -> None:
    line = "=" * len(title)
    print(f"{line}\n{title}\n{line}")


def _debug_response(message: dict, session_id: str | None) -> None:
    payload = {"session_id": session_id, "message": message}
    print(json.dumps(payload, indent=2))

    citations = message.get("citations", [])
    if citations:
        print("[debug] citations:")
        for idx, citation in enumerate(citations):
            print(f"  [{idx}] id={citation.get('id')} source={citation.get('source')}")
            snippet = citation.get("snippet", "")
            print(f"      snippet={snippet[:200].replace(os.linesep, ' ')}")


def _collect_stream(client: TestClient, prompt: str) -> tuple[list[str], dict]:
    tokens: list[str] = []
    final_event: dict | None = None

    with client.websocket_connect("/v1/query/ws") as websocket:
        websocket.send_json({"prompt": prompt})

        while True:
            event = websocket.receive_json()
            if event.get("event") == "token":
                token = event.get("data", {}).get("content", "")
                if token:
                    tokens.append(token)
            elif event.get("event") == "done":
                final_event = event.get("data") or {}
                break
            elif event.get("event") == "error":
                detail = event.get("data", {}).get("message", "Streaming error")
                raise RuntimeError(detail)

    if final_event is None:
        raise RuntimeError("Streaming ended without completion event.")

    return tokens, final_event


def test_real_query_retrieval(capsys) -> None:
    """Run a realistic query against the live RAG stack."""

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY required for real retrieval test")

    app = create_app()
    client = TestClient(app)

    prompt = "What were Ford's 2023 wholesale volumes?"
    _, final_event = _collect_stream(client, prompt)

    message = final_event.get("message", {})
    assert message.get("role") == "assistant"

    debug_text = capsys.readouterr().out
    if debug_text:
        print("[debug] captured backend logs:")
        print(debug_text)

    _debug_response(message, final_event.get("session_id"))


def run_custom_query(prompt: str) -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY not configured; cannot run real query.")
        sys.exit(1)

    app = create_app()
    client = TestClient(app)

    _print_header(f"Prompt: {prompt}")
    tokens, final_event = _collect_stream(client, prompt)
    print(f"[debug] received {len(tokens)} token events")

    _debug_response(final_event.get("message", {}), final_event.get("session_id"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_query.py \"Your prompt here\"")
        sys.exit(1)
    run_custom_query(sys.argv[1])
