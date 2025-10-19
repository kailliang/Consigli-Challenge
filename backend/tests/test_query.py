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


def _debug_response(response_json: dict) -> None:
    message = response_json.get("message", {})
    print(json.dumps(response_json, indent=2))

    citations = message.get("citations", [])
    if citations:
        print("[debug] citations:")
        for idx, citation in enumerate(citations):
            print(f"  [{idx}] id={citation.get('id')} source={citation.get('source')}")
            snippet = citation.get("snippet", "")
            print(f"      snippet={snippet[:200].replace(os.linesep, ' ')}")


def test_real_query_retrieval(capsys) -> None:
    """Run a realistic query against the live RAG stack."""

    settings = get_settings()
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY required for real retrieval test")

    app = create_app()
    client = TestClient(app)

    prompt = "What were Ford's 2023 wholesale volumes?"
    response = client.post("/v1/query", json={"prompt": prompt})

    assert response.status_code == 200
    body = response.json()
    assert body["message"]["role"] == "assistant"

    debug_text = capsys.readouterr().out
    if debug_text:
        print("[debug] captured backend logs:")
        print(debug_text)

    _debug_response(body)


def run_custom_query(prompt: str) -> None:
    settings = get_settings()
    if not settings.openai_api_key:
        print("OPENAI_API_KEY not configured; cannot run real query.")
        sys.exit(1)

    app = create_app()
    client = TestClient(app)

    _print_header(f"Prompt: {prompt}")
    response = client.post("/v1/query", json={"prompt": prompt})
    print(f"[debug] status={response.status_code}")

    _debug_response(response.json())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tests/test_query.py \"Your prompt here\"")
        sys.exit(1)
    run_custom_query(sys.argv[1])
