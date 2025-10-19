"""Unit tests for the upgraded RAG workflow."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.core.config import AppSettings
from app.services import memory
from app.services.rag import RAGContext, _get_or_create_memory_store, build_rag_context, generate_response


class StubLLM:
    """Minimal async-compatible LLM stub."""

    def __init__(self, responses: list[AIMessage], *, default: AIMessage | None = None) -> None:
        self._responses = responses
        self._default = default
        self.calls = 0
        self.bound_tools: list[dict] | None = None
        self.last_messages = None

    def bind_tools(self, tools: list[dict]) -> "StubLLM":
        self.bound_tools = tools
        return self

    async def ainvoke(self, messages) -> AIMessage:  # noqa: D401 - match langchain signature
        self.last_messages = messages
        if self.calls < len(self._responses):
            response = self._responses[self.calls]
        elif self._default is not None:
            response = self._default
        else:  # pragma: no cover - defensive guard
            raise AssertionError("StubLLM received more calls than configured")

        self.calls += 1
        return response


class StubRetriever:
    """Async retriever that records the queries it receives."""

    def __init__(self, mapping: dict[str, list[Document]]) -> None:
        self._mapping = mapping
        self.calls: list[str] = []

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        self.calls.append(query)
        return list(self._mapping.get(query, []))


class StubEmbeddings:
    async def aembed_query(self, text: str) -> list[float]:
        return [1.0, 1.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for index, text in enumerate(texts):
            weight = 1.0 if "premium" in text.lower() else max(0.1, 0.8 - index * 0.1)
            vectors.append([1.0, weight])
        return vectors


class StubChain:
    async def ainvoke(self, inputs) -> str:
        self.last_inputs = inputs  # type: ignore[attr-defined]
        return "final answer"


@pytest.mark.asyncio
async def test_generate_response_skips_retrieval_when_gating_declines() -> None:
    settings = AppSettings(
        openai_api_key="stub",
        query_expansion_count=3,
        rerank_top_k=2,
    )

    gating_response = AIMessage(content=json.dumps({"should_retrieve": False, "assistant_response": "Hello there!"}))

    decider_stub = StubLLM([gating_response])
    expander_stub = StubLLM([], default=AIMessage(content=json.dumps({"queries": []})))

    memory_store = memory.ConversationMemoryStore(max_turns=5, summary_max_chars=2000)
    memory_store.append_turn("session", "Prior question", "Prior answer")

    context = RAGContext(
        retriever=None,
        base_vectorstore=None,
        base_retriever=StubRetriever({}),
        decider_llm=decider_stub,
        expander_llm=expander_stub,
        chain=StubChain(),
        llm=None,
        prompt=None,
        settings=settings,
        memory_store=memory_store,
        embeddings=None,
    )

    message = await generate_response("hi", session_id="session", context=context)

    assert message.content == "Hello there!"
    assert message.metadata is not None
    retrieval = message.metadata.get("retrieval", {})
    assert retrieval.get("used") is False
    assert retrieval.get("gating_reason") is None
    assert retrieval.get("queries") is None or retrieval.get("queries") == []

    assert decider_stub.last_messages is not None
    assert any("Prior question" in getattr(msg, "content", "") for msg in decider_stub.last_messages)


@pytest.mark.asyncio
async def test_generate_response_performs_expansion_and_reranking() -> None:
    prompt = "Summarise premium revenue"
    variant_one = f"{prompt} for recent years"
    variant_two = f"{prompt} with table excerpts"

    doc_a = Document(page_content="Base chunk", metadata={"chunk_id": "chunk-a", "source": "ReportA"})
    doc_b = Document(page_content="Premium revenue details", metadata={"chunk_id": "chunk-b", "source": "ReportB"})

    retriever = StubRetriever(
        {
            prompt: [doc_a],
            variant_one: [doc_b],
            variant_two: [doc_a],
        }
    )

    settings = AppSettings(
        openai_api_key="stub",
        query_expansion_count=4,
        rerank_top_k=2,
    )

    gating_response = AIMessage(content=json.dumps({"should_retrieve": True, "assistant_response": ""}))
    expansion_response = AIMessage(
        content=json.dumps({"queries": [variant_one, variant_two]})
    )

    decider_stub = StubLLM([gating_response])
    expander_stub = StubLLM([expansion_response], default=expansion_response)

    memory_store = memory.ConversationMemoryStore(max_turns=5, summary_max_chars=2000)
    memory_store.append_turn("session", "Initial topic", "Initial insight")

    context = RAGContext(
        retriever=None,
        base_vectorstore=None,
        base_retriever=retriever,
        decider_llm=decider_stub,
        expander_llm=expander_stub,
        chain=StubChain(),
        llm=None,
        prompt=None,
        settings=settings,
        memory_store=memory_store,
        embeddings=StubEmbeddings(),
    )

    message = await generate_response(prompt, session_id="session", context=context)

    assert message.metadata is not None
    retrieval = message.metadata.get("retrieval", {})
    assert retrieval.get("used") is True

    queries = retrieval.get("queries")
    assert isinstance(queries, list)
    assert queries[0] == prompt
    assert variant_one in queries
    assert variant_two in queries

    selection = retrieval.get("selection")
    assert isinstance(selection, list)
    assert selection[0]["chunk_id"] == "chunk-b"
    assert selection[0]["query"] == variant_one
    assert isinstance(selection[0]["score"], float)

    assert retriever.calls == [prompt, variant_one, variant_two]

    assert expander_stub.last_messages is not None
    assert any("Initial topic" in getattr(msg, "content", "") for msg in expander_stub.last_messages)


def test_memory_store_is_reused_across_contexts() -> None:
    settings = AppSettings(
        openai_api_key="stub",
        memory_max_turns=2,
        memory_summary_max_chars=100,
    )

    store_first = _get_or_create_memory_store(settings)
    assert store_first is not None

    store_first.append_turn("session", "first question", "first answer")

    context_one = build_rag_context(settings)
    assert context_one.memory_store is store_first

    # Force rebuild and confirm store persists
    context_two = build_rag_context(settings)
    assert context_two.memory_store is store_first
    rendered = store_first.render("session")
    assert "first question" in rendered
