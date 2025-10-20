"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecodeError
from textwrap import shorten
from typing import Any, Iterable

import asyncio
from collections.abc import AsyncIterator
import re

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

try:  # pragma: no cover - optional dependency
    from langsmith.run_helpers import traceable
except Exception:  # pragma: no cover - environment without LangSmith
    def traceable(*_args, **_kwargs):
        def decorator(func):
            return func

        return decorator

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, Citation
from app.services import memory, structured, vectorstore

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGContext:
    """Container for reusable RAG components."""

    retriever: Any | None
    base_vectorstore: Any | None
    base_retriever: Any | None
    decider_llm: ChatOpenAI | None
    expander_llm: ChatOpenAI | None
    chain: Runnable | None
    llm: ChatOpenAI | None
    prompt: ChatPromptTemplate | None
    settings: AppSettings
    memory_store: memory.ConversationMemoryStore | None
    embeddings: OpenAIEmbeddings | None


@dataclass(slots=True)
class RetrievalDecision:
    should_retrieve: bool
    response: str | None
    reason: str | None


@dataclass(slots=True)
class RetrievalResult:
    documents: list[Document]
    table_lookup: dict[str, Any]
    context_text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class RankedDocument:
    document: Document
    score: float
    query: str


CURRENT_TIME_TOOL = {
    "name": "get_current_time",
    "description": "Return the current UTC date and time in ISO-8601 format.",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


_context: RAGContext | None = None
_memory_store: memory.ConversationMemoryStore | None = None
_memory_signature: tuple[int, int] | None = None


def build_rag_context(settings: AppSettings) -> RAGContext:
    """Instantiate retriever and chain resources."""

    if not settings.openai_api_key:
        logger.error("rag.context.missing_api_key", message="OpenAI key not configured")
        raise RuntimeError("OpenAI API key required for retrieval.")

    try:
        embeddings = vectorstore.create_embeddings(settings)
        vector_store = vectorstore.get_langchain_vectorstore(
            settings.vector_collection_name,
            settings=settings,
            embeddings=embeddings,
        )
    except Exception as exc:  # pragma: no cover - depends on runtime resources
        logger.exception("rag.context.vectorstore_init_failed", exc_info=exc)
        raise

    base_retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    decider_llm = ChatOpenAI(
        model=settings.retrieval_decider_model,
        temperature=settings.gating_temperature,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    expander_llm = ChatOpenAI(
        model=settings.query_expansion_model,
        temperature=settings.query_expansion_temperature,
        openai_api_key=settings.openai_api_key,
        openai_api_base=settings.openai_api_base,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an automotive financial analyst. "
                    "Use the provided context to answer with precise numbers and currency units. "
                    "If context is insufficient, explicitly state the limitation. "
                    "Incorporate relevant details from the conversation memory when it helps clarify the user's intent."
                ),
            ),
            (
                "human",
                "Conversation memory:\n{memory}\n\nContext:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    chain: Runnable = prompt | llm | StrOutputParser()

    memory_store = _get_or_create_memory_store(settings)

    return RAGContext(
        retriever=base_retriever,
        base_vectorstore=vector_store,
        base_retriever=base_retriever,
        decider_llm=decider_llm,
        expander_llm=expander_llm,
        chain=chain,
        llm=llm,
        prompt=prompt,
        settings=settings,
        memory_store=memory_store,
        embeddings=embeddings,
    )


def get_rag_context(settings: AppSettings | None = None) -> RAGContext:
    """Return a cached RAG context."""

    global _context
    settings = settings or get_settings()

    if _context is None or _context.settings is not settings:
        _context = build_rag_context(settings)

    return _context


def _get_or_create_memory_store(settings: AppSettings) -> memory.ConversationMemoryStore | None:
    """Return a shared memory store configured from settings."""

    global _memory_store, _memory_signature

    if not settings.memory_max_turns:
        _memory_store = None
        _memory_signature = None
        return None

    signature = (settings.memory_max_turns, settings.memory_summary_max_chars)

    if _memory_store is None or _memory_signature != signature:
        _memory_store = memory.ConversationMemoryStore(
            max_turns=settings.memory_max_turns,
            summary_max_chars=settings.memory_summary_max_chars,
        )
        _memory_signature = signature

    return _memory_store


@traceable(name="rag.generate_response")
async def generate_response(
    prompt_text: str,
    *,
    session_id: str,
    context: RAGContext,
    db_session: AsyncSession | None = None,
) -> ChatMessage:
    """Generate a response using the configured RAG pipeline."""

    if context.base_retriever is None or context.chain is None:
        raise RuntimeError("RAG context is not fully initialised.")

    memory_text = "None."
    memory_store = context.memory_store
    if memory_store is not None:
        rendered = memory_store.render(session_id).strip()
        memory_text = rendered if rendered else "None."

    decision = await _decide_retrieval(
        prompt_text,
        context=context,
        memory_text=memory_text,
    )
    retrieval_metadata: dict[str, Any] = {
        "retrieval": {
            "used": decision.should_retrieve,
            "gating_reason": decision.reason,
        }
    }

    if not decision.should_retrieve:
        direct_content = decision.response or "Happy to chat! How can I help with annual report analysis?"
        message = ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=direct_content,
            citations=[],
            metadata=retrieval_metadata,
        )
        if memory_store is not None:
            memory_store.append_turn(session_id, prompt_text, message.content)
        return message

    expanded_queries = await _expand_queries(
        prompt_text,
        context=context,
        limit=context.settings.query_expansion_count,
        memory_text=memory_text,
    )
    retrieval_metadata["retrieval"]["queries"] = expanded_queries

    retrieval = await _prepare_retrieval(
        context=context,
        prompt_text=prompt_text,
        db_session=db_session,
        queries=expanded_queries,
        metadata=retrieval_metadata,
    )

    if isinstance(retrieval, ChatMessage):
        if memory_store is not None:
            memory_store.append_turn(session_id, prompt_text, retrieval.content)
        return retrieval

    retrieval_metadata["retrieval"].update(retrieval.metadata)
    documents = retrieval.documents
    table_lookup = retrieval.table_lookup
    context_text = retrieval.context_text

    logger.debug(
        "rag.retrieval.summary",
        used=decision.should_retrieve,
        query_count=len(retrieval_metadata["retrieval"].get("queries", [])),
        selected_chunks=len(documents),
    )

    try:
        answer_text: str = await context.chain.ainvoke(
            {"question": prompt_text, "context": context_text, "memory": memory_text}
        )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.generation.error", exc_info=exc)
        message = _placeholder_response(
            "Language model failed to generate a response.",
            metadata=retrieval_metadata,
        )
        if memory_store is not None:
            memory_store.append_turn(session_id, prompt_text, message.content)
        return message

    validated_content, _ = _apply_numeric_validation(answer_text, table_lookup)
    citations = _build_citations(documents, table_lookup)

    message = ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=validated_content,
        citations=citations,
        metadata=retrieval_metadata,
    )
    if memory_store is not None:
        memory_store.append_turn(session_id, prompt_text, message.content)
    return message


async def stream_response(
    prompt_text: str,
    *,
    session_id: str,
    context: RAGContext,
    db_session: AsyncSession | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a response event sequence.

    Streaming is currently disabled; return a single chunk response.
    """

    message = await generate_response(
        prompt_text,
        session_id=session_id,
        context=context,
        db_session=db_session,
    )
    yield {"event": "token", "data": {"content": message.content}}
    yield {"event": "done", "data": {"session_id": session_id, "message": message.model_dump(mode='json')}}


@traceable(name="rag.prepare_retrieval")
async def _prepare_retrieval(
    *,
    context: RAGContext,
    prompt_text: str,
    db_session: AsyncSession | None,
    queries: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ChatMessage | RetrievalResult:
    query_list = queries or [prompt_text]
    metadata = metadata or {"retrieval": {}}
    retrieval_section = metadata.setdefault("retrieval", {})
    retrieval_section.setdefault("queries", query_list)

    try:
        documents_with_origin = await _multi_query_retrieve(context.base_retriever, query_list)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.retriever.error", exc_info=exc)
        retrieval_section["error"] = "retriever_failed"
        return _placeholder_response(
            "Retriever failed to fetch supporting context. Please retry later.",
            metadata=metadata,
        )

    if not documents_with_origin:
        retrieval_section["error"] = "no_results"
        return _placeholder_response("No relevant context found for that query.", metadata=metadata)

    rerank_query = query_list[0] if query_list else prompt_text

    ranked_documents = await _rerank_documents(
        documents_with_origin,
        rerank_query,
        embeddings=context.embeddings,
        top_k=context.settings.rerank_top_k,
    )

    if not ranked_documents:
        retrieval_section["error"] = "rerank_filtered"
        return _placeholder_response("No relevant context found for that query.", metadata=metadata)

    documents: list[Document] = []
    selection: list[dict[str, Any]] = []

    for ranked in ranked_documents:
        doc = ranked.document
        metadata = dict(doc.metadata or {})
        metadata.setdefault("retrieval_query", ranked.query)
        metadata["retrieval_score"] = ranked.score
        doc.metadata = metadata

        documents.append(doc)
        selection.append(
            {
                "chunk_id": metadata.get("chunk_id"),
                "chunk_text": doc.page_content,
                "query": ranked.query,
                "score": ranked.score,
            }
        )

    table_lookup: dict[str, Any] = {}
    if db_session:
        table_ids = [doc.metadata.get("table_id") for doc in documents if doc.metadata.get("table_id")]
        table_lookup = await structured.fetch_tables_by_ids(db_session, table_ids)

    context_text = _format_context(documents, table_lookup)

    return RetrievalResult(
        documents=documents,
        table_lookup=table_lookup,
        context_text=context_text,
        metadata={"selection": selection},
    )


@traceable(name="rag.multi_query_retrieve")
async def _multi_query_retrieve(
    retriever: Any,
    queries: list[str],
) -> list[tuple[Document, str]]:
    """Execute multiple queries against a retriever, emulating MultiQueryRetriever."""

    tasks = [_run_single_query(retriever, query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    collected: list[tuple[Document, str]] = []
    for query, result in zip(queries, results):
        if isinstance(result, Exception):  # pragma: no cover - defensive logging
            logger.warning("rag.retriever.query_failed", query=query, error=str(result))
            continue
        for doc in result:
            collected.append((doc, query))

    unique = _deduplicate_documents_with_origin(collected)
    return unique


@traceable(name="rag.run_single_query")
async def _run_single_query(retriever: Any, query: str) -> list[Document]:
    """Invoke a retriever with a single query, supporting sync and async APIs."""

    if hasattr(retriever, "aget_relevant_documents"):
        documents = await retriever.aget_relevant_documents(query)  # type: ignore[assignment]
    elif hasattr(retriever, "get_relevant_documents"):
        documents = await asyncio.to_thread(retriever.get_relevant_documents, query)  # type: ignore[arg-type]
    elif hasattr(retriever, "ainvoke"):
        documents = await retriever.ainvoke(query)  # type: ignore[assignment]
    else:
        documents = await asyncio.to_thread(retriever.invoke, query)  # type: ignore[arg-type]

    if isinstance(documents, Document):
        return [documents]
    return list(documents or [])


def _deduplicate_documents_with_origin(
    items: list[tuple[Document, str]]
) -> list[tuple[Document, str]]:
    """Remove duplicate documents while preserving query provenance."""

    seen: set[str] = set()
    unique: list[tuple[Document, str]] = []

    for doc, query in items:
        key = _document_identity(doc)
        if key in seen:
            continue
        seen.add(key)
        unique.append((doc, query))

    return unique


def _document_identity(doc: Document) -> str:
    metadata = doc.metadata or {}
    for key in ("chunk_id", "id", "source", "document_name"):
        value = metadata.get(key)
        if value:
            return str(value)
    digest = hashlib.sha1(doc.page_content.encode("utf-8")).hexdigest()
    return digest


async def _rerank_documents(
    documents_with_origin: list[tuple[Document, str]],
    query_text: str,
    *,
    embeddings: OpenAIEmbeddings | None,
    top_k: int,
) -> list[RankedDocument]:
    """Rerank retrieved documents by cosine similarity to the original query."""

    if not documents_with_origin:
        return []

    docs = [doc for doc, _ in documents_with_origin]

    if embeddings is None:
        return [RankedDocument(document=doc, score=1.0, query=query) for doc, query in documents_with_origin[:top_k]]

    try:
        query_vector = await _embed_query(embeddings, query_text)
        doc_vectors = await _embed_documents(embeddings, [doc.page_content for doc in docs])
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.rerank.error", exc_info=exc)
        return [RankedDocument(document=doc, score=1.0, query=query) for doc, query in documents_with_origin[:top_k]]

    ranked: list[RankedDocument] = []

    for (doc, query), vector in zip(documents_with_origin, doc_vectors):
        score = _cosine_similarity(query_vector, vector)
        ranked.append(RankedDocument(document=doc, score=score, query=query))

    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]


async def _embed_query(embeddings: OpenAIEmbeddings, text: str) -> list[float]:
    if hasattr(embeddings, "aembed_query"):
        return await embeddings.aembed_query(text)
    return await asyncio.to_thread(embeddings.embed_query, text)


async def _embed_documents(embeddings: OpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    if hasattr(embeddings, "aembed_documents"):
        return await embeddings.aembed_documents(texts)
    return await asyncio.to_thread(embeddings.embed_documents, texts)


def _cosine_similarity(vector_a: Iterable[float], vector_b: Iterable[float]) -> float:
    a = list(float(x) for x in vector_a)
    b = list(float(x) for x in vector_b)
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
def _format_context(documents: Iterable[Document], table_lookup: dict[str, Any]) -> str:
    context_blocks: list[str] = []

    for index, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("document_name") or metadata.get("source") or "unknown"
        chunk_type = metadata.get("chunk_type", "text")
        table_id = metadata.get("table_id")
        page_range = metadata.get("page_range")

        header_parts = [f"Source {index}", source, f"type={chunk_type}"]
        if page_range:
            header_parts.append(f"page={page_range}")
        if table_id:
            header_parts.append(f"table={table_id}")

        header = " | ".join(part for part in header_parts if part)

        if table_id and table_id in table_lookup:
            table_summary = table_lookup[table_id]
            rows = table_summary.rows or []
            preview_rows = rows[:5]
            row_text = "\n".join(
                " | ".join(f"{key}: {value}" for key, value in row.items() if value) for row in preview_rows
            )
            time_series = structured.summarize_time_series(table_summary)
            summary_text = "\n".join(time_series)
            combined = "\n\n".join(part for part in [row_text, summary_text] if part)
            body = combined or doc.page_content
        else:
            body = doc.page_content

        context_blocks.append(f"{header}\n{body}")

    return "\n\n".join(context_blocks)


def _build_citations(documents: Iterable[Document], table_lookup: dict[str, Any]) -> list[Citation]:
    citations: list[Citation] = []
    seen_ids: set[str] = set()

    for doc in documents:
        metadata = doc.metadata or {}
        chunk_id = str(metadata.get("chunk_id") or uuid.uuid4())

        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        source = str(metadata.get("document_name") or metadata.get("source") or "unknown")
        page = metadata.get("page_range")
        section = metadata.get("table_id") or metadata.get("chunk_type")

        snippet_source = doc.page_content
        if metadata.get("table_id") and metadata.get("table_id") in table_lookup:
            table_summary = table_lookup[metadata["table_id"]]
            rows = table_summary.rows or []
            first_row = rows[0] if rows else {}
            snippet_source = " | ".join(f"{k}: {v}" for k, v in first_row.items() if v) or snippet_source

        snippet = shorten(" ".join(snippet_source.split()), width=280, placeholder="…")

        citations.append(
            Citation(
                id=chunk_id,
                source=source,
                page=page,
                section=section,
                snippet=snippet,
            )
        )

    return citations


def _streaming_not_allowed(exc: Exception) -> bool:
    message = str(exc).lower()
    if "must be verified to stream this model" in message:
        return True
    if "unsupported_value" in message and "stream" in message:
        return True
    return False


def _apply_numeric_validation(
    answer_text: str,
    table_lookup: dict[str, Any],
) -> tuple[str, str | None]:
    """Validate numeric literals against retrieved table rows."""

    base_text = answer_text.rstrip()

    if not table_lookup:
        return base_text.strip(), None

    extracted_numbers = {
        _normalize_number(match) for match in re.findall(r"\b[+-]?\d[\d,\.]*\b", base_text)
    }
    extracted_numbers.discard("")

    if not extracted_numbers:
        return base_text.strip(), None

    table_numbers: set[str] = set()
    for summary in table_lookup.values():
        rows = summary.rows or []
        for row in rows:
            for value in row.values():
                normalized = _normalize_number(str(value))
                if normalized:
                    table_numbers.add(normalized)

    missing = sorted(num for num in extracted_numbers if num not in table_numbers)

    if missing:
        suffix = (
            "\n\nValidation warning: unable to verify the following values against retrieved tables — "
            + ", ".join(missing)
        )
        return base_text.strip() + suffix, suffix

    suffix = "\n\nValidation: numeric values verified against retrieved tables."
    return base_text.strip() + suffix, suffix


def _normalize_number(value: str) -> str:
    cleaned = value.replace(",", "").strip()
    if not cleaned:
        return ""
    cleaned = cleaned.rstrip(".")
    return cleaned


def _placeholder_response(message: str, *, metadata: dict[str, Any] | None = None) -> ChatMessage:
    """Return a fallback assistant message."""

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=message,
        citations=[],
        metadata=metadata,
    )


@traceable(name="rag.decide_retrieval")
async def _decide_retrieval(
    prompt_text: str,
    *,
    context: RAGContext,
    memory_text: str | None = None,
) -> RetrievalDecision:
    """Determine whether retrieval is required for a user prompt."""

    llm = context.decider_llm
    if llm is None:
        return RetrievalDecision(should_retrieve=True, response=None, reason=None)

    system_prompt = (
        "You route automotive financial analysis questions. "
        "Respond with a JSON object containing: "
        '"should_retrieve" (boolean), "assistant_response" (string), and optional concise "reason" (string), within 30 words. '
        "If retrieval is not needed, provide a short natural-language reply in "
        '"assistant_response" explaining why. If retrieval is needed, set "assistant_response" to an empty string. '
        "Consider the provided conversation memory when determining whether the user is referencing earlier turns."
    )

    memory_section = (memory_text or "").strip() or "None."
    active_entity = _extract_active_entity(memory_section)
    human_payload = (
        f"Conversation memory:\n{memory_section}\n\n"
        f"User message:\n{prompt_text}"
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_payload),
    ]

    try:
        result = await llm.ainvoke(messages)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.gating.failed", exc_info=exc)
        return RetrievalDecision(should_retrieve=True, response=None, reason="decider_error")

    payload = _parse_json_content(_message_content_to_text(result))

    should_retrieve = bool(payload.get("should_retrieve", True))
    response = payload.get("assistant_response") or payload.get("response")
    reason = payload.get("reason")

    return RetrievalDecision(should_retrieve=should_retrieve, response=response, reason=reason)


@traceable(name="rag.expand_queries")
async def _expand_queries(
    prompt_text: str,
    *,
    context: RAGContext,
    limit: int,
    memory_text: str | None = None,
) -> list[str]:
    """Generate expanded retrieval queries using an LLM with tool support."""

    system_prompt = (
        "You expand user questions into multiple focused search queries about automotive financial data. "
        "Return a JSON object with a \"queries\" array of clear, self-contained questions. "
        f"Generate between 3 and {limit} queries that cover different angles, time periods, or companies as needed. "
        "Use the conversation memory to preserve context (e.g., companies or metrics already identified)."
    )

    memory_section = (memory_text or "").strip() or "None."
    active_entity = _extract_active_entity(memory_section)
    rewritten_prompt = _rewrite_query_with_entity(prompt_text, active_entity)

    if limit <= 1:
        return [rewritten_prompt]

    llm = context.expander_llm
    if llm is None:
        return [rewritten_prompt]

    human_payload = (
        f"Conversation memory:\n{memory_section}\n\n"
        f"User message:\n{prompt_text}"
    )

    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_payload),
    ]

    try:
        ai_message = await _invoke_with_tools(llm, messages=messages, tools=[CURRENT_TIME_TOOL])
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.query_expansion.failed", exc_info=exc)
        return [prompt_text]

    payload = _parse_json_content(_message_content_to_text(ai_message))
    raw_queries = payload.get("queries")

    if not isinstance(raw_queries, list):
        return [rewritten_prompt]

    candidate_queries = [
        _rewrite_query_with_entity(str(query), active_entity)
        for query in raw_queries
        if isinstance(query, str)
    ]
    expanded = _deduplicate_queries(
        rewritten_prompt,
        candidate_queries,
        limit,
    )
    return expanded


async def _invoke_with_tools(
    llm: ChatOpenAI,
    *,
    messages: list[BaseMessage],
    tools: list[dict[str, Any]],
) -> AIMessage:
    """Invoke an LLM, fulfilling any tool calls before returning the final response."""

    if not tools:
        return await llm.ainvoke(messages)

    bound_llm = llm.bind_tools(tools)
    conversation: list[BaseMessage] = list(messages)
    result = await bound_llm.ainvoke(conversation)

    while getattr(result, "tool_calls", None):
        tool_messages: list[ToolMessage] = []
        for tool_call in result.tool_calls or []:
            tool_name = getattr(tool_call, "name", None) or tool_call.get("name")  # type: ignore[union-attr]
            tool_id = getattr(tool_call, "id", None) or tool_call.get("id")  # type: ignore[union-attr]
            args = getattr(tool_call, "args", {}) or tool_call.get("args", {})  # type: ignore[union-attr]

            if tool_name == CURRENT_TIME_TOOL["name"]:
                now = datetime.now(tz=timezone.utc).isoformat()
                content = json.dumps({"current_time": now, "args": args})
            else:
                logger.warning("rag.tool.unknown", tool_name=tool_name)
                content = json.dumps({"error": f"Unknown tool: {tool_name}", "args": args})

            tool_messages.append(ToolMessage(content=content, tool_call_id=str(tool_id or "")))

        conversation.extend([result, *tool_messages])
        result = await bound_llm.ainvoke(conversation)

    return result


def _message_content_to_text(message: AIMessage) -> str:
    """Coerce message content into a string for downstream parsing."""

    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "".join(parts)
    return str(content)


def _parse_json_content(raw: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response string."""

    if not raw:
        return {}

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fence(cleaned)

    try:
        data = json.loads(cleaned)
    except JSONDecodeError:
        logger.warning("rag.json.parse_failed", content=cleaned[:200])
        return {}

    return data if isinstance(data, dict) else {}


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1] if stripped.count("```") >= 2 else stripped.lstrip("`")
    if stripped.lower().startswith("json"):
        stripped = stripped[4:]
    return stripped.strip()


_PRONOUN_RE = re.compile(r"\b(it|its)\b", re.IGNORECASE)
_ENTITY_CANDIDATE_RE = re.compile(r"\b([A-Z][A-Za-z0-9&'/-]*)\b(?:'s)?")
_ENTITY_STOPWORDS = {
    "user",
    "assistant",
    "conversation",
    "recent",
    "dialogue",
    "summary",
    "none",
    "question",
    "context",
    "company",
    "year",
    "document",
    "table",
    "revenue",
    "profit",
    "growth",
    "report",
}


def _deduplicate_queries(original: str, candidates: Iterable[str], limit: int) -> list[str]:
    """Ensure a deterministic, de-duplicated query list."""

    seen: set[str] = set()
    ordered: list[str] = []

    def add_query(query: str) -> None:
        normalized = query.strip()
        if not normalized:
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        ordered.append(normalized)

    add_query(original)
    for candidate in candidates:
        add_query(candidate)
        if len(ordered) >= limit:
            break

    return ordered or [original]


def _rewrite_query_with_entity(query: str, entity: str | None) -> str:
    """Replace pronoun references with the provided entity."""

    if not entity:
        return query.strip()

    def _replacement(match) -> str:
        token = match.group(0)
        if token.lower() == "its":
            possessive = _to_possessive(entity)
            return possessive
        return entity

    rewritten = _PRONOUN_RE.sub(_replacement, query)
    rewritten = rewritten.strip()
    if rewritten and rewritten[0].islower():
        rewritten = rewritten[0].upper() + rewritten[1:]
    return rewritten


def _extract_active_entity(memory_text: str) -> str | None:
    """Infer the most recent capitalised entity from the conversation memory."""

    matches = list(_ENTITY_CANDIDATE_RE.finditer(memory_text))
    for match in reversed(matches):
        candidate = match.group(1).strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if lower in _ENTITY_STOPWORDS:
            continue
        if len(candidate) < 2:
            continue
        return candidate
    return None


def _to_possessive(entity: str) -> str:
    trimmed = entity.rstrip()
    if not trimmed:
        return entity
    lower = trimmed.lower()
    if lower.endswith("'s") or lower.endswith("'"):
        return trimmed
    if trimmed.endswith("s"):
        return f"{trimmed}'"
    return f"{trimmed}'s"
