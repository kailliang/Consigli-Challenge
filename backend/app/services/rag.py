"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Iterable

import asyncio
from collections.abc import AsyncIterator
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, Citation
from app.services import memory, structured, vectorstore

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGContext:
    """Container for reusable RAG components."""

    retriever: Any | None
    chain: Runnable | None
    llm: ChatOpenAI | None
    prompt: ChatPromptTemplate | None
    settings: AppSettings
    memory_store: memory.ConversationMemoryStore | None


_context: RAGContext | None = None


def build_rag_context(settings: AppSettings) -> RAGContext:
    """Instantiate retriever and chain resources."""

    if not settings.openai_api_key:
        logger.error("rag.context.missing_api_key", message="OpenAI key not configured")
        raise RuntimeError("OpenAI API key required for retrieval.")

    try:
        vector_store = vectorstore.get_langchain_vectorstore(settings.vector_collection_name)
    except Exception as exc:  # pragma: no cover - depends on runtime resources
        logger.exception("rag.context.vectorstore_init_failed", exc_info=exc)
        raise

    retriever = vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.2,
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

    memory_store: memory.ConversationMemoryStore | None = None
    if settings.memory_max_turns:
        memory_store = memory.ConversationMemoryStore(
            max_turns=settings.memory_max_turns,
            summary_max_chars=settings.memory_summary_max_chars,
        )

    return RAGContext(
        retriever=retriever,
        chain=chain,
        llm=llm,
        prompt=prompt,
        settings=settings,
        memory_store=memory_store,
    )


def get_rag_context(settings: AppSettings | None = None) -> RAGContext:
    """Return a cached RAG context."""

    global _context
    settings = settings or get_settings()

    if _context is None or _context.settings is not settings:
        _context = build_rag_context(settings)

    return _context


async def generate_response(
    prompt_text: str,
    *,
    session_id: str,
    context: RAGContext,
    db_session: AsyncSession | None = None,
) -> ChatMessage:
    """Generate a response using the configured RAG pipeline."""

    if context.retriever is None or context.chain is None:
        raise RuntimeError("RAG context is not fully initialised.")

    memory_text = "None."
    memory_store = context.memory_store
    if memory_store is not None:
        rendered = memory_store.render(session_id).strip()
        memory_text = rendered if rendered else "None."

    retrieval = await _prepare_retrieval(
        context=context,
        prompt_text=prompt_text,
        db_session=db_session,
    )

    if isinstance(retrieval, ChatMessage):
        if memory_store is not None:
            memory_store.append_turn(session_id, prompt_text, retrieval.content)
        return retrieval

    documents, table_lookup, context_text = retrieval

    try:
        answer_text: str = await context.chain.ainvoke(
            {"question": prompt_text, "context": context_text, "memory": memory_text}
        )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.generation.error", exc_info=exc)
        message = _placeholder_response("Language model failed to generate a response.")
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


async def _prepare_retrieval(
    *,
    context: RAGContext,
    prompt_text: str,
    db_session: AsyncSession | None,
) -> ChatMessage | tuple[list[Document], dict[str, Any], str]:
    try:
        if hasattr(context.retriever, "aget_relevant_documents"):
            documents = await context.retriever.aget_relevant_documents(prompt_text)  # type: ignore[assignment]
        elif hasattr(context.retriever, "get_relevant_documents"):
            documents = await asyncio.to_thread(
                context.retriever.get_relevant_documents,  # type: ignore[arg-type]
                prompt_text,
            )
        elif hasattr(context.retriever, "ainvoke"):
            documents = await context.retriever.ainvoke(prompt_text)  # type: ignore[assignment]
        else:
            documents = await asyncio.to_thread(
                context.retriever.invoke,  # type: ignore[arg-type]
                prompt_text,
            )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.retriever.error", exc_info=exc)
        return _placeholder_response("Retriever failed to fetch supporting context. Please retry later.")

    if isinstance(documents, Document):
        documents = [documents]
    documents = list(documents or [])

    if not documents:
        return _placeholder_response("No relevant context found for that query.")

    table_lookup: dict[str, Any] = {}

    if db_session:
        table_ids = [doc.metadata.get("table_id") for doc in documents if doc.metadata.get("table_id")]
        table_lookup = await structured.fetch_tables_by_ids(db_session, table_ids)

    context_text = _format_context(documents, table_lookup)
    return documents, table_lookup, context_text


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


def _placeholder_response(message: str) -> ChatMessage:
    """Return a fallback assistant message."""

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=message,
        citations=[],
    )
