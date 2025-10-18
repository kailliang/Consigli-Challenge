"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Iterable, cast

from collections.abc import AsyncIterator
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AppSettings, get_settings
from app.core.logging import get_logger
from app.models.chat import ChatMessage, Citation
from app.services import structured, vectorstore

logger = get_logger(__name__)


@dataclass(slots=True)
class RAGContext:
    """Container for reusable RAG components."""

    retriever: Any | None
    chain: Runnable | None
    llm: ChatOpenAI | None
    prompt: ChatPromptTemplate | None
    settings: AppSettings


_context: RAGContext | None = None
_chunk_cache: list[dict[str, Any]] | None = None


def _load_chunk_index(settings: AppSettings) -> list[dict[str, Any]]:
    global _chunk_cache

    if _chunk_cache is not None:
        return _chunk_cache

    path = Path(settings.chunk_index_path)
    if not path.exists():
        logger.warning("rag.chunk_index.missing", path=str(path))
        _chunk_cache = []
        return _chunk_cache

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            _chunk_cache = cast(list[dict[str, Any]], data)
        else:
            logger.warning("rag.chunk_index.invalid", path=str(path))
            _chunk_cache = []
    except Exception as exc:  # pragma: no cover
        logger.exception("rag.chunk_index.load_failed", exc_info=exc, path=str(path))
        _chunk_cache = []

    return _chunk_cache


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"[\w$€£%\.]+", text)}


def _score(query_tokens: set[str], chunk_tokens: set[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    overlap = len(query_tokens & chunk_tokens)
    if overlap == 0:
        return 0.0
    return overlap / math.sqrt(len(query_tokens) * len(chunk_tokens))


def _fallback_chunk_response(prompt_text: str, settings: AppSettings) -> ChatMessage:
    chunks = _load_chunk_index(settings)
    if not chunks:
        return _placeholder_response(
            "No ingested chunks available yet. Please run the ingestion pipeline first."
        )

    query_tokens = _tokenize(prompt_text)
    best_chunk: dict[str, Any] | None = None
    best_score = 0.0

    for chunk in chunks:
        chunk_tokens = _tokenize(str(chunk.get("text", "")))
        score = _score(query_tokens, chunk_tokens)
        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_chunk is None or best_score == 0.0:
        return _placeholder_response(
            "I couldn't find a relevant chunk for that question in the current corpus."
        )

    section = best_chunk.get("section")

    citation = Citation(
        id=str(best_chunk.get("chunk_id", uuid.uuid4())),
        source=str(
            best_chunk.get("document_name")
            or best_chunk.get("source_path")
            or "unknown"
        ),
        section=section,
        page=None,
        snippet=shorten(str(best_chunk.get("text", "")), width=280, placeholder="…"),
    )

    section_suffix = f" (section: {section})" if section else ""
    content = f"Top match from {citation.source}{section_suffix}.\n\n{best_chunk.get('text', '').strip()}"

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=content,
        citations=[citation],
    )


def build_rag_context(settings: AppSettings) -> RAGContext:
    """Instantiate retriever and chain resources."""

    if not settings.openai_api_key:
        logger.warning("rag.context.missing_api_key", message="OpenAI key not configured")
        return RAGContext(retriever=None, chain=None, llm=None, prompt=None, settings=settings)

    try:
        vector_store = vectorstore.get_langchain_vectorstore(settings.vector_collection_name)
    except Exception as exc:  # pragma: no cover - depends on runtime resources
        logger.exception("rag.context.vectorstore_init_failed", exc_info=exc)
        return RAGContext(retriever=None, chain=None, llm=None, prompt=None, settings=settings)

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
                    "If context is insufficient, explicitly state the limitation."
                ),
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion: {question}",
            ),
        ]
    )

    chain: Runnable = prompt | llm | StrOutputParser()
    return RAGContext(retriever=retriever, chain=chain, llm=llm, prompt=prompt, settings=settings)


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
        return _fallback_chunk_response(prompt_text, context.settings)

    retrieval = await _prepare_retrieval(
        context=context,
        prompt_text=prompt_text,
        db_session=db_session,
    )

    if isinstance(retrieval, ChatMessage):
        return retrieval

    documents, table_lookup, context_text = retrieval

    try:
        answer_text: str = await context.chain.ainvoke(
            {"question": prompt_text, "context": context_text}
        )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.generation.error", exc_info=exc)
        return _placeholder_response("Language model failed to generate a response.")

    validated_content, _ = _apply_numeric_validation(answer_text, table_lookup)
    citations = _build_citations(documents, table_lookup)

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=validated_content,
        citations=citations,
    )


async def stream_response(
    prompt_text: str,
    *,
    session_id: str,
    context: RAGContext,
    db_session: AsyncSession | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a response event sequence.

    This stub currently streams the final message in one chunk to unblock frontend plumbing.
    """

    if context.retriever is None or context.chain is None or context.llm is None or context.prompt is None:
        message = await generate_response(
            prompt_text=prompt_text,
            session_id=session_id,
            context=context,
            db_session=db_session,
        )
        yield {"event": "token", "data": {"content": message.content}}
        yield {"event": "done", "data": {"session_id": session_id, "message": message.model_dump()}}
        return

    retrieval = await _prepare_retrieval(
        context=context,
        prompt_text=prompt_text,
        db_session=db_session,
    )

    if isinstance(retrieval, ChatMessage):
        yield {"event": "token", "data": {"content": retrieval.content}}
        yield {"event": "done", "data": {"session_id": session_id, "message": retrieval.model_dump()}}
        return

    documents, table_lookup, context_text = retrieval

    prompt_value = context.prompt.format_prompt(question=prompt_text, context=context_text)
    messages = prompt_value.to_messages()

    accumulated: list[str] = []

    try:
        async for chunk in context.llm.astream(messages):
            token = getattr(chunk, "content", None)
            if not token:
                token = getattr(chunk.message, "content", None)
            if not token:
                continue
            accumulated.append(token)
            yield {"event": "token", "data": {"content": token}}
    except Exception as exc:  # pragma: no cover
        logger.exception("rag.streaming.error", exc_info=exc)
        message = _placeholder_response("Streaming failed unexpectedly. Please retry.")
        yield {"event": "token", "data": {"content": message.content}}
        yield {"event": "done", "data": {"session_id": session_id, "message": message.model_dump()}}
        return

    final_content = "".join(accumulated)
    validated_content, suffix = _apply_numeric_validation(final_content, table_lookup)

    if suffix:
        yield {"event": "token", "data": {"content": suffix}}

    citations = _build_citations(documents, table_lookup)
    message = ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=validated_content,
        citations=citations,
    )

    yield {"event": "done", "data": {"session_id": session_id, "message": message.model_dump()}}


async def _prepare_retrieval(
    *,
    context: RAGContext,
    prompt_text: str,
    db_session: AsyncSession | None,
) -> ChatMessage | tuple[list[Document], dict[str, Any], str]:
    try:
        documents: list[Document] = await context.retriever.aget_relevant_documents(prompt_text)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.retriever.error", exc_info=exc)
        return _fallback_chunk_response(prompt_text, context.settings)

    if not documents:
        return _fallback_chunk_response(prompt_text, context.settings)

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
