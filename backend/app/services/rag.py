"""Retrieval-augmented generation orchestration."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Iterable

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
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
    settings: AppSettings


_context: RAGContext | None = None


def build_rag_context(settings: AppSettings) -> RAGContext:
    """Instantiate retriever and chain resources."""

    if not settings.openai_api_key:
        logger.warning("rag.context.missing_api_key", message="OpenAI key not configured")
        return RAGContext(retriever=None, chain=None, settings=settings)

    try:
        vector_store = vectorstore.get_langchain_vectorstore(settings.vector_collection_name)
    except Exception as exc:  # pragma: no cover - depends on runtime resources
        logger.exception("rag.context.vectorstore_init_failed", exc_info=exc)
        return RAGContext(retriever=None, chain=None, settings=settings)

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
    return RAGContext(retriever=retriever, chain=chain, settings=settings)


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
        return _placeholder_response(
            "RAG pipeline is not yet connected to embeddings. "
            "Once ingestion is ready, this endpoint will return grounded answers with citations.",
        )

    try:
        documents: list[Document] = await context.retriever.aget_relevant_documents(prompt_text)
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.retriever.error", exc_info=exc)
        return _placeholder_response("Retrieval failed. Please verify vector store availability.")

    if not documents:
        return _placeholder_response(
            "No indexed content matched the question. Ingest reports or adjust the query."
        )

    table_lookup: dict[str, Any] = {}

    if db_session:
        table_ids = [doc.metadata.get("table_id") for doc in documents if doc.metadata.get("table_id")]
        table_lookup = await structured.fetch_tables_by_ids(db_session, table_ids)

    context_text = _format_context(documents, table_lookup)

    try:
        answer_text: str = await context.chain.ainvoke(
            {"question": prompt_text, "context": context_text}
        )
    except Exception as exc:  # pragma: no cover - external dependency
        logger.exception("rag.generation.error", exc_info=exc)
        return _placeholder_response("Language model failed to generate a response.")

    citations = _build_citations(documents, table_lookup)

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=answer_text.strip(),
        citations=citations,
    )


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
            body = row_text or doc.page_content
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


def _placeholder_response(message: str) -> ChatMessage:
    """Return a fallback assistant message."""

    return ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=message,
        citations=[],
    )
