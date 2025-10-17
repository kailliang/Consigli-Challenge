"""Query endpoints for analyst interactions."""

from __future__ import annotations

import uuid

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api import deps
from app.models.chat import QueryRequest, QueryResponse
from app.services import rag

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question about the indexed annual reports.",
)
async def query_reports(
    request: QueryRequest,
    rag_context: rag.RAGContext = Depends(deps.get_rag_context),
    db_session: AsyncSession = Depends(deps.get_db_session),
) -> QueryResponse:
    """Handle analyst questions via RAG pipeline."""

    try:
        validated = deps.validate_query(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session_id = validated.session_id or str(uuid.uuid4())

    message = await rag.generate_response(
        prompt_text=validated.prompt,
        session_id=session_id,
        context=rag_context,
        db_session=db_session,
    )

    return QueryResponse(message=message, session_id=session_id)


@router.post(
    "/query/stream",
    status_code=status.HTTP_200_OK,
    summary="Stream a RAG response via Server-Sent Events.",
)
async def query_reports_stream(
    request: QueryRequest,
    rag_context: rag.RAGContext = Depends(deps.get_rag_context),
    db_session: AsyncSession = Depends(deps.get_db_session),
) -> StreamingResponse:
    """Stream analyst answers using SSE."""

    try:
        validated = deps.validate_query(request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    session_id = validated.session_id or str(uuid.uuid4())

    async def event_generator():
        async for event in rag.stream_response(
            validated.prompt,
            session_id=session_id,
            context=rag_context,
            db_session=db_session,
        ):
            payload = json.dumps(event)
            yield f"data: {payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
