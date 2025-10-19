"""Query endpoints for analyst interactions."""

from __future__ import annotations

import json
import uuid
from json import JSONDecodeError

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import ValidationError

from app.api import deps
from app.core.db import get_session
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


@router.websocket("/query/ws")
async def query_reports_websocket(
    websocket: WebSocket,
    rag_context: rag.RAGContext = Depends(deps.get_rag_context),
) -> None:
    """Stream analyst answers over a WebSocket connection."""

    await websocket.accept()

    try:
        while True:
            try:
                raw_message = await websocket.receive_text()
            except WebSocketDisconnect:
                break

            try:
                payload = json.loads(raw_message)
            except JSONDecodeError:
                await websocket.send_json({"event": "error", "data": {"message": "Invalid JSON payload."}})
                continue

            try:
                request = QueryRequest(**payload)
                validated = deps.validate_query(request)
            except (ValidationError, ValueError) as exc:
                await websocket.send_json({"event": "error", "data": {"message": str(exc)}})
                continue

            session_id = validated.session_id or str(uuid.uuid4())

            try:
                async with get_session() as db_session:
                    async for event in rag.stream_response(
                        validated.prompt,
                        session_id=session_id,
                        context=rag_context,
                        db_session=db_session,
                    ):
                        await websocket.send_json(event)
            except WebSocketDisconnect:
                break
            except Exception as exc:  # pragma: no cover - defensive logging
                await websocket.send_json(
                    {
                        "event": "error",
                        "data": {"message": "Query processing failed. Please retry.", "detail": str(exc)},
                    }
                )
    finally:
        try:
            await websocket.close()
        except RuntimeError:
            pass
