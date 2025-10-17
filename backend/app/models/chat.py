"""Pydantic schemas for chat interactions."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Role = Literal["user", "assistant", "system"]


class Citation(BaseModel):
    id: str = Field(..., description="Unique identifier for the citation.")
    source: str = Field(..., description="Document identifier (file/path).")
    page: str | None = Field(default=None, description="Page reference string.")
    section: str | None = Field(default=None, description="Section title or table id.")
    snippet: str = Field(..., description="Quoted text supporting the answer.")


class ChatMessage(BaseModel):
    id: str = Field(..., description="Unique message identifier.")
    role: Role = Field(..., description="Conversation role.")
    content: str = Field(..., description="Message contents.")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    citations: list[Citation] = Field(default_factory=list)


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User question about the reports.")
    session_id: str | None = Field(default=None, description="Optional conversation session id.")


class QueryResponse(BaseModel):
    message: ChatMessage
    session_id: str
