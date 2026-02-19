from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


AnswerMode = Literal["short", "5_mark", "10_mark", "detailed", "numerical"]


class ChunkRef(BaseModel):
    source_name: str
    source_type: str
    page: Optional[int] = None
    subject: str
    unit: Optional[str] = None
    topic: Optional[str] = None


class ChatRequest(BaseModel):
    subject: str = Field(..., min_length=2, max_length=64)
    question: str = Field(..., min_length=2)
    mode: AnswerMode = "5_mark"
    top_k: int = Field(6, ge=1, le=20)


class ChatResponse(BaseModel):
    answer: str
    chunks_used: list[ChunkRef] = Field(default_factory=list)


class IngestResponse(BaseModel):
    subject: str
    chunks_added: int
    source_name: str
    source_type: str
    details: dict[str, Any] = {}
