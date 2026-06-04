"""Pydantic request/response models, the public API contract."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# Ingestion
class IngestTextRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw document text.")
    source: str = Field(..., description="Identifier/URI of the document.")
    title: str | None = None
    metadata: dict = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    id: uuid.UUID
    source: str
    title: str | None
    metadata: dict
    chunk_count: int
    created_at: datetime


class IngestResponse(BaseModel):
    document: DocumentResponse
    chunks_created: int


# Query
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(
        default=None, ge=1, le=50, description="Override default retrieval depth."
    )
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    document_ids: list[uuid.UUID] | None = Field(
        default=None, description="Restrict retrieval to these documents."
    )


class SourceChunk(BaseModel):
    """A retrieved chunk, surfaced so the caller can audit the answer."""

    citation: int = Field(..., description="The [n] marker used in the answer.")
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source: str
    title: str | None
    chunk_index: int
    similarity: float
    content: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    model: str
