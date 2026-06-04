"""High-level RAG orchestration tying retrieval and generation together."""

from __future__ import annotations

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.llm import generate_answer
from app.retrieval import RetrievedChunk, retrieve
from app.schemas import QueryResponse, SourceChunk


def _to_sources(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    return [
        SourceChunk(
            citation=i,
            chunk_id=c.chunk_id,
            document_id=c.document_id,
            source=c.source,
            title=c.title,
            chunk_index=c.chunk_index,
            similarity=c.similarity,
            content=c.content,
        )
        for i, c in enumerate(chunks, start=1)
    ]


async def answer_question(
    session: AsyncSession,
    question: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> QueryResponse:
    """Retrieve relevant context and synthesise a cited answer."""
    chunks = await retrieve(
        session,
        question,
        top_k=top_k,
        min_similarity=min_similarity,
        document_ids=document_ids,
    )
    answer = await generate_answer(question, chunks)
    return QueryResponse(
        question=question,
        answer=answer,
        sources=_to_sources(chunks),
        model=settings.llm_model,
    )
