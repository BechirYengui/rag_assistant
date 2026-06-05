"""High-level RAG orchestration tying retrieval and generation together."""

from __future__ import annotations

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import answer_cache
from app.config import settings
from app.llm import generate_answer
from app.retrieval import RetrievedChunk, fit_to_budget, retrieve
from app.schemas import QueryResponse, SourceChunk, Usage

logger = logging.getLogger("rag")


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


async def retrieve_context(
    session: AsyncSession,
    question: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the most relevant chunks, trimmed to the token budget."""
    chunks = await retrieve(
        session,
        question,
        top_k=top_k,
        min_similarity=min_similarity,
        document_ids=document_ids,
    )
    return fit_to_budget(chunks)


async def answer_question(
    session: AsyncSession,
    question: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> QueryResponse:
    """Retrieve relevant context and synthesise a cited answer.

    Short-circuits on a cache hit (zero Claude tokens) and reports the token
    usage billed for every live generation.
    """
    chunks = await retrieve_context(
        session,
        question,
        top_k=top_k,
        min_similarity=min_similarity,
        document_ids=document_ids,
    )
    sources = _to_sources(chunks)

    cache_key = answer_cache.make_key(question, [str(c.chunk_id) for c in chunks])
    cached = answer_cache.get(cache_key)
    if cached is not None:
        logger.info("answer cache hit (0 tokens)")
        return QueryResponse(
            question=question,
            answer=cached,
            sources=sources,
            model=settings.llm_model,
            usage=Usage(),
            cached=True,
        )

    answer, usage = await generate_answer(question, chunks)
    answer_cache.set(cache_key, answer)
    logger.info(
        "generated answer (in=%d out=%d tokens, %d chunks)",
        usage.input_tokens,
        usage.output_tokens,
        len(chunks),
    )
    return QueryResponse(
        question=question,
        answer=answer,
        sources=sources,
        model=settings.llm_model,
        usage=usage,
        cached=False,
    )
