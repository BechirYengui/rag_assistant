"""Vector similarity search over chunk embeddings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embeddings import get_embedding_provider
from app.models import Chunk, Document
from app.tokens import estimate_tokens


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source: str
    title: str | None
    chunk_index: int
    content: str
    similarity: float
    # Carried only between the vector search and the reranker; cleared before
    # the chunk leaves this module (never serialised to the API).
    embedding: list[float] | None = None


async def retrieve(
    session: AsyncSession,
    question: str,
    *,
    top_k: int | None = None,
    min_similarity: float | None = None,
    document_ids: list[uuid.UUID] | None = None,
) -> list[RetrievedChunk]:
    """Embed the question and return the most similar chunks (cosine)."""
    k = top_k or settings.retrieval_top_k
    floor = (
        min_similarity
        if min_similarity is not None
        else settings.retrieval_min_similarity
    )

    provider = get_embedding_provider()
    query_vec = await provider.embed_query(question)

    # When reranking, pull a larger candidate pool so MMR has room to trade
    # relevance for diversity; otherwise fetch exactly what we return.
    reranking = settings.rerank == "mmr"
    pool = (
        min(k * settings.rerank_fetch_multiplier, settings.rerank_max_pool)
        if reranking
        else k
    )

    # pgvector cosine_distance is in [0, 2]; similarity = 1 - distance.
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(Chunk, Document, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(pool)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    rows = (await session.execute(stmt)).all()

    candidates: list[RetrievedChunk] = []
    for chunk, document, dist in rows:
        similarity = 1.0 - float(dist)
        if similarity < floor:
            continue
        candidates.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                source=document.source,
                title=document.title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                similarity=round(similarity, 4),
                embedding=list(chunk.embedding) if reranking else None,
            )
        )

    if reranking and len(candidates) > k:
        from app.rerank import mmr_select

        results = mmr_select(candidates, k, settings.mmr_lambda)
    else:
        results = candidates[:k]

    # Drop embeddings: they are heavy and never leave this module.
    for chunk in results:
        chunk.embedding = None
    return results


def fit_to_budget(
    chunks: list[RetrievedChunk], max_tokens: int | None = None
) -> list[RetrievedChunk]:
    """Keep the highest-ranked chunks that fit within a token budget.

    Chunks arrive sorted by descending similarity. We add them until the next
    one would exceed ``max_context_tokens``, capping the input cost sent to
    Claude. The single most relevant chunk is always kept, even if it alone is
    over budget, so a question is never answered with empty context.
    """
    budget = max_tokens if max_tokens is not None else settings.max_context_tokens
    kept: list[RetrievedChunk] = []
    total = 0
    for chunk in chunks:
        cost = estimate_tokens(chunk.content)
        if kept and total + cost > budget:
            break
        kept.append(chunk)
        total += cost
    return kept
