"""Vector similarity search over chunk embeddings."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.embeddings import get_embedding_provider
from app.models import Chunk, Document


@dataclass
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source: str
    title: str | None
    chunk_index: int
    content: str
    similarity: float


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

    # pgvector cosine_distance is in [0, 2]; similarity = 1 - distance.
    distance = Chunk.embedding.cosine_distance(query_vec).label("distance")
    stmt = (
        select(Chunk, Document, distance)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(k)
    )
    if document_ids:
        stmt = stmt.where(Chunk.document_id.in_(document_ids))

    rows = (await session.execute(stmt)).all()

    results: list[RetrievedChunk] = []
    for chunk, document, dist in rows:
        similarity = 1.0 - float(dist)
        if similarity < floor:
            continue
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                source=document.source,
                title=document.title,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                similarity=round(similarity, 4),
            )
        )
    return results
