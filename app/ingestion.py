"""Ingestion pipeline: text -> chunks -> embeddings -> persisted rows."""

from __future__ import annotations

import hashlib

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import chunk_text
from app.config import settings
from app.embeddings import get_embedding_provider
from app.models import Chunk, Document


def content_digest(content: str) -> str:
    """Stable fingerprint of a document's content (whitespace-normalised)."""
    return hashlib.sha256(content.strip().encode("utf-8")).hexdigest()


async def ingest_document(
    session: AsyncSession,
    *,
    content: str,
    source: str,
    title: str | None = None,
    metadata: dict | None = None,
) -> tuple[Document, int, bool]:
    """Chunk, embed and store a document.

    Returns (document, chunk_count, created). If a document with identical
    content was already ingested, the existing one is returned untouched with
    created=False, so re-ingesting the same file costs no embedding calls and
    creates no duplicate rows.

    The document and all its chunks are written in a single transaction (the
    caller's session), so a failure mid-way leaves nothing partially ingested.
    """
    digest = content_digest(content)
    existing = await session.scalar(
        select(Document).where(Document.content_hash == digest)
    )
    if existing is not None:
        count = await session.scalar(
            select(func.count(Chunk.id)).where(Chunk.document_id == existing.id)
        )
        return existing, count or 0, False

    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Document produced no chunks (empty after normalisation)")

    provider = get_embedding_provider()

    # Embed in batches to bound memory and play nicely with hosted rate limits.
    texts = [c.content for c in chunks]
    embeddings: list[list[float]] = []
    batch = settings.embedding_batch_size
    for start in range(0, len(texts), batch):
        embeddings.extend(await provider.embed_documents(texts[start : start + batch]))

    document = Document(
        source=source,
        title=title,
        content_hash=digest,
        doc_metadata=metadata or {},
    )
    document.chunks = [
        Chunk(
            chunk_index=chunk.index,
            content=chunk.content,
            char_count=chunk.char_count,
            embedding=vector,
        )
        for chunk, vector in zip(chunks, embeddings, strict=True)
    ]

    session.add(document)
    await session.flush()  # assign PKs without ending the transaction
    return document, len(document.chunks), True
