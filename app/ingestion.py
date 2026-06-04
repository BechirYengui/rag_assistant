"""Ingestion pipeline: text -> chunks -> embeddings -> persisted rows."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.chunking import chunk_text
from app.config import settings
from app.embeddings import get_embedding_provider
from app.models import Chunk, Document


async def ingest_document(
    session: AsyncSession,
    *,
    content: str,
    source: str,
    title: str | None = None,
    metadata: dict | None = None,
) -> tuple[Document, int]:
    """Chunk, embed and store a document. Returns (document, chunk_count).

    The document and all its chunks are written in a single transaction (the
    caller's session), so a failure mid-way leaves nothing partially ingested.
    """
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
    return document, len(document.chunks)
