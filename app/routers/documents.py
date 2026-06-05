"""Document ingestion and management endpoints."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.ingestion import ingest_document
from app.models import Chunk, Document
from app.schemas import DocumentResponse, IngestResponse, IngestTextRequest

router = APIRouter(prefix="/documents", tags=["documents"])


async def _to_document_response(
    session: AsyncSession, document: Document
) -> DocumentResponse:
    count = await session.scalar(
        select(func.count(Chunk.id)).where(Chunk.document_id == document.id)
    )
    return DocumentResponse(
        id=document.id,
        source=document.source,
        title=document.title,
        metadata=document.doc_metadata,
        chunk_count=count or 0,
        created_at=document.created_at,
    )


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_text(
    payload: IngestTextRequest, session: AsyncSession = Depends(get_session)
) -> IngestResponse:
    """Ingest a raw-text document into the knowledge base."""
    try:
        document, created, is_new = await ingest_document(
            session,
            content=payload.content,
            source=payload.source,
            title=payload.title,
            metadata=payload.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return IngestResponse(
        document=await _to_document_response(session, document),
        chunks_created=created if is_new else 0,
        deduplicated=not is_new,
    )


@router.post(
    "/upload", response_model=IngestResponse, status_code=status.HTTP_201_CREATED
)
async def ingest_file(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    """Ingest an uploaded file (.txt, .md, or .pdf)."""
    raw = await file.read()
    name = file.filename or "upload"

    if name.lower().endswith(".pdf"):
        content = _extract_pdf_text(raw)
    else:
        content = raw.decode("utf-8", errors="replace")

    if not content.strip():
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "File contained no extractable text"
        )

    try:
        document, created, is_new = await ingest_document(
            session,
            content=content,
            source=name,
            title=title or name,
            metadata={"content_type": file.content_type},
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return IngestResponse(
        document=await _to_document_response(session, document),
        chunks_created=created if is_new else 0,
        deduplicated=not is_new,
    )


def _extract_pdf_text(raw: bytes) -> str:
    import io

    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


@router.get("", response_model=list[DocumentResponse])
async def list_documents(
    limit: int = 50,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
) -> list[DocumentResponse]:
    documents = (
        await session.scalars(
            select(Document)
            .order_by(Document.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
    ).all()
    return [await _to_document_response(session, d) for d in documents]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, session: AsyncSession = Depends(get_session)
) -> None:
    result = await session.execute(
        delete(Document).where(Document.id == document_id)
    )
    if result.rowcount == 0:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
