"""Query endpoints: synchronous JSON answer and streaming SSE answer."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.llm import stream_answer
from app.rag import _to_sources, answer_question, retrieve_context
from app.schemas import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(
    payload: QueryRequest, session: AsyncSession = Depends(get_session)
) -> QueryResponse:
    """Answer a natural-language question with cited sources."""
    return await answer_question(
        session,
        payload.question,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity,
        document_ids=payload.document_ids,
    )


@router.post("/stream")
async def query_stream(
    payload: QueryRequest, session: AsyncSession = Depends(get_session)
) -> StreamingResponse:
    """Stream the answer token-by-token as Server-Sent Events.

    The first event carries the retrieved ``sources``; subsequent ``token``
    events carry answer deltas; a final ``done`` event closes the stream.
    """
    chunks = await retrieve_context(
        session,
        payload.question,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity,
        document_ids=payload.document_ids,
    )

    async def event_stream():
        sources = [s.model_dump(mode="json") for s in _to_sources(chunks)]
        yield _sse("sources", {"sources": sources})
        async for token in stream_answer(payload.question, chunks):
            yield _sse("token", {"text": token})
        yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
