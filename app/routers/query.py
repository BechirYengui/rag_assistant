"""Query endpoints: synchronous JSON answer and streaming SSE answer."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache import answer_cache
from app.db import get_session
from app.llm import ensure_available, stream_answer
from app.rag import _cache_key, _to_sources, answer_question, retrieve_context
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
    events carry answer deltas; a final ``done`` event closes the stream and
    reports whether the answer came from the cache (``cached``).
    """
    chunks = await retrieve_context(
        session,
        payload.question,
        top_k=payload.top_k,
        min_similarity=payload.min_similarity,
        document_ids=payload.document_ids,
    )
    cache_key = _cache_key(payload.question, chunks)
    cached_answer = answer_cache.get(cache_key)
    # Surface a missing API key as a clean 503 before the stream opens, but only
    # when we will actually call Claude: a cache hit needs no key. Once the SSE
    # response has started, an error can no longer change the status code.
    if chunks and cached_answer is None:
        ensure_available()

    async def event_stream():
        sources = [s.model_dump(mode="json") for s in _to_sources(chunks)]
        yield _sse("sources", {"sources": sources})

        if cached_answer is not None:
            # Identical question + context already answered: replay it verbatim,
            # zero Claude tokens, consistent with the synchronous /query path.
            yield _sse("token", {"text": cached_answer})
            yield _sse("done", {"cached": True})
            return

        parts: list[str] = []
        async for token in stream_answer(payload.question, chunks):
            parts.append(token)
            yield _sse("token", {"text": token})
        answer_cache.set(cache_key, "".join(parts))
        yield _sse("done", {"cached": False})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
