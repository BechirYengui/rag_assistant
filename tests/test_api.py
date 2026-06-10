"""API + pipeline tests that don't require Postgres or a live Claude key.

The database session and the retrieval/LLM calls are replaced with fakes, so
these exercise routing, request validation, citation mapping and response
shaping in isolation.
"""

from __future__ import annotations

import uuid

import httpx
import pytest

from app.llm import build_context
from app.main import app
from app.rag import _to_sources
from app.retrieval import RetrievedChunk


def _chunk(i: int, text: str, sim: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source=f"doc-{i}.md",
        title=f"Doc {i}",
        chunk_index=i,
        content=text,
        similarity=sim,
    )


def test_build_context_numbers_sources_from_one():
    ctx = build_context([_chunk(0, "Alpha fact.", 0.9), _chunk(1, "Beta fact.", 0.8)])
    assert "[1] (source: Doc 0)" in ctx
    assert "[2] (source: Doc 1)" in ctx
    assert "Alpha fact." in ctx and "Beta fact." in ctx


def test_to_sources_assigns_contiguous_citations():
    sources = _to_sources([_chunk(0, "a", 0.9), _chunk(1, "b", 0.7)])
    assert [s.citation for s in sources] == [1, 2]
    assert sources[0].similarity == 0.9


@pytest.mark.asyncio
async def test_health_endpoint():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["llm_model"]


@pytest.mark.asyncio
async def test_query_validation_rejects_empty_question():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/query", json={"question": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_readiness_ok_when_db_reachable(monkeypatch):
    async def reachable():
        return True

    monkeypatch.setattr("app.main._database_reachable", reachable)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_readiness_503_when_db_unreachable(monkeypatch):
    async def unreachable():
        return False

    monkeypatch.setattr("app.main._database_reachable", unreachable)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["database"] == "unreachable"


@pytest.mark.asyncio
async def test_stream_serves_cached_answer_without_calling_claude(monkeypatch):
    # A cache hit on /query/stream must replay the stored answer and never touch
    # the LLM, so a repeated streamed question costs zero Claude tokens.
    from app.cache import answer_cache
    from app.rag import _cache_key

    chunks = [_chunk(0, "Some relevant fact.", 0.9)]

    async def fake_retrieve_context(*args, **kwargs):
        return chunks

    async def exploding_stream(*args, **kwargs):
        raise AssertionError("stream_answer must not be called on a cache hit")
        yield  # pragma: no cover - makes this an async generator

    monkeypatch.setattr("app.routers.query.retrieve_context", fake_retrieve_context)
    monkeypatch.setattr("app.routers.query.stream_answer", exploding_stream)
    answer_cache.set(_cache_key("Cached question?", chunks), "A cached answer [1].")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/query/stream", json={"question": "Cached question?"})

    assert resp.status_code == 200
    body = resp.text
    assert "A cached answer [1]." in body
    assert '"cached": true' in body


@pytest.mark.asyncio
async def test_stream_caches_generated_answer(monkeypatch):
    # On a cache miss, the streamed tokens must be reassembled and stored so the
    # next identical streamed (or synchronous) query is a free cache hit.
    from app.cache import answer_cache
    from app.rag import _cache_key

    chunks = [_chunk(0, "Some relevant fact.", 0.9)]
    key = _cache_key("Fresh question?", chunks)
    answer_cache._store.pop(key, None)  # ensure a miss

    async def fake_retrieve_context(*args, **kwargs):
        return chunks

    async def fake_stream(*args, **kwargs):
        for token in ("Hello ", "world", " [1]."):
            yield token

    monkeypatch.setattr("app.routers.query.retrieve_context", fake_retrieve_context)
    monkeypatch.setattr("app.routers.query.stream_answer", fake_stream)
    monkeypatch.setattr("app.routers.query.ensure_available", lambda: None)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/query/stream", json={"question": "Fresh question?"})

    assert resp.status_code == 200
    assert '"cached": false' in resp.text
    assert answer_cache.get(key) == "Hello world [1]."


@pytest.mark.asyncio
async def test_query_returns_503_when_api_key_missing(monkeypatch):
    # Retrieval succeeds and yields context, but no Claude key is configured:
    # the opaque 500 must become a clean 503 with an actionable message.
    import app.llm as llm

    async def fake_retrieve_context(*args, **kwargs):
        return [_chunk(0, "Some relevant fact.", 0.9)]

    monkeypatch.setattr("app.rag.retrieve_context", fake_retrieve_context)
    monkeypatch.setattr(llm.settings, "anthropic_api_key", None, raising=False)
    monkeypatch.setattr(llm, "_client", None, raising=False)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        resp = await client.post("/query", json={"question": "What is up?"})
    assert resp.status_code == 503
    assert "ANTHROPIC_API_KEY" in resp.json()["detail"]
