"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text

from app.config import settings
from app.db import dispose_db, engine, init_db
from app.routers import documents, query

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("rag")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database (%s)…", settings.environment)
    await init_db()
    yield
    await dispose_db()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    summary="Retrieval-augmented Q&A over a document corpus, powered by Claude.",
    lifespan=lifespan,
)

app.include_router(documents.router)
app.include_router(query.router)


async def _database_reachable() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/health", tags=["meta"])
async def health() -> dict:
    return {
        "status": "ok",
        "database": "ok" if await _database_reachable() else "unreachable",
        "embedding_provider": settings.embedding_provider,
        "embedding_dim": settings.embedding_dim,
        "llm_model": settings.llm_model,
        "llm_effort": settings.llm_effort,
        "llm_thinking": settings.llm_thinking,
        "rerank": settings.rerank,
        "max_context_tokens": settings.max_context_tokens,
        "answer_cache_size": settings.answer_cache_size,
    }
