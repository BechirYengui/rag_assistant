"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.config import settings
from app.db import dispose_db, engine, init_db
from app.llm import LLMConfigurationError
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


@app.exception_handler(LLMConfigurationError)
async def _llm_configuration_error_handler(
    request: Request, exc: LLMConfigurationError
) -> JSONResponse:
    """Turn a missing/invalid LLM configuration into a clean 503.

    Without this, a missing ``ANTHROPIC_API_KEY`` surfaces as an opaque 500.
    """
    logger.warning("LLM unavailable: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
    )


async def _database_reachable() -> bool:
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness + active configuration. Always 200 while the process is up."""
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


@app.get("/health/ready", tags=["meta"])
async def readiness(response: Response) -> dict:
    """Readiness probe: 200 only when the database is reachable, else 503.

    Orchestrators (Docker healthcheck, Kubernetes) gate traffic on the HTTP
    status, so a down database must not report ready.
    """
    db_ok = await _database_reachable()
    if not db_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ready" if db_ok else "not ready",
        "database": "ok" if db_ok else "unreachable",
    }
