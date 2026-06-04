"""Centralised application configuration.

All settings are read from environment variables (or a local ``.env`` file)
through pydantic-settings, so the same image runs in dev, CI and prod with no
code change. Import the singleton ``settings`` everywhere instead of reaching
for ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RAG Assistant"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"

    # Database
    # Async SQLAlchemy URL. Defaults match the bundled docker-compose service.
    database_url: str = Field(
        default="postgresql+asyncpg://rag:rag@localhost:5432/rag",
    )
    db_pool_size: int = 10
    db_max_overflow: int = 20

    # Embeddings
    # "fastembed" runs locally (ONNX, no API key, ~50MB model).
    # "voyage" calls Voyage AI (Anthropic's recommended embeddings provider).
    embedding_provider: Literal["fastembed", "voyage"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # MUST match the model's output dimension. bge-small = 384, voyage-3 = 1024.
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    voyage_api_key: str | None = None

    # Chunking
    chunk_size: int = 1000          # target characters per chunk
    chunk_overlap: int = 150        # characters shared between adjacent chunks

    # Retrieval
    retrieval_top_k: int = 5
    # Cosine similarity floor [0, 1]; chunks below this are dropped.
    retrieval_min_similarity: float = 0.2

    # LLM
    anthropic_api_key: str | None = None
    llm_model: str = "claude-opus-4-8"
    llm_max_tokens: int = 4096
    # low | medium | high | max, controls thinking depth and token spend.
    llm_effort: Literal["low", "medium", "high", "max"] = "high"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
