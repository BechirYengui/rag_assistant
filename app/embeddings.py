"""Pluggable embedding providers.

Anthropic does not offer an embeddings endpoint, so embeddings come from a
dedicated model. Two interchangeable backends are provided:

* ``fastembed``, local ONNX inference, no API key, default. (~50MB model,
  downloaded once and cached.)
* ``voyage``, Voyage AI, Anthropic's recommended hosted provider.

Both expose the same small async interface. Heavy imports are deferred to
construction time so the rest of the app (and the chunking tests) load without
the optional dependencies installed.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from app.config import settings


@runtime_checkable
class EmbeddingProvider(Protocol):
    dim: int

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class FastEmbedProvider:
    """Local embeddings via fastembed (ONNX runtime, CPU-friendly)."""

    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding  # deferred heavy import

        self._model = TextEmbedding(model_name=model_name)
        self.dim = dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        # fastembed yields numpy arrays lazily; materialise to plain lists.
        return [vec.tolist() for vec in self._model.embed(texts)]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts)

    async def embed_query(self, text: str) -> list[float]:
        [vec] = await asyncio.to_thread(self._encode, [text])
        return vec


class VoyageProvider:
    """Hosted embeddings via Voyage AI."""

    def __init__(self, model_name: str, dim: int, api_key: str) -> None:
        import voyageai  # deferred import

        self._client = voyageai.Client(api_key=api_key)
        self._model = model_name
        self.dim = dim

    def _encode(self, texts: list[str], input_type: str) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type=input_type)
        return result.embeddings

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, texts, "document")

    async def embed_query(self, text: str) -> list[float]:
        [vec] = await asyncio.to_thread(self._encode, [text], "query")
        return vec


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return the process-wide singleton provider, building it on first use."""
    global _provider
    if _provider is not None:
        return _provider

    if settings.embedding_provider == "voyage":
        if not settings.voyage_api_key:
            raise RuntimeError("VOYAGE_API_KEY is required for the voyage provider")
        _provider = VoyageProvider(
            settings.embedding_model, settings.embedding_dim, settings.voyage_api_key
        )
    else:
        _provider = FastEmbedProvider(settings.embedding_model, settings.embedding_dim)

    if _provider.dim != settings.embedding_dim:
        raise RuntimeError(
            f"Configured embedding_dim={settings.embedding_dim} does not match "
            f"provider dim={_provider.dim}"
        )
    return _provider
