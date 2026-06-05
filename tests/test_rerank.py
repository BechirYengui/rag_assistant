"""Tests for MMR reranking and idempotent-ingestion hashing."""

from __future__ import annotations

import uuid

from app.ingestion import content_digest
from app.rerank import _cosine, mmr_select
from app.retrieval import RetrievedChunk


def _chunk(emb: list[float], sim: float, text: str = "x") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source="s",
        title="t",
        chunk_index=0,
        content=text,
        similarity=sim,
        embedding=emb,
    )


def test_cosine_basic():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0  # zero vector is safe


def test_mmr_prefers_diversity_over_near_duplicate():
    # c1 and c2 are near-identical; c3 is diverse but slightly less relevant.
    c1 = _chunk([1.0, 0.0, 0.0], 0.90, "alpha")
    c2 = _chunk([1.0, 0.0, 0.01], 0.88, "alpha-bis")  # redundant with c1
    c3 = _chunk([0.0, 1.0, 0.0], 0.70, "beta")  # diverse

    selected = mmr_select([c1, c2, c3], k=2, lambda_=0.5)
    contents = {c.content for c in selected}

    assert "alpha" in contents          # most relevant is kept
    assert "beta" in contents           # diversity beats the near-duplicate
    assert "alpha-bis" not in contents


def test_mmr_returns_all_when_pool_not_larger_than_k():
    chunks = [_chunk([1.0, 0.0], 0.9), _chunk([0.0, 1.0], 0.8)]
    assert mmr_select(chunks, k=5, lambda_=0.5) == chunks


def test_mmr_pure_relevance_keeps_ranking():
    c1 = _chunk([1.0, 0.0], 0.9, "a")
    c2 = _chunk([1.0, 0.0], 0.8, "b")
    c3 = _chunk([1.0, 0.0], 0.7, "c")
    # lambda_=1.0 -> ignore diversity entirely, follow similarity order.
    selected = mmr_select([c1, c2, c3], k=2, lambda_=1.0)
    assert [c.content for c in selected] == ["a", "b"]


def test_content_digest_is_stable_and_whitespace_insensitive():
    a = content_digest("Hello world")
    b = content_digest("  Hello world  ")  # surrounding whitespace ignored
    c = content_digest("Hello mars")
    assert a == b
    assert a != c
    assert len(a) == 64  # sha-256 hex
