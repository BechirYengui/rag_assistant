"""Tests for the token-optimisation logic: context budgeting and answer cache."""

from __future__ import annotations

import uuid

from app.cache import AnswerCache
from app.retrieval import RetrievedChunk, fit_to_budget
from app.tokens import estimate_tokens


def _chunk(text: str, sim: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        source="s",
        title="t",
        chunk_index=0,
        content=text,
        similarity=sim,
    )


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a") == 1
    assert estimate_tokens("x" * 400) >= estimate_tokens("x" * 40)


def test_fit_to_budget_stops_at_the_ceiling():
    chunks = [_chunk("x" * 400, 0.9) for _ in range(10)]  # ~100 tokens each
    kept = fit_to_budget(chunks, max_tokens=250)
    assert 0 < len(kept) < len(chunks)


def test_fit_to_budget_keeps_top_chunk_even_if_over_budget():
    kept = fit_to_budget([_chunk("x" * 10_000, 0.95)], max_tokens=10)
    assert len(kept) == 1  # never answer with empty context


def test_fit_to_budget_preserves_similarity_order():
    chunks = [_chunk("x" * 40, 0.9), _chunk("y" * 40, 0.8), _chunk("z" * 40, 0.7)]
    kept = fit_to_budget(chunks, max_tokens=1000)
    assert [c.similarity for c in kept] == [0.9, 0.8, 0.7]


def test_answer_cache_round_trip_and_lru_eviction():
    cache = AnswerCache(max_size=2)
    k1, k2, k3 = "k1", "k2", "k3"
    cache.set(k1, "a1")
    cache.set(k2, "a2")
    assert cache.get(k1) == "a1"  # touch k1 so k2 is now least-recently used
    cache.set(k3, "a3")  # evicts k2
    assert cache.get(k2) is None
    assert cache.get(k1) == "a1"
    assert cache.get(k3) == "a3"


def test_answer_cache_disabled_when_size_zero():
    cache = AnswerCache(max_size=0)
    cache.set("k", "v")
    assert cache.get("k") is None


def test_cache_key_depends_on_question_and_chunks():
    a = AnswerCache.make_key("question", ["c1", "c2"])
    b = AnswerCache.make_key("question", ["c1", "c3"])
    c = AnswerCache.make_key("question", ["c1", "c2"])
    assert a == c
    assert a != b
