"""Maximal Marginal Relevance (MMR) reranking.

Pure nearest-neighbour retrieval tends to return chunks that are all very
similar to each other (the same passage phrased three ways), which wastes
context tokens and narrows coverage. MMR reranks a candidate pool so each
selected chunk is relevant to the query but also different from the chunks
already selected.

The score for a candidate c, given the already-selected set S, is:

    score(c) = lambda * relevance(c) - (1 - lambda) * max_{s in S} sim(c, s)

where relevance(c) is the cosine similarity to the query (already computed by
the vector search) and sim(c, s) is the cosine similarity between two chunk
embeddings. Pure Python, no numpy dependency, fast for the small pools used
here.
"""

from __future__ import annotations

from math import sqrt

from app.retrieval import RetrievedChunk


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=False):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (sqrt(na) * sqrt(nb))


def mmr_select(
    candidates: list[RetrievedChunk], k: int, lambda_: float
) -> list[RetrievedChunk]:
    """Select up to ``k`` chunks balancing relevance and diversity.

    Candidates must carry their ``embedding``. They are returned in selection
    order (most relevant first, then progressively more diverse).
    """
    if k <= 0 or not candidates:
        return []
    if len(candidates) <= k:
        return candidates

    remaining = list(candidates)
    selected: list[RetrievedChunk] = []

    while remaining and len(selected) < k:
        best: RetrievedChunk | None = None
        best_score = float("-inf")
        for cand in remaining:
            if cand.embedding is None:
                redundancy = 0.0
            else:
                redundancy = max(
                    (
                        _cosine(cand.embedding, s.embedding)
                        for s in selected
                        if s.embedding is not None
                    ),
                    default=0.0,
                )
            score = lambda_ * cand.similarity - (1.0 - lambda_) * redundancy
            if score > best_score:
                best_score = score
                best = cand
        assert best is not None
        selected.append(best)
        remaining.remove(best)

    return selected
