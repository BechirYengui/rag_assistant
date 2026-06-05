"""Cheap, dependency-free token estimation.

A heuristic (characters / chars_per_token) is used instead of the real
tokenizer or the count_tokens API on purpose: budgeting the retrieval context
must not itself cost a network round trip or extra tokens. The estimate only
needs to be good enough to keep the context under a safe ceiling.
"""

from __future__ import annotations

from math import ceil

from app.config import settings


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text) / settings.chars_per_token))
