"""Process-local LRU cache for generated answers.

Keyed by the question, the exact set of retrieved chunks, and the model config.
A cache hit returns the stored answer without calling Claude at all, so a
repeated question costs zero tokens. This is a single-process cache; behind
several workers, put a shared store (Redis) behind the same interface.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict

from app.config import settings


class AnswerCache:
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, str] = OrderedDict()

    @staticmethod
    def make_key(question: str, chunk_ids: list[str]) -> str:
        raw = json.dumps(
            {
                "q": question.strip(),
                "chunks": chunk_ids,
                "model": settings.llm_model,
                "effort": settings.llm_effort,
                "thinking": settings.llm_thinking,
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)  # mark as most-recently used
        return self._store[key]

    def set(self, key: str, answer: str) -> None:
        if self._max_size <= 0:
            return
        self._store[key] = answer
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)  # evict least-recently used


answer_cache = AnswerCache(settings.answer_cache_size)
