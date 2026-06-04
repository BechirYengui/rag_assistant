"""Document chunking.

A recursive, separator-aware splitter: it tries to break on the largest
natural boundary that fits (paragraph -> line -> sentence -> word), then packs
the pieces into ~``chunk_size`` windows with a fixed character overlap so
context isn't lost at the seams. Pure-Python, zero dependencies, so it is
trivially unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings

# Ordered from coarsest to finest. The splitter descends this list only when a
# fragment is still larger than the target window.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


@dataclass(frozen=True)
class Chunk:
    index: int
    content: str

    @property
    def char_count(self) -> int:
        return len(self.content)


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Break ``text`` into fragments no larger than ``size`` where possible."""
    if len(text) <= size:
        return [text] if text else []

    if not separators:
        # No separator left: hard-split on the size boundary.
        return [text[i : i + size] for i in range(0, len(text), size)]

    sep, *rest = separators
    parts = text.split(sep)
    fragments: list[str] = []
    for i, part in enumerate(parts):
        # Re-attach the separator we split on (except after the last part).
        piece = part + sep if i < len(parts) - 1 else part
        if len(piece) <= size:
            if piece:
                fragments.append(piece)
        else:
            fragments.extend(_split_recursive(piece, size, rest))
    return fragments


def chunk_text(
    text: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    The fragments produced by the recursive splitter are greedily packed into
    windows of at most ``chunk_size`` characters. Adjacent windows share the
    last ``chunk_overlap`` characters of the previous window.
    """
    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalised = re.sub(r"[ \t]+\n", "\n", text.strip())
    if not normalised:
        return []

    fragments = _split_recursive(normalised, size, _SEPARATORS)

    chunks: list[Chunk] = []
    buffer = ""
    for fragment in fragments:
        if len(buffer) + len(fragment) <= size:
            buffer += fragment
            continue
        if buffer:
            chunks.append(buffer.strip())
        # Seed the next buffer with the tail of the current one for continuity.
        tail = buffer[-overlap:] if overlap and buffer else ""
        buffer = tail + fragment

    if buffer.strip():
        chunks.append(buffer.strip())

    return [Chunk(index=i, content=c) for i, c in enumerate(chunks) if c]
