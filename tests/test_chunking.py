"""Unit tests for the chunker, pure Python, no external services needed."""

from __future__ import annotations

import pytest

from app.chunking import chunk_text


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\t ") == []


def test_short_text_is_a_single_chunk():
    chunks = chunk_text("Hello world.", chunk_size=1000, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world."
    assert chunks[0].index == 0


def test_long_text_is_split_and_indexed_contiguously():
    text = ". ".join(f"Sentence number {i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert all(c.char_count <= 240 for c in chunks)  # size + overlap headroom


def test_overlap_preserves_continuity():
    text = "".join(f"word{i} " for i in range(500))
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
    # The tail of one chunk should reappear at the head of the next.
    overlaps = sum(
        1
        for a, b in zip(chunks, chunks[1:], strict=False)
        if a.content[-20:].strip().split()[-1] in b.content[:80]
    )
    assert overlaps >= 1


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, chunk_overlap=100)


def test_hard_split_when_no_separators():
    text = "x" * 1000  # no whitespace or punctuation to break on
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 10
    assert "".join(c.content for c in chunks) == text
