"""Tests for the chunker."""

from __future__ import annotations

import pytest

from app.rag.chunker import chunk_document


def test_chunk_ids_and_indices():
    text = "\n\n".join(f"Paragraph number {i} with some text content." for i in range(10))
    chunks = chunk_document("doc.md", text, chunk_size=120, chunk_overlap=20)
    assert len(chunks) > 1
    for i, c in enumerate(chunks):
        assert c.chunk_id == f"doc.md::chunk_{i}"
        assert c.chunk_index == i
        assert c.document_name == "doc.md"
        assert c.text


def test_respects_chunk_size_roughly():
    text = "word " * 1000
    chunks = chunk_document("big.md", text, chunk_size=200, chunk_overlap=40)
    # Hard-split pieces must not exceed the configured size.
    assert all(len(c.text) <= 200 for c in chunks)
    assert len(chunks) > 1


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_document("d.md", "text", chunk_size=100, chunk_overlap=100)


def test_small_document_single_chunk():
    chunks = chunk_document("small.md", "Short content.", chunk_size=600, chunk_overlap=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Short content."
