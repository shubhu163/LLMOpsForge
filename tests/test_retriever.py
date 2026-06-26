"""Tests for embeddings + the vector retriever (offline hashing backend)."""

from __future__ import annotations

from app.rag.embeddings import HashingEmbedder
from app.rag.retriever import IndexedChunk, VectorRetriever


def _retriever() -> VectorRetriever:
    chunks = [
        IndexedChunk("refund.md::chunk_0", "refund.md", "Refunds are available within 30 days."),
        IndexedChunk(
            "pricing.md::chunk_0", "pricing.md", "The Pro plan costs 49 dollars per month."
        ),
        IndexedChunk(
            "security.md::chunk_0", "security.md", "Data is encrypted at rest using AES-256."
        ),
    ]
    r = VectorRetriever(HashingEmbedder())
    r.build(chunks)
    return r


def test_returns_top_k():
    r = _retriever()
    results = r.query("refund window", top_k=2)
    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_retrieves_relevant_document_first():
    r = _retriever()
    top = r.query("How much does the Pro plan cost?", top_k=1)
    assert top[0].document_name == "pricing.md"


def test_empty_index_returns_empty():
    r = VectorRetriever(HashingEmbedder())
    r.build([])
    assert r.query("anything", top_k=3) == []


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder()
    a = e.embed(["hello world"])
    b = e.embed(["hello world"])
    assert (a == b).all()
