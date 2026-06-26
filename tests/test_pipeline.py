"""Tests for ingestion and the end-to-end RAG query path."""

from __future__ import annotations

from app.storage import repository


def test_ingest_persists_documents_and_chunks(pipeline, docs_dir, session):
    result = pipeline.ingest(docs_dir, chunk_size=600, chunk_overlap=100)
    assert result["documents"] >= 5
    assert result["chunks"] > result["documents"]
    assert repository.count_documents(session) == result["documents"]
    assert repository.count_chunks(session) == result["chunks"]


def test_query_returns_grounded_answer_with_citations(ingested_pipeline):
    result = ingested_pipeline.query(
        "How many days do customers have to request a full refund?",
        top_k=4,
        prompt_template_id="prompt_v2",
        model_config_id="mock-small",
    )
    assert "30 days" in result.answer
    assert result.citations
    assert any(c.document_name == "refund_policy.md" for c in result.citations)
    assert result.estimated_tokens > 0


def test_reingestion_replaces_chunks(pipeline, docs_dir, session):
    pipeline.ingest(docs_dir)
    first = repository.count_chunks(session)
    pipeline.ingest(docs_dir)  # same docs again
    assert repository.count_chunks(session) == first  # no duplication


def test_query_persists_rag_query(ingested_pipeline, session):
    from app.storage import models

    ingested_pipeline.query("How much does the Pro plan cost?", persist=True)
    rows = session.query(models.RagQuery).all()
    assert len(rows) == 1
    assert rows[0].question
