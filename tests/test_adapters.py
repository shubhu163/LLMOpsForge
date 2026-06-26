"""Tests for external RAG adapters and assessing them through the eval runner."""

from __future__ import annotations

import json

from app.config import load_eval_config
from app.evals.adapters import FunctionRagAdapter, _coerce_citations, _coerce_contexts
from app.evals.runner import EvalRunner


def test_coerce_contexts_from_strings_and_dicts():
    ctx = _coerce_contexts(
        ["plain text", {"document_name": "a.md", "chunk_id": "a::0", "text": "x"}]
    )
    assert len(ctx) == 2
    assert ctx[1].document_name == "a.md"
    assert ctx[1].chunk_id == "a::0"


def test_coerce_citations_defaults_to_contexts():
    ctx = _coerce_contexts([{"document_name": "a.md", "chunk_id": "a::0", "text": "x"}])
    cites = _coerce_citations(None, ctx)
    assert cites[0].document_name == "a.md"


def test_function_adapter_returns_generation_result():
    adapter = FunctionRagAdapter(
        lambda q: {
            "answer": "Refunds are available within 30 days.",
            "retrieved_contexts": [
                {
                    "document_name": "refund_policy.md",
                    "chunk_id": "refund_policy.md::0",
                    "text": "Customers may request a full refund within 30 days.",
                }
            ],
        },
        name="my-rag",
    )
    result = adapter.query("How many days for a refund?")
    assert "30 days" in result.answer
    assert result.model_name == "my-rag"
    assert result.retrieved_contexts[0].document_name == "refund_policy.md"
    assert result.citations  # defaulted from contexts
    assert result.latency_ms >= 0


def test_assess_external_rag_through_runner(session, tmp_path):
    # A tiny labelled dataset.
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "r1",
                "question": "How many days for a refund?",
                "expected_answer": "30 days",
                "expected_sources": ["refund_policy.md"],
                "answer_keywords": ["30 days"],
                "requires_json": False,
                "difficulty": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    # An "external" RAG that answers correctly and grounds in context.
    adapter = FunctionRagAdapter(
        lambda q: {
            "answer": "Customers may request a full refund within 30 days.",
            "retrieved_contexts": [
                {
                    "document_name": "refund_policy.md",
                    "chunk_id": "refund_policy.md::0",
                    "text": "Customers may request a full refund within 30 days.",
                }
            ],
        }
    )
    cfg = load_eval_config("configs/default.yaml")
    run = EvalRunner(session, pipeline=adapter).run(
        dataset_path=str(dataset), config=cfg, name="ext"
    )
    assert run.summary["total_examples"] == 1
    assert run.summary["passed"] == 1
