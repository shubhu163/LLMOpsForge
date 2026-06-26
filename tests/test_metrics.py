"""Tests for the deterministic evaluation metrics."""

from __future__ import annotations

from app.config import EvalThresholds
from app.evals.metrics import (
    EvalTask,
    answer_correctness,
    citation_correctness,
    evaluate_example,
    grounding,
    hallucination,
    json_validity,
    retrieval_relevance,
)
from app.providers.base import RetrievedContext
from app.rag.citations import Citation
from app.rag.generator import GenerationResult


def _result(answer: str, *, cited=("refund_policy.md",), retrieved=("refund_policy.md",)):
    contexts = [
        RetrievedContext(f"{d}::chunk_0", d, "Refunds are available within 30 days of purchase.")
        for d in retrieved
    ]
    citations = [Citation(f"{d}::chunk_0", d, 1.0) for d in cited]
    return GenerationResult(
        answer=answer,
        citations=citations,
        retrieved_contexts=contexts,
        latency_ms=1.0,
        estimated_tokens=10,
        estimated_cost=0.0,
        model_name="mock-small",
        prompt_template_id="prompt_v1",
    )


def test_answer_correctness_keywords_and_exact():
    task = EvalTask(
        id="t", question="q", expected_answer="30 days", answer_keywords=["30 days", "refund"]
    )
    score, detail = answer_correctness(task, "Refunds are available within 30 days.")
    assert score == 1.0
    assert "30 days" in detail["matched_keywords"]


def test_answer_correctness_partial():
    task = EvalTask(id="t", question="q", answer_keywords=["30 days", "missing-term"])
    score, _ = answer_correctness(task, "Refunds within 30 days.")
    assert 0.0 < score < 1.0


def test_citation_correctness():
    task = EvalTask(id="t", question="q", expected_sources=["refund_policy.md"])
    score, _ = citation_correctness(task, _result("x", cited=("refund_policy.md",)))
    assert score == 1.0
    score2, _ = citation_correctness(task, _result("x", cited=("pricing_policy.md",)))
    assert score2 == 0.0


def test_grounding_high_when_answer_from_context():
    score, _ = grounding(_result("Refunds are available within 30 days"), "Refunds within 30 days")
    assert score > 0.8


def test_retrieval_relevance():
    task = EvalTask(id="t", question="q", expected_sources=["refund_policy.md"])
    score, _ = retrieval_relevance(task, _result("x", retrieved=("refund_policy.md",)))
    assert score == 1.0


def test_hallucination_flag_on_forbidden_claim():
    task = EvalTask(id="t", question="q", forbidden_claims=["90 days"])
    flag, detail = hallucination(task, "Refunds within 90 days", grounding_score=1.0)
    assert flag is True
    assert "90 days" in detail["forbidden_hits"]


def test_json_validity_pass_and_fail():
    schema = {
        "type": "object",
        "required": ["answer"],
        "properties": {"answer": {"type": "string"}},
    }
    task = EvalTask(id="t", question="q", requires_json=True, expected_json_schema=schema)
    assert json_validity(task, '{"answer": "30 days"}')[0] == "pass"
    assert json_validity(task, "not json")[0] == "fail"
    assert json_validity(task, '{"wrong": 1}')[0] == "fail"


def test_json_validity_na_when_not_required():
    task = EvalTask(id="t", question="q", requires_json=False)
    assert json_validity(task, "anything")[0] == "n/a"


def test_evaluate_example_passes():
    task = EvalTask(
        id="t",
        question="What is the refund period?",
        expected_answer="30 days",
        expected_sources=["refund_policy.md"],
        answer_keywords=["30 days"],
    )
    metric = evaluate_example(
        task, _result("Refunds are available within 30 days of purchase."), EvalThresholds()
    )
    assert metric.passed is True
    assert metric.hallucination_flag is False
    assert metric.json_validity == "n/a"
