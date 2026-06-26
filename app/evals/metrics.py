"""Deterministic, explainable evaluation metrics.

Every metric here is rule-based and reproducible — no LLM calls, no randomness.
This makes the eval suite fast, free, and suitable for CI gating. An
:class:`LLMJudge` protocol is defined so an LLM-as-judge scorer can be layered on
later without changing the runner, but it is entirely optional.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from jsonschema import Draft7Validator
from pydantic import BaseModel, Field

from app.config import EvalThresholds
from app.rag.generator import GenerationResult

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "with",
    "as",
    "at",
    "by",
    "be",
    "can",
    "this",
    "that",
    "it",
    "you",
    "we",
    "your",
    "our",
    "from",
    "if",
    "there",
    "their",
    "have",
    "has",
}


class EvalTask(BaseModel):
    """A single evaluation example loaded from the JSONL dataset."""

    id: str
    question: str
    expected_answer: str = ""
    expected_sources: list[str] = Field(default_factory=list)
    answer_keywords: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    requires_json: bool = False
    expected_json_schema: dict[str, Any] | None = None
    difficulty: str = "medium"


class LLMJudge(Protocol):
    """Optional extension point for LLM-as-judge scoring (not required)."""

    def score(self, task: EvalTask, answer: str, contexts: list[str]) -> float: ...


@dataclass
class MetricResult:
    """All metrics for one evaluated example."""

    task_id: str
    answer_correctness_score: float
    citation_correctness_score: float
    grounding_score: float
    retrieval_relevance_score: float
    hallucination_flag: bool
    json_validity: str  # pass | fail | n/a
    latency_ms: float
    estimated_tokens: int
    estimated_cost_usd: float
    error_count: int
    passed: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS}


def _normalize_answer_text(answer: str) -> str:
    """Return text used for keyword/grounding checks.

    If the answer is valid JSON, concatenate its string values so that JSON-mode
    answers are scored on their content rather than on structural keys.
    """
    stripped = answer.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return answer

        def _collect(obj: Any) -> list[str]:
            if isinstance(obj, str):
                return [obj]
            if isinstance(obj, dict):
                return [s for v in obj.values() for s in _collect(v)]
            if isinstance(obj, list):
                return [s for v in obj for s in _collect(v)]
            return [str(obj)]

        return " ".join(_collect(parsed))
    return answer


# --------------------------------------------------------------------------- #
# Individual metrics
# --------------------------------------------------------------------------- #


def answer_correctness(task: EvalTask, answer: str) -> tuple[float, dict[str, Any]]:
    text = _normalize_answer_text(answer).lower()
    parts: list[float] = []
    detail: dict[str, Any] = {}

    if task.answer_keywords:
        matched = [k for k in task.answer_keywords if k.lower() in text]
        parts.append(len(matched) / len(task.answer_keywords))
        detail["matched_keywords"] = matched
        detail["missing_keywords"] = [k for k in task.answer_keywords if k.lower() not in text]

    if task.expected_answer:
        exact = task.expected_answer.lower() in text
        parts.append(1.0 if exact else 0.0)
        detail["expected_answer_found"] = exact

    score = sum(parts) / len(parts) if parts else 0.0
    return round(score, 4), detail


def citation_correctness(task: EvalTask, result: GenerationResult) -> tuple[float, dict[str, Any]]:
    cited_docs = {c.document_name for c in result.citations}
    if not task.expected_sources:
        return 1.0, {"cited_docs": sorted(cited_docs), "note": "no expected sources"}
    matched = [s for s in task.expected_sources if s in cited_docs]
    score = len(matched) / len(task.expected_sources)
    return round(score, 4), {"cited_docs": sorted(cited_docs), "matched_sources": matched}


def grounding(result: GenerationResult, answer: str) -> tuple[float, dict[str, Any]]:
    ans_tokens = _content_tokens(_normalize_answer_text(answer))
    ctx_tokens: set[str] = set()
    for ctx in result.retrieved_contexts:
        ctx_tokens |= _content_tokens(ctx.text)
    if not ans_tokens:
        return 0.0, {"note": "empty answer"}
    grounded = ans_tokens & ctx_tokens
    score = len(grounded) / len(ans_tokens)
    return round(score, 4), {"ungrounded_tokens": sorted(ans_tokens - ctx_tokens)[:10]}


def retrieval_relevance(task: EvalTask, result: GenerationResult) -> tuple[float, dict[str, Any]]:
    retrieved_docs = {c.document_name for c in result.retrieved_contexts}
    if not task.expected_sources:
        return 1.0, {"retrieved_docs": sorted(retrieved_docs), "note": "no expected sources"}
    matched = [s for s in task.expected_sources if s in retrieved_docs]
    score = len(matched) / len(task.expected_sources)
    return round(score, 4), {"retrieved_docs": sorted(retrieved_docs), "matched_sources": matched}


def hallucination(
    task: EvalTask, answer: str, grounding_score: float, *, grounding_floor: float = 0.5
) -> tuple[bool, dict[str, Any]]:
    text = _normalize_answer_text(answer).lower()
    forbidden_hits = [c for c in task.forbidden_claims if c.lower() in text]
    unsupported = grounding_score < grounding_floor
    flag = bool(forbidden_hits) or unsupported
    return flag, {
        "forbidden_hits": forbidden_hits,
        "unsupported_low_grounding": unsupported,
        "grounding_floor": grounding_floor,
    }


def json_validity(task: EvalTask, answer: str) -> tuple[str, dict[str, Any]]:
    if not task.requires_json:
        return "n/a", {}
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError as exc:
        return "fail", {"reason": f"invalid JSON: {exc.msg}"}
    if task.expected_json_schema:
        errors = sorted(Draft7Validator(task.expected_json_schema).iter_errors(parsed), key=str)
        if errors:
            return "fail", {
                "reason": "schema validation failed",
                "errors": [e.message for e in errors],
            }
    return "pass", {}


# --------------------------------------------------------------------------- #
# Aggregate evaluation
# --------------------------------------------------------------------------- #


def evaluate_example(
    task: EvalTask,
    result: GenerationResult,
    thresholds: EvalThresholds,
    *,
    error_count: int = 0,
) -> MetricResult:
    """Compute all metrics and the final pass/fail status for one example."""
    correctness, corr_detail = answer_correctness(task, result.answer)
    citation, cite_detail = citation_correctness(task, result)
    ground, ground_detail = grounding(result, result.answer)
    retrieval, retr_detail = retrieval_relevance(task, result)
    halluc_flag, halluc_detail = hallucination(task, result.answer, ground)
    json_status, json_detail = json_validity(task, result.answer)

    passed = (
        error_count == 0
        and correctness >= thresholds.answer_correctness_score
        and citation >= thresholds.citation_correctness_score
        and ground >= thresholds.grounding_score
        and retrieval >= thresholds.retrieval_relevance_score
        and (thresholds.allow_hallucination or not halluc_flag)
        and json_status != "fail"
    )

    return MetricResult(
        task_id=task.id,
        answer_correctness_score=correctness,
        citation_correctness_score=citation,
        grounding_score=ground,
        retrieval_relevance_score=retrieval,
        hallucination_flag=halluc_flag,
        json_validity=json_status,
        latency_ms=result.latency_ms,
        estimated_tokens=result.estimated_tokens,
        estimated_cost_usd=result.estimated_cost,
        error_count=error_count,
        passed=passed,
        detail={
            "correctness": corr_detail,
            "citation": cite_detail,
            "grounding": ground_detail,
            "retrieval": retr_detail,
            "hallucination": halluc_detail,
            "json": json_detail,
        },
    )
