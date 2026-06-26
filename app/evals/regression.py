"""Prompt/model regression comparison between two evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy.orm import Session

from app.logging_config import get_logger
from app.storage import models, repository

logger = get_logger(__name__)

# Metrics where a higher value is better (used to interpret deltas).
_HIGHER_IS_BETTER = {
    "pass_rate",
    "answer_correctness_avg",
    "citation_correctness_avg",
    "grounding_avg",
    "retrieval_relevance_avg",
    "json_validity_rate",
}
# Metrics where a lower value is better.
_LOWER_IS_BETTER = {"hallucination_rate", "avg_latency_ms", "total_estimated_cost_usd"}

_COMPARED_METRICS = sorted(_HIGHER_IS_BETTER | _LOWER_IS_BETTER)


@dataclass
class TaskState:
    passed: bool
    hallucination: bool
    correctness: float


def _task_states(results: list[models.EvalExampleResult]) -> dict[str, TaskState]:
    states: dict[str, TaskState] = {}
    for r in results:
        states[r.task_id] = TaskState(
            passed=bool(r.passed),
            hallucination=bool(r.hallucination_flag),
            correctness=float(r.metrics.get("answer_correctness_score", 0.0)),
        )
    return states


def _pct_change(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None:
        return None
    if baseline == 0:
        return None if candidate == 0 else float("inf")
    return round((candidate - baseline) / abs(baseline) * 100.0, 2)


def _delta_table(base: dict[str, Any], cand: dict[str, Any]) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for metric in _COMPARED_METRICS:
        b = base.get(metric)
        c = cand.get(metric)
        if b is None and c is None:
            continue
        delta = None if (b is None or c is None) else round(c - b, 4)
        improved = None
        if delta is not None:
            if metric in _HIGHER_IS_BETTER:
                improved = delta > 0
            elif metric in _LOWER_IS_BETTER:
                improved = delta < 0
        table[metric] = {
            "baseline": b,
            "candidate": c,
            "delta": delta,
            "pct_change": _pct_change(b, c),
            "improved": improved,
        }
    return table


def _recommendation(
    deltas: dict[str, dict[str, Any]], newly_failed: list[str], critical: list[str]
) -> str:
    pass_rate_delta = deltas.get("pass_rate", {}).get("delta") or 0.0
    halluc_delta = deltas.get("hallucination_rate", {}).get("delta") or 0.0

    if critical or pass_rate_delta < -0.1 or halluc_delta > 0.05:
        return "blocked"
    if not newly_failed and pass_rate_delta >= 0 and halluc_delta <= 0:
        return "safe to ship"
    return "investigate"


def compare_runs(
    session: Session, *, baseline_run_id: str, candidate_run_id: str, persist: bool = True
) -> models.RegressionReport:
    """Compare a candidate eval run against a baseline and build a report."""
    baseline = repository.get_eval_run(session, baseline_run_id)
    candidate = repository.get_eval_run(session, candidate_run_id)
    if baseline is None:
        raise KeyError(f"Baseline run not found: {baseline_run_id}")
    if candidate is None:
        raise KeyError(f"Candidate run not found: {candidate_run_id}")

    base_states = _task_states(repository.get_results_for_run(session, baseline_run_id))
    cand_states = _task_states(repository.get_results_for_run(session, candidate_run_id))
    shared = sorted(set(base_states) & set(cand_states))

    newly_failed = [t for t in shared if base_states[t].passed and not cand_states[t].passed]
    fixed = [t for t in shared if not base_states[t].passed and cand_states[t].passed]
    improved_tasks = [t for t in shared if cand_states[t].correctness > base_states[t].correctness]
    # Critical = a task that did not hallucinate in baseline but does in candidate.
    critical_regressions = [
        t for t in shared if not base_states[t].hallucination and cand_states[t].hallucination
    ]

    deltas = _delta_table(baseline.summary, candidate.summary)
    recommendation = _recommendation(deltas, newly_failed, critical_regressions)

    summary = {
        "baseline_run_id": baseline_run_id,
        "candidate_run_id": candidate_run_id,
        "baseline_summary": baseline.summary,
        "candidate_summary": candidate.summary,
        "deltas": deltas,
        "newly_failed": newly_failed,
        "fixed": fixed,
        "improved_tasks": improved_tasks,
        "critical_regressions": critical_regressions,
        "shared_tasks": len(shared),
        "recommendation": recommendation,
    }

    report = models.RegressionReport(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        summary=summary,
        recommendation=recommendation,
    )
    if persist:
        session.add(report)
        session.commit()
    logger.info(
        "Regression %s vs %s -> %s (%d newly failed, %d critical)",
        baseline_run_id,
        candidate_run_id,
        recommendation,
        len(newly_failed),
        len(critical_regressions),
    )
    return report
