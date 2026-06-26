"""Report generation for eval runs and regression comparisons (Markdown + JSON)."""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.storage import models, repository


def eval_report_dict(session: Session, run_id: str) -> dict[str, Any]:
    """Build a complete JSON-serialisable report for an eval run."""
    run = repository.get_eval_run(session, run_id)
    if run is None:
        raise KeyError(f"Eval run not found: {run_id}")
    results = repository.get_results_for_run(session, run_id)
    return {
        "run": {
            "id": run.id,
            "name": run.name,
            "dataset_path": run.dataset_path,
            "prompt_template_id": run.prompt_template_id,
            "model_config_id": run.model_config_id,
            "num_examples": run.num_examples,
            "created_at": run.created_at.isoformat(),
            "config": run.config,
        },
        "summary": run.summary,
        "results": [
            {
                "task_id": r.task_id,
                "question": r.question,
                "answer": r.answer,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "hallucination_flag": r.hallucination_flag,
                "json_validity": r.json_validity,
                "citations": r.citations,
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
                "estimated_tokens": r.estimated_tokens,
                "estimated_cost": r.estimated_cost,
            }
            for r in results
        ],
    }


def eval_report_json(session: Session, run_id: str) -> str:
    return json.dumps(eval_report_dict(session, run_id), indent=2)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def eval_report_markdown(session: Session, run_id: str) -> str:
    data = eval_report_dict(session, run_id)
    run, summary, results = data["run"], data["summary"], data["results"]

    lines: list[str] = []
    lines.append(f"# Evaluation Report — `{run['id']}`")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- **Name:** {run['name']}")
    lines.append(f"- **Dataset:** `{run['dataset_path']}`")
    lines.append(f"- **Prompt template:** `{run['prompt_template_id']}`")
    lines.append(f"- **Model config:** `{run['model_config_id']}`")
    lines.append(f"- **Examples:** {run['num_examples']}")
    lines.append(f"- **Created:** {run['created_at']}")
    lines.append("")

    lines.append("## Summary Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    metric_order = [
        ("Pass rate", "pass_rate"),
        ("Passed", "passed"),
        ("Failed", "failed"),
        ("Answer correctness (avg)", "answer_correctness_avg"),
        ("Citation correctness (avg)", "citation_correctness_avg"),
        ("Grounding (avg)", "grounding_avg"),
        ("Retrieval relevance (avg)", "retrieval_relevance_avg"),
        ("Hallucination count", "hallucination_count"),
        ("Hallucination rate", "hallucination_rate"),
        ("JSON validity rate", "json_validity_rate"),
        ("Avg latency (ms)", "avg_latency_ms"),
        ("Total tokens", "total_estimated_tokens"),
        ("Total cost (USD)", "total_estimated_cost_usd"),
        ("Errors", "error_count"),
    ]
    for label, key in metric_order:
        lines.append(f"| {label} | {_fmt(summary.get(key))} |")
    lines.append("")

    failed = [r for r in results if not r["passed"]]
    halluc = [r for r in results if r["hallucination_flag"]]
    citation_failures = [
        r for r in results if r["metrics"].get("citation_correctness_score", 1.0) < 1.0
    ]
    retrieval_failures = [
        r for r in results if r["metrics"].get("retrieval_relevance_score", 1.0) < 1.0
    ]

    lines.append(f"## Failed Examples ({len(failed)})")
    lines.append("")
    if failed:
        lines.append(
            "| Task | Difficulty | Correctness | Grounding | Citation | JSON | Hallucination |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for r in failed:
            m = r["metrics"]
            lines.append(
                f"| `{r['task_id']}` | {r['difficulty']} "
                f"| {_fmt(m.get('answer_correctness_score'))} "
                f"| {_fmt(m.get('grounding_score'))} "
                f"| {_fmt(m.get('citation_correctness_score'))} "
                f"| {r['json_validity']} "
                f"| {'⚠️' if r['hallucination_flag'] else '—'} |"
            )
    else:
        lines.append("_None — all examples passed._")
    lines.append("")

    lines.append(f"## Hallucination Cases ({len(halluc)})")
    lines.append("")
    if halluc:
        for r in halluc:
            detail = r["metrics"].get("detail", {}).get("hallucination", {})
            lines.append(f"- `{r['task_id']}`: {json.dumps(detail)}")
    else:
        lines.append("_None detected._")
    lines.append("")

    lines.append(f"## Citation Failures ({len(citation_failures)})")
    lines.append("")
    if citation_failures:
        for r in citation_failures:
            cited = [c.get("document_name") for c in r["citations"]]
            lines.append(f"- `{r['task_id']}`: cited {cited or '[]'}")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append(f"## Retrieval Failures ({len(retrieval_failures)})")
    lines.append("")
    if retrieval_failures:
        for r in retrieval_failures:
            detail = r["metrics"].get("detail", {}).get("retrieval", {})
            lines.append(f"- `{r['task_id']}`: {json.dumps(detail)}")
    else:
        lines.append("_None._")
    lines.append("")

    lines.append("## Latency / Cost Summary")
    lines.append("")
    lines.append(f"- **Average latency:** {_fmt(summary.get('avg_latency_ms'))} ms")
    lines.append(f"- **Total estimated tokens:** {_fmt(summary.get('total_estimated_tokens'))}")
    lines.append(f"- **Total estimated cost:** ${_fmt(summary.get('total_estimated_cost_usd'))}")
    lines.append("")

    lines.append("## Recommendations")
    lines.append("")
    for rec in _recommendations(summary, failed, halluc):
        lines.append(f"- {rec}")
    lines.append("")

    return "\n".join(lines)


def _recommendations(summary: dict, failed: list, halluc: list) -> list[str]:
    recs: list[str] = []
    pass_rate = summary.get("pass_rate", 0.0) or 0.0
    if pass_rate >= 0.9:
        recs.append("Pass rate is healthy (≥90%).")
    else:
        recs.append(f"Pass rate is {pass_rate:.0%} — investigate the failed examples above.")
    if halluc:
        recs.append(f"{len(halluc)} hallucination case(s) detected — tighten grounding/prompts.")
    if summary.get("json_validity_rate") not in (None, 1.0):
        recs.append("JSON validity below 100% — enforce strict JSON mode in the prompt.")
    if not failed:
        recs.append("All examples passed; consider expanding the dataset with harder cases.")
    return recs


# --------------------------------------------------------------------------- #
# Regression report
# --------------------------------------------------------------------------- #


def regression_report_markdown(report: models.RegressionReport) -> str:
    s = report.summary
    deltas = s.get("deltas", {})

    lines: list[str] = []
    lines.append("# Regression Report")
    lines.append("")
    lines.append(f"- **Baseline run:** `{s['baseline_run_id']}`")
    lines.append(f"- **Candidate run:** `{s['candidate_run_id']}`")
    lines.append(f"- **Shared tasks:** {s.get('shared_tasks', 0)}")
    lines.append("")

    lines.append("## Baseline vs Candidate")
    lines.append("")
    lines.append("| Metric | Baseline | Candidate | Δ | % change | Verdict |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for metric, row in deltas.items():
        verdict = "—"
        if row.get("improved") is True:
            verdict = "✅ better"
        elif row.get("improved") is False:
            verdict = "🔻 worse"
        pct = row.get("pct_change")
        pct_str = "—" if pct is None else f"{pct:+.1f}%"
        lines.append(
            f"| {metric} | {_fmt(row.get('baseline'))} | {_fmt(row.get('candidate'))} "
            f"| {_fmt(row.get('delta'))} | {pct_str} | {verdict} |"
        )
    lines.append("")

    def _list_section(title: str, items: list[str]) -> None:
        lines.append(f"## {title} ({len(items)})")
        lines.append("")
        if items:
            for t in items:
                lines.append(f"- `{t}`")
        else:
            lines.append("_None._")
        lines.append("")

    _list_section("Improved Tasks", s.get("improved_tasks", []))
    _list_section("Fixed Tasks", s.get("fixed", []))
    _list_section("Newly Failed Tasks", s.get("newly_failed", []))
    _list_section("Critical Hallucination Regressions", s.get("critical_regressions", []))

    rec = s.get("recommendation", "investigate")
    emoji = {"safe to ship": "✅", "investigate": "🟡", "blocked": "⛔"}.get(rec, "🟡")
    lines.append("## Final Recommendation")
    lines.append("")
    lines.append(f"**{emoji} {rec.upper()}**")
    lines.append("")
    return "\n".join(lines)
