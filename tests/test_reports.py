"""Tests for report generation (Markdown + JSON)."""

from __future__ import annotations

import json

from app.config import load_eval_config
from app.evals.regression import compare_runs
from app.evals.reports import (
    eval_report_dict,
    eval_report_json,
    eval_report_markdown,
    regression_report_markdown,
)
from app.evals.runner import EvalRunner

DATASET = "datasets/qa_eval.jsonl"


def _run(session, pipeline, name="run"):
    cfg = load_eval_config("configs/default.yaml")
    return EvalRunner(session, pipeline=pipeline).run(dataset_path=DATASET, config=cfg, name=name)


def test_eval_report_markdown(session, ingested_pipeline):
    run = _run(session, ingested_pipeline)
    md = eval_report_markdown(session, run.id)
    assert "# Evaluation Report" in md
    assert "## Summary Metrics" in md
    assert "## Failed Examples" in md
    assert "## Recommendations" in md


def test_eval_report_json_parseable(session, ingested_pipeline):
    run = _run(session, ingested_pipeline)
    payload = json.loads(eval_report_json(session, run.id))
    assert payload["run"]["id"] == run.id
    assert "summary" in payload
    assert len(payload["results"]) == run.num_examples


def test_eval_report_dict_structure(session, ingested_pipeline):
    run = _run(session, ingested_pipeline)
    data = eval_report_dict(session, run.id)
    assert set(data.keys()) == {"run", "summary", "results"}


def test_regression_report_markdown(session, ingested_pipeline):
    base = _run(session, ingested_pipeline, "baseline")
    cand = _run(session, ingested_pipeline, "candidate")
    report = compare_runs(session, baseline_run_id=base.id, candidate_run_id=cand.id)
    md = regression_report_markdown(report)
    assert "# Regression Report" in md
    assert "Final Recommendation" in md
    assert "Baseline vs Candidate" in md
