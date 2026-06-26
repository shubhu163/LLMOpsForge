"""Tests for the eval runner and the regression comparator."""

from __future__ import annotations

import pytest

from app.config import load_eval_config
from app.evals.regression import compare_runs
from app.evals.runner import EvalRunner

DATASET = "datasets/qa_eval.jsonl"


@pytest.fixture
def configs():
    v1 = load_eval_config("configs/default.yaml")
    v2 = v1.model_copy(update={"prompt_template_id": "prompt_v2", "name": "candidate-v2"})
    return v1, v2


def _run(session, ingested_pipeline, config, name):
    runner = EvalRunner(session, pipeline=ingested_pipeline)
    return runner.run(dataset_path=DATASET, config=config, name=name)


def test_eval_run_summary_shape(session, ingested_pipeline, configs):
    v1, _ = configs
    run = _run(session, ingested_pipeline, v1, "baseline")
    s = run.summary
    assert s["total_examples"] >= 20
    assert 0.0 <= s["pass_rate"] <= 1.0
    assert "answer_correctness_avg" in s
    assert "grounding_avg" in s
    assert s["total_estimated_cost_usd"] == 0.0  # mock provider is free


def test_v2_improves_json_validity_over_v1(session, ingested_pipeline, configs):
    v1, v2 = configs
    run_v1 = _run(session, ingested_pipeline, v1, "baseline")
    run_v2 = _run(session, ingested_pipeline, v2, "candidate")
    # v2 enforces strict JSON; v1 does not -> v2 should be at least as good.
    assert run_v2.summary["json_validity_rate"] >= run_v1.summary["json_validity_rate"]
    assert run_v2.summary["json_validity_rate"] == 1.0


def test_regression_compare(session, ingested_pipeline, configs):
    v1, v2 = configs
    baseline = _run(session, ingested_pipeline, v1, "baseline")
    candidate = _run(session, ingested_pipeline, v2, "candidate")

    report = compare_runs(session, baseline_run_id=baseline.id, candidate_run_id=candidate.id)
    assert report.recommendation in {"safe to ship", "investigate", "blocked"}
    assert "deltas" in report.summary
    assert report.summary["shared_tasks"] >= 20
    assert isinstance(report.summary["newly_failed"], list)
    assert isinstance(report.summary["fixed"], list)


def test_regression_identical_runs_safe(session, ingested_pipeline, configs):
    v1, _ = configs
    run = _run(session, ingested_pipeline, v1, "baseline")
    report = compare_runs(session, baseline_run_id=run.id, candidate_run_id=run.id)
    assert report.summary["newly_failed"] == []
    assert report.recommendation == "safe to ship"


def test_missing_run_raises(session, ingested_pipeline, configs):
    v1, _ = configs
    run = _run(session, ingested_pipeline, v1, "baseline")
    with pytest.raises(KeyError):
        compare_runs(session, baseline_run_id=run.id, candidate_run_id="does-not-exist")
