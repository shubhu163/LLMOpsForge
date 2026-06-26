"""Tests for the Typer CLI."""

from __future__ import annotations

from typer.testing import CliRunner

from app.cli import app
from app.config import load_eval_config
from app.evals.runner import EvalRunner

runner = CliRunner()
DATASET = "datasets/qa_eval.jsonl"


def test_ingest_and_query(temp_env):
    ingest = runner.invoke(app, ["ingest", "--docs-path", "documents/"])
    assert ingest.exit_code == 0, ingest.output
    assert "Ingested" in ingest.output

    query = runner.invoke(
        app,
        [
            "query",
            "How many days do customers have to request a full refund?",
            "--prompt",
            "prompt_v2",
        ],
    )
    assert query.exit_code == 0, query.output
    assert "30 days" in query.output


def test_query_without_ingest_fails(temp_env):
    result = runner.invoke(app, ["query", "anything"])
    assert result.exit_code == 1
    assert "No documents" in result.output


def test_eval_command(temp_env):
    runner.invoke(app, ["ingest", "--docs-path", "documents/"])
    result = runner.invoke(app, ["eval", "--dataset", DATASET, "--config", "configs/default.yaml"])
    assert result.exit_code == 0, result.output
    assert "Eval run complete" in result.output


def test_report_command(temp_env, session, ingested_pipeline):
    cfg = load_eval_config("configs/default.yaml")
    run = EvalRunner(session, pipeline=ingested_pipeline).run(
        dataset_path=DATASET, config=cfg, name="cli-report"
    )
    result = runner.invoke(app, ["report", "--eval-run-id", run.id, "--format", "json"])
    assert result.exit_code == 0, result.output
    assert run.id in result.output


def test_regression_command(temp_env, session, ingested_pipeline):
    cfg = load_eval_config("configs/default.yaml")
    runner_ = EvalRunner(session, pipeline=ingested_pipeline)
    base = runner_.run(dataset_path=DATASET, config=cfg, name="baseline")
    cand = runner_.run(dataset_path=DATASET, config=cfg, name="candidate")
    result = runner.invoke(
        app,
        ["regression", "--baseline-run-id", base.id, "--candidate-run-id", cand.id],
    )
    assert result.exit_code == 0, result.output
    assert "Regression Report" in result.output
