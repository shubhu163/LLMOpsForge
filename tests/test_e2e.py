"""End-to-end test: ingest -> eval -> report -> regression."""

from __future__ import annotations

import json

from app.config import load_eval_config
from app.evals.regression import compare_runs
from app.evals.reports import eval_report_json, eval_report_markdown, regression_report_markdown
from app.evals.runner import EvalRunner

DATASET = "datasets/qa_eval.jsonl"


def test_full_lifecycle(session, pipeline, docs_dir, tmp_path):
    # 1. Ingest the corpus.
    ingest = pipeline.ingest(docs_dir)
    assert ingest["documents"] >= 5

    # 2. Run baseline (v1) and candidate (v2) evaluations.
    v1 = load_eval_config("configs/default.yaml")
    v2 = v1.model_copy(update={"prompt_template_id": "prompt_v2", "name": "candidate-v2"})
    runner = EvalRunner(session, pipeline=pipeline)
    baseline = runner.run(dataset_path=DATASET, config=v1, name="baseline-v1")
    candidate = runner.run(dataset_path=DATASET, config=v2, name="candidate-v2")

    assert baseline.summary["total_examples"] >= 20
    assert candidate.summary["total_examples"] >= 20

    # 3. Generate reports (Markdown + JSON) and write them to disk.
    md = eval_report_markdown(session, candidate.id)
    js = eval_report_json(session, candidate.id)
    md_path = tmp_path / "report.md"
    js_path = tmp_path / "report.json"
    md_path.write_text(md, encoding="utf-8")
    js_path.write_text(js, encoding="utf-8")
    assert md_path.read_text().startswith("# Evaluation Report")
    assert json.loads(js_path.read_text())["run"]["id"] == candidate.id

    # 4. Run regression comparison and assert a verdict is produced.
    report = compare_runs(session, baseline_run_id=baseline.id, candidate_run_id=candidate.id)
    reg_md = regression_report_markdown(report)
    assert "Final Recommendation" in reg_md
    assert report.recommendation in {"safe to ship", "investigate", "blocked"}

    # The candidate (v2) should not regress JSON validity vs the baseline (v1).
    assert candidate.summary["json_validity_rate"] >= baseline.summary["json_validity_rate"]
