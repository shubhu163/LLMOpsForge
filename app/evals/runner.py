"""Evaluation runner: execute a dataset against the RAG pipeline and persist results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.config import EvalConfig
from app.evals.metrics import EvalTask, MetricResult, evaluate_example
from app.logging_config import get_logger
from app.rag.generator import GenerationResult
from app.rag.pipeline import RagPipeline
from app.storage import models

logger = get_logger(__name__)


def load_dataset(path: str | Path) -> list[EvalTask]:
    """Load evaluation tasks from a JSONL file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    tasks: list[EvalTask] = []
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                tasks.append(EvalTask(**json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Invalid dataset row at line {line_no}: {exc}") from exc
    if not tasks:
        raise ValueError(f"Dataset is empty: {p}")
    logger.info("Loaded %d eval tasks from %s", len(tasks), p)
    return tasks


def _empty_result(model_name: str, template_id: str) -> GenerationResult:
    return GenerationResult(
        answer="",
        citations=[],
        retrieved_contexts=[],
        latency_ms=0.0,
        estimated_tokens=0,
        estimated_cost=0.0,
        model_name=model_name,
        prompt_template_id=template_id,
    )


def _summarize(metrics: list[MetricResult]) -> dict[str, Any]:
    n = len(metrics)
    if n == 0:
        return {"total_examples": 0}

    def avg(key: str) -> float:
        return round(sum(getattr(m, key) for m in metrics) / n, 4)

    json_considered = [m for m in metrics if m.json_validity != "n/a"]
    json_pass = sum(1 for m in json_considered if m.json_validity == "pass")
    passed = sum(1 for m in metrics if m.passed)

    judged = [m.judge_score for m in metrics if m.judge_score is not None]
    judge_avg = round(sum(judged) / len(judged), 4) if judged else None

    return {
        "total_examples": n,
        "passed": passed,
        "failed": n - passed,
        "pass_rate": round(passed / n, 4),
        "answer_correctness_avg": avg("answer_correctness_score"),
        "citation_correctness_avg": avg("citation_correctness_score"),
        "grounding_avg": avg("grounding_score"),
        "retrieval_relevance_avg": avg("retrieval_relevance_score"),
        "hallucination_count": sum(1 for m in metrics if m.hallucination_flag),
        "hallucination_rate": round(sum(1 for m in metrics if m.hallucination_flag) / n, 4),
        "json_validity_rate": round(json_pass / len(json_considered), 4)
        if json_considered
        else None,
        "json_tasks": len(json_considered),
        "avg_latency_ms": avg("latency_ms"),
        "total_estimated_tokens": sum(m.estimated_tokens for m in metrics),
        "total_estimated_cost_usd": round(sum(m.estimated_cost_usd for m in metrics), 6),
        "error_count": sum(m.error_count for m in metrics),
        "judge_overall_avg": judge_avg,
    }


class EvalRunner:
    """Runs an evaluation dataset and stores an :class:`EvalRun` with results."""

    def __init__(self, session: Session, *, pipeline=None, judge=None):
        # ``pipeline`` is any RagSystem (the built-in RagPipeline or an external
        # adapter). ``judge`` is an optional LLM-as-judge scorer.
        self.session = session
        self.pipeline = pipeline or RagPipeline(session)
        self.judge = judge

    def run(
        self,
        *,
        dataset_path: str,
        config: EvalConfig,
        name: str | None = None,
    ) -> models.EvalRun:
        """Execute every task and persist a complete eval run."""
        tasks = load_dataset(dataset_path)
        run = models.EvalRun(
            name=name or config.name,
            dataset_path=str(dataset_path),
            prompt_template_id=config.prompt_template_id,
            model_config_id=config.model_config_id,
            config=config.model_dump(),
            num_examples=len(tasks),
            summary={},
        )
        self.session.add(run)
        self.session.flush()

        metrics: list[MetricResult] = []
        for task in tasks:
            error_count = 0
            try:
                result = self.pipeline.query(
                    task.question,
                    top_k=config.rag.top_k,
                    prompt_template_id=config.prompt_template_id,
                    model_config_id=config.model_config_id,
                    require_citations=config.rag.require_citations,
                    requires_json=task.requires_json,
                    expected_json_schema=task.expected_json_schema,
                    persist=False,
                )
            except Exception as exc:  # keep the run going; record the failure
                logger.error("Task %s failed: %s", task.id, exc)
                error_count = 1
                result = _empty_result(config.model_config_id, config.prompt_template_id)

            metric = evaluate_example(task, result, config.thresholds, error_count=error_count)

            # Optional LLM-as-judge pass (additive; does not change pass/fail).
            if self.judge is not None and error_count == 0:
                try:
                    judge_detail = self.judge.evaluate(
                        task, result.answer, [c.text for c in result.retrieved_contexts]
                    )
                    metric.judge_score = judge_detail.get("overall")
                    metric.judge_detail = judge_detail
                except Exception as exc:  # judge failures must not abort the run
                    logger.warning("Judge failed for task %s: %s", task.id, exc)

            metrics.append(metric)

            self.session.add(
                models.EvalExampleResult(
                    eval_run_id=run.id,
                    task_id=task.id,
                    question=task.question,
                    answer=result.answer,
                    difficulty=task.difficulty,
                    citations=[c.to_dict() for c in result.citations],
                    retrieved_contexts=result.to_dict()["retrieved_contexts"],
                    metrics=metric.to_dict(),
                    passed=metric.passed,
                    hallucination_flag=metric.hallucination_flag,
                    json_validity=metric.json_validity,
                    latency_ms=metric.latency_ms,
                    estimated_tokens=metric.estimated_tokens,
                    estimated_cost=metric.estimated_cost_usd,
                    error_count=metric.error_count,
                )
            )

        run.summary = _summarize(metrics)
        self.session.commit()
        logger.info(
            "Eval run %s complete: %d/%d passed",
            run.id,
            run.summary.get("passed", 0),
            run.summary.get("total_examples", 0),
        )
        return run
