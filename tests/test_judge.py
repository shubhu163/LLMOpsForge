"""Tests for the LLM-as-judge scorer using a fake provider (no network)."""

from __future__ import annotations

import json

from app.evals.judge import ProviderLLMJudge
from app.evals.metrics import EvalTask
from app.providers.base import BaseLLMProvider, LLMResponse


class FakeJudgeProvider(BaseLLMProvider):
    """Returns a fixed JSON verdict, simulating a real judge model."""

    def __init__(self, payload: dict):
        super().__init__("fake-judge")
        self._payload = payload

    @property
    def name(self) -> str:
        return "fake"

    def generate(self, *, system, prompt, **kwargs) -> LLMResponse:
        return LLMResponse(text=json.dumps(self._payload), model_name=self.model_name)


def test_judge_parses_scores():
    judge = ProviderLLMJudge(
        FakeJudgeProvider(
            {
                "correctness": 0.9,
                "faithfulness": 0.8,
                "relevancy": 1.0,
                "overall": 0.85,
                "reasoning": "matches reference",
            }
        )
    )
    task = EvalTask(id="t", question="q", expected_answer="30 days")
    detail = judge.evaluate(task, "Refunds within 30 days.", ["context text"])
    assert detail["overall"] == 0.85
    assert detail["correctness"] == 0.9
    assert judge.score(task, "Refunds within 30 days.", ["ctx"]) == 0.85


def test_judge_handles_garbage_output():
    judge = ProviderLLMJudge(FakeJudgeProvider({}))

    class Garbage(FakeJudgeProvider):
        def generate(self, *, system, prompt, **kwargs):
            return LLMResponse(text="not json at all", model_name="fake-judge")

    judge = ProviderLLMJudge(Garbage({}))
    task = EvalTask(id="t", question="q")
    assert judge.score(task, "answer", ["ctx"]) == 0.0


def test_judge_integration_in_runner(session, tmp_path):
    from app.config import load_eval_config
    from app.evals.adapters import FunctionRagAdapter
    from app.evals.runner import EvalRunner

    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "r1",
                "question": "refund days?",
                "expected_answer": "30 days",
                "answer_keywords": ["30 days"],
                "difficulty": "easy",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = FunctionRagAdapter(lambda q: {"answer": "30 days", "retrieved_contexts": ["30 days"]})
    judge = ProviderLLMJudge(FakeJudgeProvider({"overall": 0.75, "correctness": 0.75}))

    run = EvalRunner(session, pipeline=adapter, judge=judge).run(
        dataset_path=str(dataset), config=load_eval_config("configs/default.yaml"), name="judged"
    )
    assert run.summary["judge_overall_avg"] == 0.75
