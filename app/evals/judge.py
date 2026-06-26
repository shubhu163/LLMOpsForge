"""LLM-as-judge: grade answers with a real model when ground-truth labels are weak.

Deterministic metrics need labels (expected answer/sources). When assessing an
external RAG over documents you haven't hand-labelled, an LLM judge can score
faithfulness and correctness instead. This is optional and additive — the
deterministic metrics remain the primary, reproducible signal.

``ProviderLLMJudge`` works with ANY provider (OpenRouter, OpenAI, Ollama, or even
the mock for tests), so the judge model is just another model config id.
"""

from __future__ import annotations

import json
from typing import Any

from app.evals.metrics import EvalTask
from app.logging_config import get_logger
from app.providers.base import BaseLLMProvider

logger = get_logger(__name__)

_JUDGE_SYSTEM = (
    "You are a strict evaluator of question-answering systems. "
    "You score answers only on the evidence provided. Respond ONLY with JSON."
)

_JUDGE_TEMPLATE = """Question:
{question}

Reference answer (may be empty):
{reference}

Retrieved context the system was given:
{context}

System's answer:
{answer}

Rate the system's answer on a 0.0-1.0 scale for each criterion and return JSON:
{{"correctness": <0-1>, "faithfulness": <0-1>, "relevancy": <0-1>, "overall": <0-1>, "reasoning": "<one sentence>"}}
- correctness: does it match the reference / known facts?
- faithfulness: is every claim supported by the retrieved context (no hallucination)?
- relevancy: does it actually answer the question?
Return ONLY the JSON object."""


def _clamp(x: Any) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


class ProviderLLMJudge:
    """Scores answers using an LLM provider. Returns an overall 0-1 score."""

    def __init__(self, provider: BaseLLMProvider):
        self.provider = provider

    def evaluate(self, task: EvalTask, answer: str, contexts: list[str]) -> dict[str, Any]:
        prompt = _JUDGE_TEMPLATE.format(
            question=task.question,
            reference=task.expected_answer or "(none)",
            context="\n---\n".join(contexts) or "(none)",
            answer=answer or "(empty)",
        )
        resp = self.provider.generate(system=_JUDGE_SYSTEM, prompt=prompt, requires_json=True)
        text = resp.text.strip()
        try:
            # Be tolerant of models that wrap JSON in prose/code fences.
            start, end = text.find("{"), text.rfind("}")
            data = json.loads(text[start : end + 1]) if start >= 0 else {}
        except json.JSONDecodeError:
            logger.warning("Judge returned non-JSON output; scoring 0.")
            data = {}
        return {
            "correctness": _clamp(data.get("correctness")),
            "faithfulness": _clamp(data.get("faithfulness")),
            "relevancy": _clamp(data.get("relevancy")),
            "overall": _clamp(data.get("overall", data.get("correctness"))),
            "reasoning": str(data.get("reasoning", ""))[:300],
            "judge_model": resp.model_name,
        }

    def score(self, task: EvalTask, answer: str, contexts: list[str]) -> float:
        """Satisfies the LLMJudge protocol: return the overall 0-1 score."""
        return self.evaluate(task, answer, contexts)["overall"]


def build_judge(model_config_id: str) -> ProviderLLMJudge:
    """Construct a judge from a model config id (e.g. 'openrouter-free')."""
    from app.config import load_model_config
    from app.providers.factory import build_provider

    return ProviderLLMJudge(build_provider(load_model_config(model_config_id)))
