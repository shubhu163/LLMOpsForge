"""Answer generation: assemble the prompt, call the provider, track cost/citations."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.config import ModelConfigSpec, PromptTemplateConfig
from app.providers.base import BaseLLMProvider, RetrievedContext
from app.rag.citations import Citation, extract_citations


@dataclass
class GenerationResult:
    answer: str
    citations: list[Citation]
    retrieved_contexts: list[RetrievedContext]
    latency_ms: float
    estimated_tokens: int
    estimated_cost: float
    model_name: str
    prompt_template_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_contexts": [
                {
                    "chunk_id": c.chunk_id,
                    "document_name": c.document_name,
                    "text": c.text,
                    "score": c.score,
                }
                for c in self.retrieved_contexts
            ],
            "latency_ms": self.latency_ms,
            "estimated_tokens": self.estimated_tokens,
            "estimated_cost": self.estimated_cost,
            "model_name": self.model_name,
            "prompt_template_id": self.prompt_template_id,
        }


def _format_context_block(contexts: list[RetrievedContext]) -> str:
    lines = []
    for c in contexts:
        lines.append(f"[source: {c.document_name} | chunk: {c.chunk_id}]\n{c.text}")
    return "\n\n".join(lines) if lines else "(no context retrieved)"


def _estimate_cost(prompt_tokens: int, completion_tokens: int, spec: ModelConfigSpec) -> float:
    return round(
        (prompt_tokens / 1000.0) * spec.input_cost_per_1k
        + (completion_tokens / 1000.0) * spec.output_cost_per_1k,
        8,
    )


def generate_answer(
    *,
    provider: BaseLLMProvider,
    template: PromptTemplateConfig,
    model_spec: ModelConfigSpec,
    question: str,
    contexts: list[RetrievedContext],
    require_citations: bool = True,
    requires_json: bool = False,
    expected_json_schema: dict[str, Any] | None = None,
) -> GenerationResult:
    """Run a single generation and return answer, citations, cost, and latency."""
    context_block = _format_context_block(contexts)
    prompt = template.instructions.format(question=question, context=context_block)

    start = time.perf_counter()
    response = provider.generate(
        system=template.system,
        prompt=prompt,
        contexts=contexts,
        question=question,
        behavior=template.behavior.model_dump(),
        requires_json=requires_json,
        expected_json_schema=expected_json_schema,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    citations = extract_citations(
        answer=response.text,
        contexts=contexts,
        used_chunk_ids=response.raw.get("used_chunk_ids"),
        require_citations=require_citations,
        always_cite=bool(template.behavior.always_cite),
    )

    cost = _estimate_cost(response.prompt_tokens, response.completion_tokens, model_spec)

    return GenerationResult(
        answer=response.text,
        citations=citations,
        retrieved_contexts=contexts,
        latency_ms=round(latency_ms, 3),
        estimated_tokens=response.total_tokens,
        estimated_cost=cost,
        model_name=response.model_name,
        prompt_template_id=template.id,
    )
