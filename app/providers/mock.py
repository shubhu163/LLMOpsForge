"""Deterministic, zero-cost LLM provider.

The mock synthesises answers by extracting the sentences from the retrieved
context that best overlap the question. Because the output is drawn directly
from context, it is *grounded by construction* and fully reproducible — ideal
for tests, CI, and offline demos.

Prompt-behavior flags make v1 and v2 templates behave differently so regression
comparisons are meaningful:

* ``verbosity``      -> how many context sentences are returned (more = higher
  grounding/correctness but more tokens/cost).
* ``strict_json``    -> emit valid JSON for JSON tasks (v2) vs prose (v1).
* ``always_cite``    -> cite every used chunk (v2) vs only the strongest (v1).
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    RetrievedContext,
    estimate_tokens,
)

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "what",
    "which",
    "how",
    "do",
    "does",
    "i",
    "you",
    "we",
    "it",
    "this",
    "that",
    "with",
    "as",
    "at",
    "by",
    "be",
    "can",
    "my",
    "your",
    "from",
    "if",
    "there",
    "their",
    "have",
    "has",
    "long",
    "much",
    "many",
    "when",
    "where",
}

_VERBOSITY_SENTENCES = {"low": 1, "medium": 2, "high": 3}


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _split_sentences(text: str) -> list[str]:
    # Drop Markdown headings and code-fence lines: they often contain the
    # question's keywords (e.g. "## Refund Period") but carry no answer content,
    # so they must not be selectable as answer sentences.
    kept_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("```"):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)

    # Split on sentence punctuation and newlines/bullets; keep non-trivial pieces.
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\s)[-*]\s+", cleaned)
    return [p.strip(" -*\t_") for p in parts if p and len(p.strip(" -*\t_")) > 12]


class MockLLMProvider(BaseLLMProvider):
    """A deterministic provider that grounds answers in retrieved context."""

    @property
    def name(self) -> str:
        return "mock"

    def generate(
        self,
        *,
        system: str,
        prompt: str,
        contexts: list[RetrievedContext] | None = None,
        question: str | None = None,
        behavior: dict[str, Any] | None = None,
        requires_json: bool = False,
        expected_json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        behavior = behavior or {}
        contexts = contexts or []
        question = question or ""

        verbosity = behavior.get("verbosity", "low")
        n_sentences = _VERBOSITY_SENTENCES.get(verbosity, 1)
        strict_json = bool(behavior.get("strict_json", False))

        ranked = self._rank_sentences(question, contexts)
        top = ranked[:n_sentences]

        used_chunk_ids = [chunk_id for _, chunk_id, _ in top]
        used_sources = []
        for _, _, doc in top:
            if doc not in used_sources:
                used_sources.append(doc)

        if requires_json:
            if strict_json:
                answer_text = json.dumps(
                    {
                        "answer": " ".join(s for s, _, _ in top) or "unknown",
                        "sources": used_sources,
                    }
                )
            else:
                # v1: ignores the JSON instruction and answers in prose -> invalid JSON.
                answer_text = (
                    " ".join(s for s, _, _ in top) or "I could not find this in the documentation."
                )
        else:
            answer_text = (
                " ".join(s for s, _, _ in top) or "I could not find this in the documentation."
            )

        prompt_tokens = estimate_tokens(system) + estimate_tokens(prompt)
        completion_tokens = estimate_tokens(answer_text)

        return LLMResponse(
            text=answer_text,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            raw={"used_chunk_ids": used_chunk_ids, "sources": used_sources},
        )

    @staticmethod
    def _rank_sentences(
        question: str, contexts: list[RetrievedContext]
    ) -> list[tuple[str, str, str]]:
        """Return (sentence, chunk_id, document_name) ranked by question overlap.

        Ties are broken by retrieval order then sentence text, keeping the output
        fully deterministic.
        """
        q_tokens = set(_tokenize(question))
        n = len(contexts)
        scored: list[tuple[float, int, str, str, str]] = []
        for order, ctx in enumerate(contexts):
            # Small retrieval-order bonus biases sentence selection toward the
            # most relevant document (chunk 0) only as a tiebreaker; token
            # overlap dominates so a strongly-matching sentence in a lower-ranked
            # chunk still wins.
            order_bonus = 0.25 * (n - order)
            for sentence in _split_sentences(ctx.text):
                s_tokens = set(_tokenize(sentence))
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens)
                # Length-normalised tiebreaker favours focused, on-topic sentences.
                score = overlap + order_bonus + 0.01 * (overlap / max(1, len(s_tokens)))
                scored.append((score, order, sentence, ctx.chunk_id, ctx.document_name))

        # Sort: higher score first, then earlier retrieval order, then text.
        scored.sort(key=lambda x: (-x[0], x[1], x[2]))
        return [(sentence, cid, doc) for _, _, sentence, cid, doc in scored]
