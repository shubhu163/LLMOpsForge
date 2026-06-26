"""Tests for the deterministic MockLLMProvider."""

from __future__ import annotations

import json

from app.providers.base import RetrievedContext
from app.providers.mock import MockLLMProvider

CONTEXTS = [
    RetrievedContext(
        chunk_id="refund.md::chunk_0",
        document_name="refund.md",
        text="Customers may request a full refund within 30 days of purchase.",
        score=0.9,
    ),
    RetrievedContext(
        chunk_id="pricing.md::chunk_0",
        document_name="pricing.md",
        text="The Pro plan costs 49 dollars per user per month.",
        score=0.4,
    ),
]


def test_deterministic_output():
    p = MockLLMProvider("mock-small")
    r1 = p.generate(
        system="s", prompt="p", contexts=CONTEXTS, question="What is the refund period?"
    )
    r2 = p.generate(
        system="s", prompt="p", contexts=CONTEXTS, question="What is the refund period?"
    )
    assert r1.text == r2.text


def test_answer_is_grounded_in_context():
    p = MockLLMProvider("mock-small")
    r = p.generate(system="s", prompt="p", contexts=CONTEXTS, question="What is the refund period?")
    assert "30 days" in r.text
    assert "refund.md::chunk_0" in r.raw["used_chunk_ids"]


def test_strict_json_behavior_emits_valid_json():
    p = MockLLMProvider("mock-large")
    r = p.generate(
        system="s",
        prompt="p",
        contexts=CONTEXTS,
        question="What is the refund period?",
        behavior={"strict_json": True, "verbosity": "medium"},
        requires_json=True,
    )
    parsed = json.loads(r.text)
    assert "answer" in parsed and "sources" in parsed


def test_non_strict_json_returns_prose():
    p = MockLLMProvider("mock-small")
    r = p.generate(
        system="s",
        prompt="p",
        contexts=CONTEXTS,
        question="What is the refund period?",
        behavior={"strict_json": False, "verbosity": "low"},
        requires_json=True,
    )
    # v1-style: ignores JSON instruction -> not parseable as a JSON object.
    try:
        parsed = json.loads(r.text)
        assert not isinstance(parsed, dict)
    except json.JSONDecodeError:
        pass  # expected


def test_token_counts_present():
    p = MockLLMProvider("mock-small")
    r = p.generate(system="system text", prompt="prompt text", contexts=CONTEXTS, question="refund")
    assert r.prompt_tokens > 0
    assert r.completion_tokens > 0
    assert r.total_tokens == r.prompt_tokens + r.completion_tokens
