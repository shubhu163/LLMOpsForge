"""LLM provider abstraction.

All providers implement :class:`BaseLLMProvider`. The generator passes the
assembled prompt plus *structured* context (retrieved chunks, the question, and
prompt-behavior flags). Real providers ignore the structured extras and use the
text prompt; the deterministic :class:`MockLLMProvider` uses them to synthesise
grounded, reproducible answers without any network call.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

# Rough heuristic: ~4 characters per token. Good enough for deterministic,
# provider-agnostic cost/token estimation in a local eval harness.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length."""
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))


@dataclass
class RetrievedContext:
    """A retrieved chunk handed to the provider/generator."""

    chunk_id: str
    document_name: str
    text: str
    score: float = 0.0


@dataclass
class LLMResponse:
    """Normalised provider response."""

    text: str
    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ProviderError(RuntimeError):
    """Raised when a provider cannot fulfil a request."""


class BaseLLMProvider(abc.ABC):
    """Interface implemented by every LLM provider."""

    def __init__(self, model_name: str, *, temperature: float = 0.0, max_tokens: int = 512):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short provider identifier, e.g. ``mock``."""

    @abc.abstractmethod
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
        """Generate a completion. Implementations must be side-effect free."""
        raise NotImplementedError
