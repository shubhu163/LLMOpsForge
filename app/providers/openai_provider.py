"""Optional OpenAI provider stub.

This is intentionally a thin, lazy stub: it imports the ``openai`` SDK only when
instantiated and requires ``OPENAI_API_KEY``. It is never used by default and is
not needed for any test or local demo (use the mock provider for those).
"""

from __future__ import annotations

from typing import Any

from app.config import get_settings
from app.logging_config import get_logger
from app.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderError,
    RetrievedContext,
    estimate_tokens,
)

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """Calls the OpenAI Chat Completions API. Requires the ``openai`` extra."""

    @property
    def name(self) -> str:
        return "openai"

    def _client(self):
        settings = get_settings()
        if not settings.openai_api_key:
            raise ProviderError(
                "OPENAI_API_KEY is not set. The OpenAI provider is optional; "
                "use the default 'mock' provider for local, zero-cost runs."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                "The 'openai' package is not installed. Install with: pip install -e '.[openai]'"
            ) from exc
        return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

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
        client = self._client()
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if requires_json:
            kwargs["response_format"] = {"type": "json_object"}

        resp = client.chat.completions.create(**kwargs)  # pragma: no cover - network
        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", estimate_tokens(system + prompt)),
            completion_tokens=getattr(usage, "completion_tokens", estimate_tokens(text)),
            raw={"provider": "openai"},
        )
