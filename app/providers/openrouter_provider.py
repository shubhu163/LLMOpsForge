"""OpenRouter provider — real LLMs through an OpenAI-compatible gateway.

`OpenRouter <https://openrouter.ai>`_ exposes hundreds of models (OpenAI, Anthropic,
Llama, Gemini, plus several **free** models) behind a single OpenAI-compatible
``/chat/completions`` API. Because it speaks the OpenAI protocol, we reuse the
``openai`` SDK and just point it at OpenRouter's base URL with an
``OPENROUTER_API_KEY``.

This is the recommended way to run *real* RAG with this project: one key, many
models, and free options for zero-cost testing.
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


class OpenRouterProvider(BaseLLMProvider):
    """Calls OpenRouter's OpenAI-compatible chat API. Requires the ``openai`` SDK."""

    @property
    def name(self) -> str:
        return "openrouter"

    def _client(self):
        settings = get_settings()
        if not settings.openrouter_api_key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Get a free key at "
                "https://openrouter.ai/keys, then put it in your .env file."
            )
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ProviderError(
                "The 'openai' package is not installed. Install with: pip install -e '.[openrouter]'"
            ) from exc

        # Optional ranking headers OpenRouter recommends (safe to omit).
        default_headers = {}
        if settings.openrouter_referer:
            default_headers["HTTP-Referer"] = settings.openrouter_referer
        if settings.openrouter_title:
            default_headers["X-Title"] = settings.openrouter_title

        return OpenAI(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            default_headers=default_headers or None,
        )

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
        # Many OpenRouter models honour JSON mode; harmless for those that ignore it.
        if requires_json:
            kwargs["response_format"] = {"type": "json_object"}

        try:  # pragma: no cover - network
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:  # surface a clean error for the CLI/API
            raise ProviderError(f"OpenRouter request failed: {exc}") from exc

        text = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            prompt_tokens=getattr(usage, "prompt_tokens", None) or estimate_tokens(system + prompt),
            completion_tokens=getattr(usage, "completion_tokens", None) or estimate_tokens(text),
            raw={"provider": "openrouter", "model": self.model_name},
        )
