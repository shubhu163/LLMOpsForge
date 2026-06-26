"""Provider factory: build a provider instance from a model config spec."""

from __future__ import annotations

from app.config import ModelConfigSpec
from app.providers.base import BaseLLMProvider, ProviderError
from app.providers.mock import MockLLMProvider


def build_provider(spec: ModelConfigSpec) -> BaseLLMProvider:
    """Instantiate the provider named by ``spec.provider``.

    Optional providers are imported lazily so the core install never depends on
    ``openai`` or a running Ollama server.
    """
    provider = spec.provider.lower()
    kwargs = {
        "model_name": spec.model_name,
        "temperature": spec.temperature,
        "max_tokens": spec.max_tokens,
    }

    if provider == "mock":
        return MockLLMProvider(**kwargs)
    if provider == "openai":
        from app.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(**kwargs)
    if provider == "ollama":
        from app.providers.ollama_provider import OllamaProvider

        return OllamaProvider(**kwargs)
    if provider == "openrouter":
        from app.providers.openrouter_provider import OpenRouterProvider

        return OpenRouterProvider(**kwargs)

    raise ProviderError(
        f"Unknown provider '{spec.provider}'. Supported: mock, openai, ollama, openrouter."
    )
