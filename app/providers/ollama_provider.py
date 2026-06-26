"""Optional Ollama provider stub for local open-weight models.

Talks to a locally running Ollama server (default ``http://localhost:11434``).
Never used by default; provided so the same RAG/eval harness can target a local
model with no code changes. Uses only the stdlib so it adds no dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
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


class OllamaProvider(BaseLLMProvider):
    """Calls a local Ollama server's ``/api/chat`` endpoint."""

    @property
    def name(self) -> str:
        return "ollama"

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
        base_url = get_settings().ollama_base_url.rstrip("/")
        payload = {
            "model": self.model_name,
            "stream": False,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        if requires_json:
            payload["format"] = "json"

        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:  # pragma: no cover - requires a running Ollama server
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ProviderError(
                f"Could not reach Ollama at {base_url}. Is the server running? "
                "The Ollama provider is optional; use 'mock' for local runs."
            ) from exc

        text = body.get("message", {}).get("content", "")
        return LLMResponse(
            text=text,
            model_name=self.model_name,
            prompt_tokens=body.get("prompt_eval_count", estimate_tokens(system + prompt)),
            completion_tokens=body.get("eval_count", estimate_tokens(text)),
            raw={"provider": "ollama"},
        )
