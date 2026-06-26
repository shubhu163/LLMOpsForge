"""Adapters that let the eval harness assess ANY RAG system, not just the built-in one.

The evaluation engine only needs a RAG system that, given a question, returns an
``answer`` plus the ``retrieved_contexts`` and ``citations`` it used. Anything that
implements :class:`RagSystem` (a ``query(...) -> GenerationResult``) can be graded
by the exact same metrics, regression comparator, reports, and dashboard.

Provided adapters:
* :class:`FunctionRagAdapter` — wrap any Python callable (e.g. a LangChain /
  LlamaIndex chain) that returns ``{answer, contexts, citations}``.
* :class:`HttpRagAdapter`    — POST the question to an HTTP endpoint exposed by an
  external RAG app, and grade its JSON response.

This is how you point LLMOpsForge at a "famous RAG pipeline on GitHub": expose it
behind a tiny HTTP endpoint (or import its callable) and assess it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from app.providers.base import RetrievedContext
from app.rag.citations import Citation
from app.rag.generator import GenerationResult


class RagSystem(Protocol):
    """Anything the eval runner can drive. The built-in RagPipeline satisfies this."""

    def query(self, question: str, **kwargs: Any) -> GenerationResult: ...


def _coerce_contexts(raw: Any) -> list[RetrievedContext]:
    """Accept contexts as list[str] or list[dict] and normalise to RetrievedContext."""
    contexts: list[RetrievedContext] = []
    for i, item in enumerate(raw or []):
        if isinstance(item, str):
            contexts.append(RetrievedContext(chunk_id=f"ctx_{i}", document_name="", text=item))
        elif isinstance(item, dict):
            contexts.append(
                RetrievedContext(
                    chunk_id=str(item.get("chunk_id", f"ctx_{i}")),
                    document_name=str(item.get("document_name", item.get("source", ""))),
                    text=str(item.get("text", item.get("content", ""))),
                    score=float(item.get("score", 0.0)),
                )
            )
    return contexts


def _coerce_citations(raw: Any, contexts: list[RetrievedContext]) -> list[Citation]:
    """Accept citations as list[str] (doc names) or list[dict]; default to contexts."""
    if raw is None:
        # No explicit citations -> attribute to the retrieved docs.
        seen: list[Citation] = []
        for c in contexts:
            if c.document_name:
                seen.append(
                    Citation(chunk_id=c.chunk_id, document_name=c.document_name, score=c.score)
                )
        return seen
    citations: list[Citation] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            citations.append(Citation(chunk_id=f"cite_{i}", document_name=item))
        elif isinstance(item, dict):
            citations.append(
                Citation(
                    chunk_id=str(item.get("chunk_id", f"cite_{i}")),
                    document_name=str(item.get("document_name", item.get("source", ""))),
                    score=float(item.get("score", 0.0)),
                )
            )
    return citations


class BaseRagAdapter:
    """Base adapter: subclasses implement :meth:`answer` returning a dict.

    The dict must contain ``answer`` and may contain ``retrieved_contexts`` and
    ``citations``. This base handles timing and conversion to a GenerationResult.
    """

    name: str = "external-rag"

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError

    def query(self, question: str, **kwargs: Any) -> GenerationResult:
        start = time.perf_counter()
        payload = self.answer(question, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000.0

        contexts = _coerce_contexts(payload.get("retrieved_contexts") or payload.get("contexts"))
        citations = _coerce_citations(payload.get("citations"), contexts)
        return GenerationResult(
            answer=str(payload.get("answer", "")),
            citations=citations,
            retrieved_contexts=contexts,
            latency_ms=round(latency_ms, 3),
            estimated_tokens=int(payload.get("estimated_tokens", 0)),
            estimated_cost=float(payload.get("estimated_cost", 0.0)),
            model_name=str(payload.get("model_name", self.name)),
            prompt_template_id=str(payload.get("prompt_template_id", "external")),
        )


class FunctionRagAdapter(BaseRagAdapter):
    """Wrap a Python callable ``fn(question) -> dict`` as a RAG system under test."""

    def __init__(self, fn, *, name: str = "function-rag"):
        self._fn = fn
        self.name = name

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        result = self._fn(question)
        if isinstance(result, str):
            return {"answer": result}
        return dict(result)


class HttpRagAdapter(BaseRagAdapter):
    """POST questions to an external RAG HTTP endpoint and grade its response.

    Expected response JSON (lenient):
        {
          "answer": "...",
          "retrieved_contexts": [{"document_name": "...", "chunk_id": "...", "text": "..."}],
          "citations": [{"document_name": "...", "chunk_id": "..."}]   # optional
        }
    """

    def __init__(self, url: str, *, name: str | None = None, timeout: int = 60):
        self.url = url
        self.name = name or f"http:{url}"
        self.timeout = timeout

    def answer(self, question: str, **kwargs: Any) -> dict[str, Any]:
        body = json.dumps({"question": question, "top_k": kwargs.get("top_k", 4)}).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach external RAG at {self.url}: {exc}") from exc
