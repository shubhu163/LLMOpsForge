"""Embeddings with a real model when available and a deterministic fallback.

If ``sentence-transformers`` is installed (the ``vectors`` extra) the configured
model is used. Otherwise we fall back to a deterministic hashing embedder built
on token n-grams. The fallback is fully offline, requires no downloads, and is
good enough for keyword-grounded retrieval — which keeps the whole project
runnable and testable with zero external dependencies.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from app.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbedder:
    """Deterministic bag-of-hashed-ngrams embedder. No model download required."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    @property
    def name(self) -> str:
        return f"hashing-{self.dim}"

    def _tokens(self, text: str) -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        grams = list(words)
        # Add character 3-grams of each word for sub-word robustness.
        for w in words:
            grams.extend(w[i : i + 3] for i in range(max(0, len(w) - 2)))
        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in self._tokens(text):
                h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:8], "little")
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                vecs[i, idx] += sign
        return _l2_normalize(vecs)


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model (optional dependency)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self._model_name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(
    model_name: str, *, prefer_real: bool = True
) -> HashingEmbedder | SentenceTransformerEmbedder:
    """Return the best available embedder.

    Falls back to the deterministic hashing embedder if sentence-transformers is
    not installed or the model cannot be loaded.
    """
    if prefer_real:
        try:
            embedder = SentenceTransformerEmbedder(model_name)
            logger.info("Using sentence-transformers embedder: %s", model_name)
            return embedder
        except Exception as exc:  # ImportError or model-load/network failure
            logger.warning(
                "sentence-transformers unavailable (%s); using deterministic "
                "hashing embedder. Install the 'vectors' extra for real embeddings.",
                type(exc).__name__,
            )
    return HashingEmbedder()
