"""Vector retriever backed by FAISS, with a numpy cosine fallback.

The retriever indexes chunk embeddings and returns the top-k chunks for a query.
If ``faiss`` is installed (the ``vectors`` extra) it is used for the index;
otherwise an exact numpy inner-product search is used. Because embeddings are
L2-normalised, inner product == cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.logging_config import get_logger
from app.providers.base import RetrievedContext

logger = get_logger(__name__)


@dataclass
class IndexedChunk:
    chunk_id: str
    document_name: str
    text: str


class VectorRetriever:
    """Builds an index over chunk embeddings and serves top-k similarity queries."""

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        self._chunks: list[IndexedChunk] = []
        self._matrix: np.ndarray | None = None
        self._faiss_index = None
        self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def size(self) -> int:
        return len(self._chunks)

    def build(self, chunks: list[IndexedChunk]) -> None:
        """Embed and index the given chunks. Replaces any existing index."""
        self._chunks = list(chunks)
        if not chunks:
            self._matrix = None
            self._faiss_index = None
            return

        embeddings = self._embedder.embed([c.text for c in chunks]).astype(np.float32)
        self._matrix = embeddings

        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            self._faiss_index = index
            self._backend = "faiss"
            logger.info("Built FAISS index over %d chunks", len(chunks))
        except Exception as exc:
            self._faiss_index = None
            self._backend = "numpy"
            logger.info(
                "FAISS unavailable (%s); using numpy cosine index over %d chunks",
                type(exc).__name__,
                len(chunks),
            )

    def query(self, question: str, top_k: int = 4) -> list[RetrievedContext]:
        """Return the top-k most similar chunks to the question."""
        if not self._chunks or self._matrix is None:
            return []
        top_k = max(1, min(top_k, len(self._chunks)))
        q = self._embedder.embed([question]).astype(np.float32)

        if self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(q, top_k)
            pairs = zip(idxs[0].tolist(), scores[0].tolist(), strict=False)
        else:
            sims = self._matrix @ q[0]
            order = np.argsort(-sims)[:top_k]
            pairs = ((int(i), float(sims[i])) for i in order)

        results: list[RetrievedContext] = []
        for idx, score in pairs:
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                RetrievedContext(
                    chunk_id=chunk.chunk_id,
                    document_name=chunk.document_name,
                    text=chunk.text,
                    score=float(score),
                )
            )
        return results
