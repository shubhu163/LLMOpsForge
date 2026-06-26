"""End-to-end RAG pipeline: ingest documents and answer queries.

The pipeline owns the retriever (built from chunks persisted in the database) and
coordinates loading -> chunking -> embedding -> retrieval -> generation. The
index is rebuilt from the DB on demand, so ``query`` works in a fresh process
after a separate ``ingest`` run (e.g. CLI commands).
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.config import get_settings, load_model_config, load_prompt_template
from app.logging_config import get_logger
from app.providers.factory import build_provider
from app.rag.chunker import chunk_document
from app.rag.embeddings import build_embedder
from app.rag.generator import GenerationResult, generate_answer
from app.rag.loader import load_documents
from app.rag.retriever import IndexedChunk, VectorRetriever
from app.storage import models, repository

logger = get_logger(__name__)


class RagPipeline:
    """Coordinates ingestion and querying over a SQLAlchemy session."""

    def __init__(self, session: Session, *, embedder=None, prefer_real_embedder: bool = True):
        self.session = session
        self._embedder = embedder or build_embedder(
            get_settings().embedding_model, prefer_real=prefer_real_embedder
        )
        self._retriever: VectorRetriever | None = None

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #

    def ingest(
        self, docs_path: str, *, chunk_size: int = 600, chunk_overlap: int = 100
    ) -> dict[str, int]:
        """Load, chunk, and persist documents; rebuild the retrieval index."""
        loaded = load_documents(docs_path)
        total_chunks = 0
        for doc in loaded:
            db_doc = repository.upsert_document(
                self.session, name=doc.name, path=doc.path, content_hash=doc.content_hash
            )
            chunks = chunk_document(
                doc.name, doc.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            db_chunks = [
                models.DocumentChunk(
                    id=c.chunk_id,
                    document_id=db_doc.id,
                    document_name=c.document_name,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    char_start=c.char_start,
                    char_end=c.char_end,
                )
                for c in chunks
            ]
            repository.add_chunks(self.session, db_chunks)
            db_doc.num_chunks = len(db_chunks)
            total_chunks += len(db_chunks)

        self.session.commit()
        self._build_index()
        logger.info("Ingested %d documents / %d chunks", len(loaded), total_chunks)
        return {"documents": len(loaded), "chunks": total_chunks}

    # ------------------------------------------------------------------ #
    # Indexing
    # ------------------------------------------------------------------ #

    def _build_index(self) -> None:
        chunks = repository.all_chunks(self.session)
        indexed = [
            IndexedChunk(chunk_id=c.id, document_name=c.document_name, text=c.text) for c in chunks
        ]
        retriever = VectorRetriever(self._embedder)
        retriever.build(indexed)
        self._retriever = retriever

    def ensure_index(self) -> VectorRetriever:
        if self._retriever is None:
            self._build_index()
        assert self._retriever is not None
        return self._retriever

    @property
    def retriever(self) -> VectorRetriever:
        return self.ensure_index()

    # ------------------------------------------------------------------ #
    # Querying
    # ------------------------------------------------------------------ #

    def query(
        self,
        question: str,
        *,
        top_k: int = 4,
        prompt_template_id: str = "prompt_v1",
        model_config_id: str = "mock-small",
        require_citations: bool = True,
        requires_json: bool = False,
        expected_json_schema: dict[str, Any] | None = None,
        persist: bool = True,
    ) -> GenerationResult:
        """Answer a single question end-to-end."""
        retriever = self.ensure_index()
        contexts = retriever.query(question, top_k=top_k)

        template = load_prompt_template(prompt_template_id)
        model_spec = load_model_config(model_config_id)
        provider = build_provider(model_spec)

        result = generate_answer(
            provider=provider,
            template=template,
            model_spec=model_spec,
            question=question,
            contexts=contexts,
            require_citations=require_citations,
            requires_json=requires_json,
            expected_json_schema=expected_json_schema,
        )

        if persist:
            self.session.add(
                models.RagQuery(
                    question=question,
                    top_k=top_k,
                    prompt_template_id=prompt_template_id,
                    model_config_id=model_config_id,
                    model_name=result.model_name,
                    answer=result.answer,
                    citations=[c.to_dict() for c in result.citations],
                    retrieved_contexts=result.to_dict()["retrieved_contexts"],
                    latency_ms=result.latency_ms,
                    estimated_tokens=result.estimated_tokens,
                    estimated_cost=result.estimated_cost,
                )
            )
            self.session.commit()

        return result
