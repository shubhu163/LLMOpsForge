"""SQLAlchemy ORM models for documents, queries, evaluations, and regressions."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.database import Base


def new_id() -> str:
    """Return a short unique id."""
    return uuid.uuid4().hex


def _utcnow() -> datetime:
    return datetime.utcnow()


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String, index=True)
    path: Mapped[str] = mapped_column(String, default="")
    content_hash: Mapped[str] = mapped_column(String, default="")
    num_chunks: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. refund_policy.md::chunk_0
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    document_name: Mapped[str] = mapped_column(String, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int] = mapped_column(Integer, default=0)
    char_end: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    system: Mapped[str] = mapped_column(Text, default="")
    instructions: Mapped[str] = mapped_column(Text, default="")
    behavior: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class ModelConfig(Base):
    __tablename__ = "model_configs"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    provider: Mapped[str] = mapped_column(String)
    model_name: Mapped[str] = mapped_column(String)
    temperature: Mapped[float] = mapped_column(Float, default=0.0)
    max_tokens: Mapped[int] = mapped_column(Integer, default=512)
    input_cost_per_1k: Mapped[float] = mapped_column(Float, default=0.0)
    output_cost_per_1k: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class RagQuery(Base):
    __tablename__ = "rag_queries"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    question: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer, default=4)
    prompt_template_id: Mapped[str] = mapped_column(String, default="")
    model_config_id: Mapped[str] = mapped_column(String, default="")
    model_name: Mapped[str] = mapped_column(String, default="")
    answer: Mapped[str] = mapped_column(Text, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_contexts: Mapped[list] = mapped_column(JSON, default=list)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String, default="")
    dataset_path: Mapped[str] = mapped_column(String, default="")
    prompt_template_id: Mapped[str] = mapped_column(String, default="")
    model_config_id: Mapped[str] = mapped_column(String, default="")
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    num_examples: Mapped[int] = mapped_column(Integer, default=0)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    results: Mapped[list[EvalExampleResult]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class EvalExampleResult(Base):
    __tablename__ = "eval_example_results"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.id", ondelete="CASCADE"))
    task_id: Mapped[str] = mapped_column(String, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    difficulty: Mapped[str] = mapped_column(String, default="")
    citations: Mapped[list] = mapped_column(JSON, default=list)
    retrieved_contexts: Mapped[list] = mapped_column(JSON, default=list)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    hallucination_flag: Mapped[bool] = mapped_column(Boolean, default=False)
    json_validity: Mapped[str] = mapped_column(String, default="n/a")  # pass | fail | n/a
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    estimated_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[EvalRun] = relationship(back_populates="results")


class RegressionReport(Base):
    __tablename__ = "regression_reports"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=new_id)
    baseline_run_id: Mapped[str] = mapped_column(String, index=True)
    candidate_run_id: Mapped[str] = mapped_column(String, index=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    recommendation: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
