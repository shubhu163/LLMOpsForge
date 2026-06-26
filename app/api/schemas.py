"""Pydantic request/response schemas for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    provider: str
    documents: int
    chunks: int


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestRequest(BaseModel):
    docs_path: str = Field(..., description="File or directory path to ingest")
    chunk_size: int = 600
    chunk_overlap: int = 100


class IngestResponse(BaseModel):
    documents: int
    chunks: int


# --------------------------------------------------------------------------- #
# RAG query
# --------------------------------------------------------------------------- #


class CitationModel(BaseModel):
    chunk_id: str
    document_name: str
    score: float = 0.0


class RetrievedContextModel(BaseModel):
    chunk_id: str
    document_name: str
    text: str
    score: float = 0.0


class RagQueryRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    question: str
    top_k: int = 4
    prompt_template_id: str = "prompt_v1"
    model_config_id: str = "mock-small"
    require_citations: bool = True
    requires_json: bool = False
    expected_json_schema: dict[str, Any] | None = None


class RagQueryResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    answer: str
    citations: list[CitationModel]
    retrieved_contexts: list[RetrievedContextModel]
    latency_ms: float
    estimated_tokens: int
    estimated_cost: float
    model_name: str
    prompt_template_id: str


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


class EvalRunRequest(BaseModel):
    dataset_path: str = "datasets/qa_eval.jsonl"
    config_path: str = "configs/default.yaml"
    name: str | None = None


class EvalRunResponse(BaseModel):
    eval_run_id: str
    name: str
    num_examples: int
    summary: dict[str, Any]


class EvalRunDetailResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    eval_run_id: str
    name: str
    dataset_path: str
    prompt_template_id: str
    model_config_id: str
    num_examples: int
    summary: dict[str, Any]
    results: list[dict[str, Any]]


class RegressionRequest(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


class RegressionResponse(BaseModel):
    regression_report_id: str
    recommendation: str
    summary: dict[str, Any]


class MetricsSummaryResponse(BaseModel):
    total_eval_runs: int
    latest_run: dict[str, Any] | None
    documents: int
    chunks: int
