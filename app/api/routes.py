"""FastAPI routes for ingestion, querying, evaluation, regression, and metrics."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app import __version__
from app.api import schemas
from app.config import get_settings, load_eval_config
from app.evals.regression import compare_runs
from app.evals.reports import eval_report_markdown
from app.evals.runner import EvalRunner
from app.logging_config import get_logger
from app.rag.pipeline import RagPipeline
from app.storage import repository
from app.storage.database import get_db

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health", response_model=schemas.HealthResponse, tags=["system"])
def health(db: Session = Depends(get_db)) -> schemas.HealthResponse:
    return schemas.HealthResponse(
        status="ok",
        version=__version__,
        provider=get_settings().default_provider,
        documents=repository.count_documents(db),
        chunks=repository.count_chunks(db),
    )


@router.post("/documents/ingest", response_model=schemas.IngestResponse, tags=["documents"])
def ingest_documents(
    req: schemas.IngestRequest, db: Session = Depends(get_db)
) -> schemas.IngestResponse:
    pipeline = RagPipeline(db)
    try:
        result = pipeline.ingest(
            req.docs_path, chunk_size=req.chunk_size, chunk_overlap=req.chunk_overlap
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.IngestResponse(**result)


@router.post("/rag/query", response_model=schemas.RagQueryResponse, tags=["rag"])
def rag_query(
    req: schemas.RagQueryRequest, db: Session = Depends(get_db)
) -> schemas.RagQueryResponse:
    pipeline = RagPipeline(db)
    if pipeline.retriever.size == 0:
        raise HTTPException(
            status_code=409, detail="No documents ingested. POST /documents/ingest first."
        )
    try:
        result = pipeline.query(
            req.question,
            top_k=req.top_k,
            prompt_template_id=req.prompt_template_id,
            model_config_id=req.model_config_id,
            require_citations=req.require_citations,
            requires_json=req.requires_json,
            expected_json_schema=req.expected_json_schema,
        )
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.RagQueryResponse(**result.to_dict())


@router.post("/evals/run", response_model=schemas.EvalRunResponse, tags=["evals"])
def run_eval(req: schemas.EvalRunRequest, db: Session = Depends(get_db)) -> schemas.EvalRunResponse:
    try:
        config = load_eval_config(req.config_path)
        run = EvalRunner(db).run(dataset_path=req.dataset_path, config=config, name=req.name)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return schemas.EvalRunResponse(
        eval_run_id=run.id, name=run.name, num_examples=run.num_examples, summary=run.summary
    )


@router.get("/evals/{eval_run_id}", response_model=schemas.EvalRunDetailResponse, tags=["evals"])
def get_eval(eval_run_id: str, db: Session = Depends(get_db)) -> schemas.EvalRunDetailResponse:
    run = repository.get_eval_run(db, eval_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Eval run not found: {eval_run_id}")
    results = repository.get_results_for_run(db, eval_run_id)
    return schemas.EvalRunDetailResponse(
        eval_run_id=run.id,
        name=run.name,
        dataset_path=run.dataset_path,
        prompt_template_id=run.prompt_template_id,
        model_config_id=run.model_config_id,
        num_examples=run.num_examples,
        summary=run.summary,
        results=[
            {
                "task_id": r.task_id,
                "question": r.question,
                "answer": r.answer,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "hallucination_flag": r.hallucination_flag,
                "json_validity": r.json_validity,
                "citations": r.citations,
                "metrics": r.metrics,
            }
            for r in results
        ],
    )


@router.get("/evals/{eval_run_id}/report", tags=["evals"])
def get_eval_report(eval_run_id: str, db: Session = Depends(get_db)) -> dict[str, str]:
    try:
        markdown = eval_report_markdown(db, eval_run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"eval_run_id": eval_run_id, "format": "markdown", "report": markdown}


@router.post("/evals/regression", response_model=schemas.RegressionResponse, tags=["evals"])
def run_regression(
    req: schemas.RegressionRequest, db: Session = Depends(get_db)
) -> schemas.RegressionResponse:
    try:
        report = compare_runs(
            db, baseline_run_id=req.baseline_run_id, candidate_run_id=req.candidate_run_id
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return schemas.RegressionResponse(
        regression_report_id=report.id,
        recommendation=report.recommendation,
        summary=report.summary,
    )


@router.get("/metrics/summary", response_model=schemas.MetricsSummaryResponse, tags=["metrics"])
def metrics_summary(db: Session = Depends(get_db)) -> schemas.MetricsSummaryResponse:
    runs = repository.list_eval_runs(db, limit=1)
    latest = None
    if runs:
        latest = {
            "eval_run_id": runs[0].id,
            "name": runs[0].name,
            "summary": runs[0].summary,
            "created_at": runs[0].created_at.isoformat(),
        }
    return schemas.MetricsSummaryResponse(
        total_eval_runs=len(repository.list_eval_runs(db, limit=1000)),
        latest_run=latest,
        documents=repository.count_documents(db),
        chunks=repository.count_chunks(db),
    )
