"""Tests for the FastAPI endpoints via the TestClient."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(temp_env):
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"


def test_ingest_then_query(client):
    ingest = client.post("/documents/ingest", json={"docs_path": "documents/"})
    assert ingest.status_code == 200
    assert ingest.json()["documents"] >= 5

    query = client.post(
        "/rag/query",
        json={
            "question": "How many days do customers have to request a full refund?",
            "prompt_template_id": "prompt_v2",
        },
    )
    assert query.status_code == 200
    data = query.json()
    assert "30 days" in data["answer"]
    assert data["citations"]
    assert data["model_name"] == "mock-small"


def test_query_without_documents_returns_409(client):
    resp = client.post("/rag/query", json={"question": "anything"})
    assert resp.status_code == 409


def test_full_eval_flow(client):
    client.post("/documents/ingest", json={"docs_path": "documents/"})
    run = client.post(
        "/evals/run",
        json={"dataset_path": "datasets/qa_eval.jsonl", "config_path": "configs/default.yaml"},
    )
    assert run.status_code == 200
    run_id = run.json()["eval_run_id"]
    assert run.json()["num_examples"] >= 20

    detail = client.get(f"/evals/{run_id}")
    assert detail.status_code == 200
    assert len(detail.json()["results"]) >= 20

    report = client.get(f"/evals/{run_id}/report")
    assert report.status_code == 200
    assert "# Evaluation Report" in report.json()["report"]

    regression = client.post(
        "/evals/regression",
        json={"baseline_run_id": run_id, "candidate_run_id": run_id},
    )
    assert regression.status_code == 200
    assert regression.json()["recommendation"] == "safe to ship"

    summary = client.get("/metrics/summary")
    assert summary.status_code == 200
    assert summary.json()["total_eval_runs"] >= 1


def test_eval_run_not_found(client):
    assert client.get("/evals/nope").status_code == 404
