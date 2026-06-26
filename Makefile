.PHONY: install install-all test lint fmt run-api eval ingest dashboard docker-build docker-up clean

PY ?= python
PORT ?= 8000

install:
	$(PY) -m pip install -e ".[dev]"

install-all:
	$(PY) -m pip install -e ".[dev,vectors,dashboard,openai]"

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check app tests
	$(PY) -m ruff format --check app tests

fmt:
	$(PY) -m ruff format app tests
	$(PY) -m ruff check --fix app tests

run-api:
	$(PY) -m uvicorn app.main:app --reload --port $(PORT)

ingest:
	$(PY) -m app.cli ingest --docs-path documents/

eval:
	$(PY) -m app.cli eval --dataset datasets/qa_eval.jsonl --config configs/default.yaml

dashboard:
	$(PY) -m streamlit run dashboard/streamlit_app.py

docker-build:
	docker build -t llmopsforge:latest .

docker-up:
	docker compose up --build

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ *.db data/*.db data/index
	find . -type d -name __pycache__ -exec rm -rf {} +
