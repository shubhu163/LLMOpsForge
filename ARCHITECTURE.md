# Architecture

LLMOpsForge is organised into clear layers with one-directional dependencies:
**interfaces → orchestration (RAG / eval) → providers + storage**. Nothing in the
core depends on a paid API or a network connection.

```
Interfaces (CLI / API / Dashboard)
        │
        ▼
RAG pipeline  ──►  LLM providers
        │
        ▼
Evaluation engine
        │
        ▼
Storage (SQLAlchemy + SQLite)
```

## FastAPI layer (`app/api`, `app/main.py`)

A thin REST surface over the pipeline and eval engine.

- `schemas.py` — Pydantic request/response models (validation + OpenAPI docs).
- `routes.py` — one handler per endpoint; translates domain errors into proper
  HTTP status codes (`400` bad input, `404` not found, `409` no documents).
- `main.py` — builds the `FastAPI` app and, on startup (`lifespan`), initialises
  the database schema and seeds prompt templates / model configs from YAML.

Sessions are provided per-request via the `get_db` dependency.

## RAG pipeline (`app/rag`)

A small, readable RAG implementation:

| Module | Responsibility |
| --- | --- |
| `loader.py` | Load `.txt` / `.md` files; hash content; skip unsupported types |
| `chunker.py` | Paragraph-aware, character-based chunking with overlap; stable `chunk_id`s (`doc::chunk_N`) |
| `embeddings.py` | `sentence-transformers` embedder **or** deterministic hashing fallback |
| `retriever.py` | FAISS `IndexFlatIP` **or** numpy cosine fallback; returns top-k |
| `generator.py` | Assemble prompt → call provider → measure latency → compute cost |
| `citations.py` | Map answers back to `(document_name, chunk_id)` sources |
| `pipeline.py` | Orchestrates ingest + query; rebuilds the index from the DB on demand |

The pipeline **rebuilds its retrieval index from persisted chunks on demand**, so
a `query` works in a fresh process after a separate `ingest` (important for the
CLI, where each command is its own process).

## FAISS retrieval

Embeddings are L2-normalised, so inner product equals cosine similarity. When
`faiss-cpu` is installed, a flat inner-product index serves exact top-k search;
otherwise an equivalent numpy matrix multiply is used. The retriever reports its
active backend (`faiss` or `numpy`) for transparency.

## Provider abstraction (`app/providers`)

```
BaseLLMProvider (ABC)
├── MockLLMProvider     # deterministic, grounded, zero-cost (DEFAULT)
├── OpenAIProvider      # optional stub (needs OPENAI_API_KEY + openai extra)
└── OllamaProvider      # optional stub (needs a local Ollama server)
```

`build_provider(spec)` (`factory.py`) instantiates the right provider from a
model config; optional providers are imported lazily so the core install never
depends on `openai` or a running Ollama server.

The generator hands providers both the assembled text prompt **and** structured
context (retrieved chunks, the question, prompt-behavior flags). Real providers
use the text; the mock uses the structure to synthesise grounded, reproducible
answers — see [`DESIGN_TRADEOFFS.md`](DESIGN_TRADEOFFS.md).

## Evaluation engine (`app/evals`)

| Module | Responsibility |
| --- | --- |
| `metrics.py` | Pure, deterministic metric functions + per-example pass/fail; optional `LLMJudge` protocol |
| `runner.py` | Load JSONL dataset → run pipeline per task → score → persist `EvalRun` + results → aggregate summary |
| `regression.py` | Compare two runs: metric deltas, newly-failed / fixed tasks, critical hallucination regressions, verdict |
| `reports.py` | Render Markdown + JSON reports for runs and regressions |

Each example produces a `MetricResult` with all metrics, an explainable `detail`
dict, and the final `passed` boolean (computed against configurable thresholds).

## Regression comparator

`compare_runs` joins two runs on shared task ids and produces:

- a **delta table** for every summary metric (with "higher/lower is better"
  awareness and percent change),
- **newly failed**, **fixed**, and **improved** task lists,
- **critical regressions** (tasks that newly hallucinate), and
- a final **recommendation**: `safe to ship` / `investigate` / `blocked`.

## Storage layer (`app/storage`)

SQLAlchemy 2.0 (typed `Mapped` models) over SQLite for local dev.

Models: `Document`, `DocumentChunk`, `PromptTemplate`, `ModelConfig`, `RagQuery`,
`EvalRun`, `EvalExampleResult`, `RegressionReport`.

- `database.py` — engine/session lifecycle, `init_db`, `session_scope`, the
  `get_db` FastAPI dependency, and a `reset_engine` hook used by tests.
- `repository.py` — explicit data-access helpers (seeding, upserts, queries) so
  persistence logic lives in one place.

JSON columns store nested artifacts (citations, retrieved contexts, per-example
metrics, regression summaries) to keep the schema simple while remaining queryable.

## Dashboard (`dashboard/streamlit_app.py`)

A read-only Streamlit app over the same SQLite database. It detaches ORM rows
into plain dicts before rendering so sessions close promptly. Five pages cover
summary metrics, failed examples, A/B prompt comparison, regression reports, and
retrieval inspection.

## Configuration (`app/config.py`, `configs/`)

`Settings` (pydantic-settings, `LLMOPS_` env prefix) holds runtime config with
local-first defaults. YAML loaders read prompt templates, model configs, and eval
run configs from `configs/`, decoupling behavior from code.
