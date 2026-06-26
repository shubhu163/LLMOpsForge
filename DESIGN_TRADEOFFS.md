# Design Trade-offs

This document explains the *why* behind the key decisions — the part interviewers
and reviewers actually care about.

## Why `MockLLMProvider` is the default

**Goal: the entire system must run locally with zero paid API calls.**

A real provider as the default would make the project impossible to demo, test,
or CI without keys, network, and spend. Instead, the default `MockLLMProvider` is
**deterministic and grounded by construction**: it answers by extracting the
sentences from retrieved context that best overlap the question.

Consequences:

- **Tests are fast, free, and reproducible** — the same input always yields the
  same output, so metric assertions are stable.
- **The eval/regression machinery is exercised end-to-end** without a model — the
  harness is the product here, not the model.
- **Prompt behavior is still meaningfully variable**: the mock honours prompt
  flags (`verbosity`, `strict_json`, `always_cite`), so `prompt_v1` vs `prompt_v2`
  produce genuinely different metrics and a real regression signal.

The trade-off: the mock is not a language model, so it can't demonstrate
generation *quality*. That's acceptable — swapping in `OpenAIProvider` or
`OllamaProvider` requires no pipeline changes, and the evals are provider-agnostic.

## Why deterministic evals come first

LLM-as-judge scoring is powerful but **slow, costly, and non-deterministic** —
poor properties for a CI gate. Deterministic, rule-based metrics give a fast,
free, reproducible first line of defence that catches the majority of
regressions (broken JSON, missing citations, forbidden claims, lost retrieval).

The architecture keeps the door open: an `LLMJudge` protocol lets a semantic
judge be layered on as an optional second pass **without touching the runner**.
Start deterministic; add judgment where deterministic scoring is demonstrably
insufficient. See [`EVALS.md`](EVALS.md).

## Why FAISS (with a fallback)

FAISS is the de-facto standard for similarity search: fast, battle-tested, and
the natural choice an AI engineer would reach for. Using `IndexFlatIP` over
L2-normalised embeddings gives exact cosine top-k.

But a hard FAISS dependency would break the "runs anywhere, zero-config" promise
(faiss-cpu wheels, model downloads). So retrieval **degrades gracefully**:

| Component | Preferred | Fallback |
| --- | --- | --- |
| Embeddings | `sentence-transformers` | deterministic hashing embedder |
| Index | `faiss-cpu` | numpy cosine (exact) |

The fallbacks are exact and offline; the retriever reports which backend is
active. You get production-grade tooling when it's installed and a working system
when it isn't.

## Why character-based, paragraph-aware chunking

Token-based chunking is more "correct" but pulls in a tokenizer dependency.
Character chunking with paragraph awareness keeps policy clauses and sentences
intact (which matters for grounded retrieval and citation), is dependency-free,
and is easy to reason about. Chunk size/overlap are configurable per run.

## Why SQLite + SQLAlchemy

SQLite needs no server and lives in a single file — ideal for local dev and a
portable portfolio project. SQLAlchemy's typed 2.0 models give a clean schema and
a clear migration path: switching to Postgres is a connection-string change, with
no model rewrites. JSON columns store nested artifacts (citations, contexts,
metrics) without over-normalising a fast-moving schema.

## How this scales to production

The local design maps onto production components without architectural change:

| Local (this repo) | Production equivalent |
| --- | --- |
| SQLite | Postgres / managed RDS |
| In-process FAISS rebuild | Persistent vector DB (pgvector, Qdrant, Milvus, managed FAISS) |
| MockLLMProvider | OpenAI / Anthropic / self-hosted via the same `BaseLLMProvider` interface |
| Synchronous eval runner | Queue + workers (Celery / Arq / Ray) for large datasets |
| Hashing embedder | Batched GPU embedding service |
| Streamlit dashboard | Grafana / a web app over the same tables |
| Per-request index rebuild | Incremental index updates on ingest |

Because providers, storage, and retrieval are all behind interfaces, each can be
upgraded independently.

## How to add Langfuse / Promptfoo / RAGAS-style integrations later

The data model already captures what these tools consume:

- **Langfuse-style tracing/observability** — `RagQuery` and `EvalExampleResult`
  already persist latency, tokens, cost, retrieved contexts, and citations. A
  tracing exporter could mirror these as spans; the generator is the natural hook
  point for emitting traces.
- **Promptfoo-style declarative evals** — `configs/*.yaml` + `qa_eval.jsonl` are
  effectively a prompt/test matrix. A thin adapter could import/export Promptfoo
  test specs, and the regression comparator already provides the pass/fail
  gating Promptfoo offers.
- **RAGAS-style metrics** — faithfulness, answer relevancy, and context precision
  map onto the existing `grounding`, `answer_correctness`, and
  `retrieval_relevance` metrics. RAGAS's LLM-graded versions slot in behind the
  `LLMJudge` protocol as an optional scorer.

The point of these seams is that none of them require rearchitecting — they're
additive, because the boundaries (provider, metric, storage, report) were drawn
with extension in mind.
