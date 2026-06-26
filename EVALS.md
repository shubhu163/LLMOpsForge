# Evaluation Metrics

All metrics in LLMOpsForge are **deterministic and rule-based** — no LLM calls,
no randomness. This makes the suite fast, free, reproducible, and safe to gate CI
on. Each metric returns a score **and** an explainable `detail` payload so you can
see *why* a score was assigned.

The dataset lives in `datasets/qa_eval.jsonl`; each task has:

```json
{
  "id": "refund_001",
  "question": "What is the refund period for a purchase?",
  "expected_answer": "30 days",
  "expected_sources": ["refund_policy.md"],
  "answer_keywords": ["30 days", "refund"],
  "forbidden_claims": ["90 days", "no refunds"],
  "requires_json": false,
  "expected_json_schema": null,
  "difficulty": "easy"
}
```

---

## 1. Answer correctness (`answer_correctness_score`, 0–1)

Combines two signals, averaged:

- **Keyword coverage** — fraction of `answer_keywords` present in the answer.
- **Exact expected answer** — `1.0` if `expected_answer` appears in the answer,
  else `0.0`.

If the answer is valid JSON, its string *values* are used for matching (so
JSON-mode answers are scored on content, not structural keys).

> Example: keywords `["30 days", "refund"]` both present **and** `"30 days"`
> found ⇒ `(1.0 + 1.0) / 2 = 1.0`.

## 2. Citation correctness (`citation_correctness_score`, 0–1)

Fraction of `expected_sources` that appear among the **cited** document names. If
a task lists no expected sources, the score is `1.0` (nothing required).

## 3. Grounding (`grounding_score`, 0–1)

Fraction of the answer's content tokens that also appear in the **retrieved
context**. High grounding means the answer is supported by what was retrieved;
low grounding suggests the model is drawing on outside (potentially fabricated)
information.

## 4. Hallucination flag (`hallucination_flag`, bool)

`True` when **either**:

- the answer contains any `forbidden_claims` substring (a known-wrong assertion), **or**
- `grounding_score` falls below a floor (default `0.5`), i.e. the answer is
  largely unsupported by retrieved context.

The `detail` records exactly which forbidden claims hit and whether the
low-grounding rule triggered.

## 5. Retrieval relevance (`retrieval_relevance_score`, 0–1)

Fraction of `expected_sources` that appear among the **retrieved** documents
(independent of whether they were cited). This isolates retrieval quality from
generation quality — a low score here means the retriever, not the model, is the
problem.

## 6. JSON validity (`json_validity`, pass/fail/n/a)

- `n/a` when the task does not require JSON.
- `pass` when the answer parses as JSON **and** (if provided) validates against
  `expected_json_schema` (JSON Schema Draft 7).
- `fail` otherwise.

This is where `prompt_v1` (prose) and `prompt_v2` (strict JSON) diverge sharply.

## 7. Latency / tokens / cost

- `latency_ms` — wall-clock time of the generation call.
- `estimated_tokens` — prompt + completion tokens (≈ 4 chars/token heuristic, or
  provider-reported usage when available).
- `estimated_cost_usd` — `tokens/1000 × per-1k rate` from the model config. The
  mock models are priced at `$0`, so local runs are free.

## Final pass/fail

An example **passes** only if all of the following hold (thresholds are
configurable per run in the eval config YAML):

```
error_count == 0
answer_correctness_score   >= thresholds.answer_correctness_score
citation_correctness_score >= thresholds.citation_correctness_score
grounding_score            >= thresholds.grounding_score
retrieval_relevance_score  >= thresholds.retrieval_relevance_score
(allow_hallucination OR not hallucination_flag)
json_validity != "fail"
```

---

## Limitations of deterministic scoring

Rule-based metrics are fast and reproducible but **not semantic**:

- **Keyword/substring matching** misses correct answers phrased differently
  (paraphrases, synonyms, numbers written as words).
- **Token-overlap grounding** can be fooled by lexical overlap that isn't truly
  supportive, and penalises valid abstraction/summarisation.
- **No reasoning-quality or fluency assessment** — only surface signals.
- Designed for **factual, extractive QA** over a known corpus; open-ended
  generation needs richer judging.

These are acceptable trade-offs for a fast, free, CI-friendly baseline — and they
motivate the extension below.

## Future: LLM-as-judge

`metrics.py` defines an optional `LLMJudge` protocol:

```python
class LLMJudge(Protocol):
    def score(self, task: EvalTask, answer: str, contexts: list[str]) -> float: ...
```

A judge implementation (e.g. backed by a strong model) can be layered in to score
semantic correctness, faithfulness, and answer relevancy — **without changing the
runner**. Deterministic metrics remain the fast first line; the judge becomes an
optional, higher-fidelity second pass. This mirrors how RAGAS-style faithfulness
/ answer-relevancy metrics could be integrated later.
