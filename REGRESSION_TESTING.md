# Regression Testing

Regression testing answers the question every team shipping LLM changes needs to
answer: **"Is this candidate (new prompt / new model) actually better, or did it
quietly break something?"**

LLMOpsForge compares a **candidate** evaluation run against a **baseline** run and
emits a structured verdict.

## Workflow

```bash
# 1. Baseline (prompt_v1)
llmopsforge eval --config configs/default.yaml --name baseline
#   -> prints baseline run id, e.g. 11aa...

# 2. Candidate (prompt_v2: grounded, always-cite, strict JSON)
llmopsforge eval --config configs/prompt_v2_eval.yaml --name candidate
#   -> prints candidate run id, e.g. 22bb...

# 3. Compare
llmopsforge regression --baseline-run-id 11aa... --candidate-run-id 22bb...
```

Or via the API:

```bash
curl -X POST localhost:8000/evals/regression \
  -H 'content-type: application/json' \
  -d '{"baseline_run_id": "11aa...", "candidate_run_id": "22bb..."}'
```

Or interactively on the **Regression Report** page of the dashboard.

## What gets compared

The comparator (`app/evals/regression.py`) joins the two runs on shared task ids
and reports:

**Metric deltas** (baseline → candidate, with % change and a better/worse verdict):

- correctness change
- citation correctness change
- grounding change
- retrieval relevance change
- JSON validity change
- hallucination rate change
- latency change
- cost change

**Task-level changes**:

- **Improved tasks** — correctness increased.
- **Fixed tasks** — failed in baseline, pass in candidate.
- **Newly failed tasks** — passed in baseline, fail in candidate.
- **Critical regressions** — tasks that did **not** hallucinate in the baseline
  but **do** in the candidate. These are weighted most heavily.

## The verdict

```
blocked        if  critical regressions exist
               OR  pass rate dropped by > 10 points
               OR  hallucination rate rose by > 5 points

safe to ship   if  no newly failed tasks
               AND pass rate did not drop
               AND hallucination rate did not rise

investigate    otherwise
```

## Example output

```
# Regression Report

## Baseline vs Candidate
| Metric                  | Baseline | Candidate |   Δ    | % change | Verdict   |
| ----------------------- | -------- | --------- | ------ | -------- | --------- |
| pass_rate               | 0.8846   | 1.0000    | +0.115 | +13.0%   | ✅ better |
| grounding_avg           | 0.9100   | 0.9700    | +0.060 | +6.6%    | ✅ better |
| json_validity_rate      | 0.0000   | 1.0000    | +1.000 | —        | ✅ better |
| total_estimated_cost... | 0.0000   | 0.0000    | 0.000  | —        | —         |

## Fixed Tasks (3)
- `json_refund_001`
- `json_pricing_001`
- `json_security_001`

## Newly Failed Tasks (0)
_None._

## Critical Hallucination Regressions (0)
_None._

## Final Recommendation
**✅ SAFE TO SHIP**
```

In this example, `prompt_v2` fixes the JSON-mode tasks (v1 returns prose, which
fails schema validation) and improves grounding — with no new failures or
hallucination regressions — so the candidate is cleared to ship.

## Using it in CI

Because every metric is deterministic and the mock provider is free, you can run
the baseline-vs-candidate comparison on every PR and **fail the build** when the
recommendation is `blocked`:

```bash
llmopsforge eval --config configs/default.yaml --name baseline
llmopsforge eval --config configs/prompt_v2_eval.yaml --name candidate
# parse the regression recommendation and exit non-zero on "blocked"
```
