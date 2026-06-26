"""Streamlit dashboard for LLMOpsForge.

Pages:
    1. Evaluation Summary
    2. Failed Examples
    3. Prompt/Model Comparison
    4. Regression Report
    5. Retrieval Inspection

Run with: ``streamlit run dashboard/streamlit_app.py`` (or ``llmopsforge dashboard``).
Reads directly from the SQLite database populated by eval runs.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.evals.regression import compare_runs
from app.evals.reports import regression_report_markdown
from app.storage.database import init_db, session_scope
from app.storage.repository import (
    get_results_for_run,
    list_documents,
    list_eval_runs,
)

st.set_page_config(page_title="LLMOpsForge", page_icon="🔥", layout="wide")

init_db()


def _run_options() -> dict[str, str]:
    with session_scope() as session:
        runs = list_eval_runs(session, limit=200)
        return {
            f"{r.name or 'run'} · {r.id[:8]} · {r.created_at:%Y-%m-%d %H:%M}": r.id for r in runs
        }


def _load_run(run_id: str):
    with session_scope() as session:
        runs = {r.id: r for r in list_eval_runs(session, limit=500)}
        run = runs.get(run_id)
        results = get_results_for_run(session, run_id)
        # Detach into plain dicts so the session can close.
        run_dict = (
            None
            if run is None
            else {
                "id": run.id,
                "name": run.name,
                "summary": run.summary,
                "prompt_template_id": run.prompt_template_id,
                "model_config_id": run.model_config_id,
            }
        )
        results_data = [
            {
                "task_id": r.task_id,
                "question": r.question,
                "answer": r.answer,
                "difficulty": r.difficulty,
                "passed": r.passed,
                "hallucination_flag": r.hallucination_flag,
                "json_validity": r.json_validity,
                "citations": r.citations,
                "retrieved_contexts": r.retrieved_contexts,
                "metrics": r.metrics,
                "latency_ms": r.latency_ms,
                "estimated_cost": r.estimated_cost,
            }
            for r in results
        ]
    return run_dict, results_data


st.sidebar.title("🔥 LLMOpsForge")
page = st.sidebar.radio(
    "Page",
    [
        "Evaluation Summary",
        "Failed Examples",
        "Prompt/Model Comparison",
        "Regression Report",
        "Retrieval Inspection",
    ],
)

options = _run_options()
if not options:
    st.warning("No eval runs found. Run `llmopsforge eval` first to populate the dashboard.")
    st.stop()


# --------------------------------------------------------------------------- #
# Page 1: Evaluation Summary
# --------------------------------------------------------------------------- #
if page == "Evaluation Summary":
    st.title("Evaluation Summary")
    label = st.selectbox("Eval run", list(options.keys()))
    run, results = _load_run(options[label])
    s = run["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total examples", s.get("total_examples", 0))
    c2.metric("Pass rate", f"{(s.get('pass_rate') or 0) * 100:.1f}%")
    c3.metric("Hallucinations", s.get("hallucination_count", 0))
    c4.metric("Total cost", f"${s.get('total_estimated_cost_usd', 0):.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Correctness", f"{s.get('answer_correctness_avg', 0):.3f}")
    c6.metric("Citation", f"{s.get('citation_correctness_avg', 0):.3f}")
    c7.metric("Grounding", f"{s.get('grounding_avg', 0):.3f}")
    jvr = s.get("json_validity_rate")
    c8.metric("JSON validity", "n/a" if jvr is None else f"{jvr * 100:.0f}%")

    c9, c10 = st.columns(2)
    c9.metric("Avg latency", f"{s.get('avg_latency_ms', 0):.1f} ms")
    c10.metric("Retrieval relevance", f"{s.get('retrieval_relevance_avg', 0):.3f}")

    st.subheader("Per-task results")
    df = pd.DataFrame(
        [
            {
                "task_id": r["task_id"],
                "difficulty": r["difficulty"],
                "passed": r["passed"],
                "correctness": r["metrics"].get("answer_correctness_score"),
                "grounding": r["metrics"].get("grounding_score"),
                "citation": r["metrics"].get("citation_correctness_score"),
                "retrieval": r["metrics"].get("retrieval_relevance_score"),
                "json": r["json_validity"],
                "hallucination": r["hallucination_flag"],
                "latency_ms": r["latency_ms"],
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page 2: Failed Examples
# --------------------------------------------------------------------------- #
elif page == "Failed Examples":
    st.title("Failed Examples")
    label = st.selectbox("Eval run", list(options.keys()))
    _, results = _load_run(options[label])
    failed = [r for r in results if not r["passed"]]
    st.write(f"**{len(failed)}** failed example(s).")
    for r in failed:
        with st.expander(f"❌ {r['task_id']} — {r['question']}"):
            st.markdown(f"**Answer:** {r['answer']}")
            st.markdown(f"**JSON validity:** {r['json_validity']}")
            st.markdown(f"**Hallucination:** {r['hallucination_flag']}")
            st.json(r["metrics"])


# --------------------------------------------------------------------------- #
# Page 3: Prompt/Model Comparison
# --------------------------------------------------------------------------- #
elif page == "Prompt/Model Comparison":
    st.title("Prompt / Model Comparison")
    col_a, col_b = st.columns(2)
    label_a = col_a.selectbox("Run A", list(options.keys()), key="a")
    label_b = col_b.selectbox("Run B", list(options.keys()), key="b", index=min(1, len(options) - 1))
    run_a, _ = _load_run(options[label_a])
    run_b, _ = _load_run(options[label_b])

    keys = [
        "pass_rate", "answer_correctness_avg", "citation_correctness_avg",
        "grounding_avg", "retrieval_relevance_avg", "hallucination_rate",
        "json_validity_rate", "avg_latency_ms", "total_estimated_cost_usd",
    ]
    rows = []
    for k in keys:
        a = run_a["summary"].get(k)
        b = run_b["summary"].get(k)
        delta = None if (a is None or b is None) else round(b - a, 4)
        rows.append({"metric": k, "Run A": a, "Run B": b, "Δ (B−A)": delta})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    st.caption(
        f"Run A = `{run_a['prompt_template_id']}` / `{run_a['model_config_id']}` · "
        f"Run B = `{run_b['prompt_template_id']}` / `{run_b['model_config_id']}`"
    )


# --------------------------------------------------------------------------- #
# Page 4: Regression Report
# --------------------------------------------------------------------------- #
elif page == "Regression Report":
    st.title("Regression Report")
    col_a, col_b = st.columns(2)
    base_label = col_a.selectbox("Baseline run", list(options.keys()), key="base")
    cand_label = col_b.selectbox(
        "Candidate run", list(options.keys()), key="cand", index=min(1, len(options) - 1)
    )
    if st.button("Run regression comparison"):
        with session_scope() as session:
            report = compare_runs(
                session,
                baseline_run_id=options[base_label],
                candidate_run_id=options[cand_label],
                persist=False,
            )
            md = regression_report_markdown(report)
        st.markdown(md)


# --------------------------------------------------------------------------- #
# Page 5: Retrieval Inspection
# --------------------------------------------------------------------------- #
elif page == "Retrieval Inspection":
    st.title("Retrieval Inspection")
    with session_scope() as session:
        docs = [d.name for d in list_documents(session)]
    st.write(f"**Ingested documents:** {', '.join(docs) if docs else '(none)'}")

    label = st.selectbox("Eval run", list(options.keys()))
    _, results = _load_run(options[label])
    task_ids = [r["task_id"] for r in results]
    chosen = st.selectbox("Task", task_ids)
    record = next((r for r in results if r["task_id"] == chosen), None)
    if record:
        st.markdown(f"**Question:** {record['question']}")
        st.markdown(f"**Answer:** {record['answer']}")
        st.markdown("**Cited sources:**")
        st.write([c.get("document_name") for c in record["citations"]] or "(none)")
        st.markdown("**Retrieved contexts:**")
        for ctx in record["retrieved_contexts"]:
            with st.expander(f"{ctx['document_name']} · {ctx['chunk_id']} · score={ctx['score']:.3f}"):
                st.write(ctx["text"])
