"""Typer command-line interface for LLMOpsForge.

Commands:
    llmopsforge ingest --docs-path documents/
    llmopsforge query "What is the refund policy?"
    llmopsforge eval --dataset datasets/qa_eval.jsonl --config configs/default.yaml
    llmopsforge regression --baseline-run-id <id> --candidate-run-id <id>
    llmopsforge report --eval-run-id <id> --format markdown
    llmopsforge dashboard
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from app.config import get_settings, load_eval_config
from app.evals.regression import compare_runs
from app.evals.reports import (
    eval_report_json,
    eval_report_markdown,
    regression_report_markdown,
)
from app.evals.runner import EvalRunner
from app.logging_config import configure_logging
from app.rag.pipeline import RagPipeline
from app.storage.database import init_db, session_scope
from app.storage.repository import seed_configs

app = typer.Typer(
    add_completion=False,
    help="LLMOpsForge — LLM evaluation & monitoring for RAG applications.",
)
console = Console()


def _bootstrap() -> None:
    """Ensure logging, schema, and seeded configs are ready."""
    configure_logging(get_settings().log_level)
    init_db()
    with session_scope() as session:
        seed_configs(session)


@app.command()
def ingest(
    docs_path: str = typer.Option("documents/", "--docs-path", help="File or directory to ingest."),
    chunk_size: int = typer.Option(600, help="Chunk size in characters."),
    chunk_overlap: int = typer.Option(100, help="Chunk overlap in characters."),
) -> None:
    """Ingest local documents into the vector index."""
    _bootstrap()
    with session_scope() as session:
        pipeline = RagPipeline(session)
        result = pipeline.ingest(docs_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    console.print(
        f"[green]Ingested[/green] {result['documents']} document(s), {result['chunks']} chunk(s)."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask."),
    top_k: int = typer.Option(4, help="Number of chunks to retrieve."),
    prompt: str = typer.Option("prompt_v1", "--prompt", help="Prompt template id."),
    model: str = typer.Option("mock-small", "--model", help="Model config id."),
    require_citations: bool = typer.Option(True, help="Require citations in the answer."),
) -> None:
    """Run a single RAG query and print the grounded answer."""
    _bootstrap()
    with session_scope() as session:
        pipeline = RagPipeline(session)
        if pipeline.retriever.size == 0:
            console.print("[red]No documents ingested.[/red] Run `llmopsforge ingest` first.")
            raise typer.Exit(code=1)
        result = pipeline.query(
            question,
            top_k=top_k,
            prompt_template_id=prompt,
            model_config_id=model,
            require_citations=require_citations,
        )

    console.print(f"\n[bold cyan]Answer:[/bold cyan] {result.answer}\n")
    if result.citations:
        console.print("[bold]Citations:[/bold]")
        for c in result.citations:
            console.print(f"  • {c.document_name} ({c.chunk_id})")
    console.print(
        f"\n[dim]model={result.model_name} latency={result.latency_ms:.1f}ms "
        f"tokens={result.estimated_tokens} cost=${result.estimated_cost:.6f} "
        f"backend={pipeline.retriever.backend}[/dim]"
    )


@app.command()
def eval(
    dataset: str = typer.Option("datasets/qa_eval.jsonl", "--dataset", help="JSONL dataset path."),
    config: str = typer.Option("configs/default.yaml", "--config", help="Eval config YAML."),
    name: str | None = typer.Option(None, "--name", help="Optional run name."),
) -> None:
    """Run an evaluation over a dataset and print the summary."""
    _bootstrap()
    cfg = load_eval_config(config)
    with session_scope() as session:
        run = EvalRunner(session).run(dataset_path=dataset, config=cfg, name=name)
        summary = run.summary
        run_id = run.id

    console.print(f"\n[bold green]Eval run complete:[/bold green] [bold]{run_id}[/bold]")
    table = Table(title="Summary", show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    for key in (
        "total_examples",
        "passed",
        "failed",
        "pass_rate",
        "answer_correctness_avg",
        "citation_correctness_avg",
        "grounding_avg",
        "retrieval_relevance_avg",
        "hallucination_count",
        "json_validity_rate",
        "avg_latency_ms",
        "total_estimated_cost_usd",
    ):
        if key in summary:
            table.add_row(key, str(summary[key]))
    console.print(table)
    console.print(f"\n[dim]Generate a report:[/dim] llmopsforge report --eval-run-id {run_id}")


@app.command()
def regression(
    baseline_run_id: str = typer.Option(..., "--baseline-run-id", help="Baseline eval run id."),
    candidate_run_id: str = typer.Option(..., "--candidate-run-id", help="Candidate eval run id."),
) -> None:
    """Compare two eval runs and print a regression verdict."""
    _bootstrap()
    with session_scope() as session:
        try:
            report = compare_runs(
                session, baseline_run_id=baseline_run_id, candidate_run_id=candidate_run_id
            )
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        markdown = regression_report_markdown(report)
    console.print(markdown)


@app.command()
def report(
    eval_run_id: str = typer.Option(..., "--eval-run-id", help="Eval run id to report on."),
    fmt: str = typer.Option("markdown", "--format", help="markdown or json."),
    output: str | None = typer.Option(None, "--output", help="Write to a file instead of stdout."),
) -> None:
    """Generate a Markdown or JSON evaluation report."""
    _bootstrap()
    with session_scope() as session:
        try:
            content = (
                eval_report_json(session, eval_run_id)
                if fmt == "json"
                else eval_report_markdown(session, eval_run_id)
            )
        except KeyError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    if output:
        Path(output).write_text(content, encoding="utf-8")
        console.print(f"[green]Report written to[/green] {output}")
    else:
        console.print(content)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    script = Path(__file__).resolve().parent.parent / "dashboard" / "streamlit_app.py"
    console.print(f"[cyan]Launching Streamlit dashboard:[/cyan] {script}")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(script)], check=True)
    except FileNotFoundError:
        console.print(
            "[red]Streamlit is not installed.[/red] Install with: pip install -e '.[dashboard]'"
        )
        raise typer.Exit(code=1) from None


if __name__ == "__main__":
    app()
