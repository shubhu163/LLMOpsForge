"""Repository helpers: thin, explicit data-access functions over the ORM.

Keeping persistence logic here (rather than scattered through the pipeline and
API layers) gives a single place to evolve queries and keeps call sites concise.
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import (
    ModelConfigSpec,
    PromptTemplateConfig,
    load_all_model_configs,
    load_all_prompt_templates,
)
from app.storage import models

# --------------------------------------------------------------------------- #
# Seeding prompt templates & model configs from YAML into the DB
# --------------------------------------------------------------------------- #


def seed_prompt_templates(session: Session, templates: list[PromptTemplateConfig]) -> None:
    for t in templates:
        existing = session.get(models.PromptTemplate, t.id)
        if existing is None:
            session.add(
                models.PromptTemplate(
                    id=t.id,
                    name=t.name,
                    version=t.version,
                    system=t.system,
                    instructions=t.instructions,
                    behavior=t.behavior.model_dump(),
                )
            )


def seed_model_configs(session: Session, specs: list[ModelConfigSpec]) -> None:
    for s in specs:
        existing = session.get(models.ModelConfig, s.id)
        if existing is None:
            session.add(models.ModelConfig(**s.model_dump()))


def seed_configs(session: Session) -> None:
    """Load all prompt templates and model configs from disk into the DB."""
    seed_prompt_templates(session, load_all_prompt_templates())
    seed_model_configs(session, load_all_model_configs())
    session.flush()


# --------------------------------------------------------------------------- #
# Documents & chunks
# --------------------------------------------------------------------------- #


def upsert_document(session: Session, name: str, path: str, content_hash: str) -> models.Document:
    """Insert or replace a document by name (re-ingestion replaces chunks)."""
    existing = session.scalar(select(models.Document).where(models.Document.name == name))
    if existing is not None:
        session.delete(existing)
        session.flush()
    doc = models.Document(name=name, path=path, content_hash=content_hash)
    session.add(doc)
    session.flush()
    return doc


def add_chunks(session: Session, chunks: list[models.DocumentChunk]) -> None:
    session.add_all(chunks)
    session.flush()


def all_chunks(session: Session) -> list[models.DocumentChunk]:
    return list(session.scalars(select(models.DocumentChunk)).all())


def count_documents(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(models.Document)) or 0


def count_chunks(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(models.DocumentChunk)) or 0


def list_documents(session: Session) -> list[models.Document]:
    return list(session.scalars(select(models.Document)).all())


# --------------------------------------------------------------------------- #
# Eval runs & results
# --------------------------------------------------------------------------- #


def get_eval_run(session: Session, run_id: str) -> models.EvalRun | None:
    return session.get(models.EvalRun, run_id)


def list_eval_runs(session: Session, limit: int = 50) -> list[models.EvalRun]:
    stmt = select(models.EvalRun).order_by(models.EvalRun.created_at.desc()).limit(limit)
    return list(session.scalars(stmt).all())


def get_results_for_run(session: Session, run_id: str) -> list[models.EvalExampleResult]:
    stmt = select(models.EvalExampleResult).where(models.EvalExampleResult.eval_run_id == run_id)
    return list(session.scalars(stmt).all())


def get_prompt_template(session: Session, template_id: str) -> models.PromptTemplate | None:
    return session.get(models.PromptTemplate, template_id)


def get_model_config(session: Session, model_id: str) -> models.ModelConfig | None:
    return session.get(models.ModelConfig, model_id)
