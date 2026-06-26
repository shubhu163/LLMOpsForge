"""Pytest fixtures: isolated per-test database, seeded configs, ingested corpus.

Every test runs against a fresh temporary SQLite database and uses the
deterministic mock provider with the offline hashing embedder, so the suite is
fast, reproducible, and requires no network or paid API calls.
"""

from __future__ import annotations

import pytest

from app.config import DOCUMENTS_DIR, get_settings


@pytest.fixture
def temp_env(tmp_path, monkeypatch):
    """Point the app at a throwaway database and data dir for this test."""
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("LLMOPS_DATABASE_URL", f"sqlite:///{db_file}")
    monkeypatch.setenv("LLMOPS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLMOPS_DEFAULT_PROVIDER", "mock")

    from app.storage import database

    get_settings.cache_clear()
    database.reset_engine()
    database.init_db()

    with database.session_scope() as session:
        from app.storage.repository import seed_configs

        seed_configs(session)

    yield tmp_path

    database.reset_engine()
    get_settings.cache_clear()


@pytest.fixture
def session(temp_env):
    """A SQLAlchemy session bound to the per-test database."""
    from app.storage.database import get_session_factory

    factory = get_session_factory()
    sess = factory()
    try:
        yield sess
    finally:
        sess.close()


@pytest.fixture
def pipeline(session):
    """A RagPipeline using the offline hashing embedder (no downloads)."""
    from app.rag.pipeline import RagPipeline

    return RagPipeline(session, prefer_real_embedder=False)


@pytest.fixture
def ingested_pipeline(pipeline):
    """A pipeline with the sample document corpus already ingested."""
    pipeline.ingest(str(DOCUMENTS_DIR), chunk_size=600, chunk_overlap=100)
    return pipeline


@pytest.fixture
def docs_dir() -> str:
    return str(DOCUMENTS_DIR)
