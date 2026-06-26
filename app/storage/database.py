"""Database engine, session management, and schema initialisation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


_engine = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine():
    """Return a lazily-created SQLAlchemy engine bound to the configured URL."""
    global _engine
    if _engine is None:
        url = get_settings().database_url
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, future=True, connect_args=connect_args)
        logger.debug("Created engine for %s", url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    # Import models so they are registered on Base.metadata before create_all.
    from app.storage import models  # noqa: F401

    Base.metadata.create_all(bind=get_engine())
    logger.debug("Database schema initialised")


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session scope; commits on success, rolls back on error."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Drop cached engine/session factory (used by tests that swap databases)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
