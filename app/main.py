"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.config import get_settings
from app.logging_config import configure_logging, get_logger
from app.storage.database import init_db, session_scope
from app.storage.repository import seed_configs

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the database and seed prompt/model configs on startup."""
    configure_logging(get_settings().log_level)
    init_db()
    with session_scope() as session:
        seed_configs(session)
    logger.info("LLMOpsForge API started (provider=%s)", get_settings().default_provider)
    yield


app = FastAPI(
    title="LLMOpsForge",
    description="LLM evaluation and monitoring platform for RAG applications.",
    version=__version__,
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {
        "name": "LLMOpsForge",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
    }
