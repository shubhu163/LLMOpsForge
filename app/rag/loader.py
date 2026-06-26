"""Document loader for plain-text and Markdown corpora."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from app.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class LoadedDocument:
    name: str
    path: str
    text: str
    content_hash: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def load_documents(docs_path: str | Path) -> list[LoadedDocument]:
    """Load all supported documents from a file or directory.

    Raises ``FileNotFoundError`` if the path does not exist and ``ValueError`` if
    a directory contains no supported documents.
    """
    path = Path(docs_path)
    if not path.exists():
        raise FileNotFoundError(f"Documents path does not exist: {path}")

    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES
        )

    docs: list[LoadedDocument] = []
    for f in files:
        if f.suffix.lower() not in SUPPORTED_SUFFIXES:
            logger.warning("Skipping unsupported file: %s", f.name)
            continue
        text = f.read_text(encoding="utf-8")
        docs.append(LoadedDocument(name=f.name, path=str(f), text=text, content_hash=_hash(text)))

    if not docs:
        raise ValueError(f"No supported documents (.txt/.md) found at: {path}")

    logger.info("Loaded %d document(s) from %s", len(docs), path)
    return docs
