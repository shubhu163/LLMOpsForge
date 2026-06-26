"""Character-based chunking with configurable size and overlap.

Chunking is paragraph-aware: it accumulates whole paragraphs up to ``chunk_size``
before splitting, which keeps sentences and policy clauses intact (important for
grounded retrieval). Oversized paragraphs are hard-split with overlap.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class Chunk:
    chunk_id: str
    document_name: str
    chunk_index: int
    text: str
    char_start: int
    char_end: int


def _hard_split(text: str, chunk_size: int, overlap: int) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        spans.append((text[start:end], start, end))
        if end >= n:
            break
        start += step
    return spans


def chunk_document(
    document_name: str,
    text: str,
    *,
    chunk_size: int = 600,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Split a document into overlapping, paragraph-aware chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    paragraphs = [p for p in text.split("\n\n")]
    chunks: list[Chunk] = []
    index = 0
    cursor = 0  # character offset into the original text

    buffer = ""
    buffer_start = 0

    def flush(buf: str, start: int) -> None:
        nonlocal index
        buf = buf.strip()
        if not buf:
            return
        chunks.append(
            Chunk(
                chunk_id=f"{document_name}::chunk_{index}",
                document_name=document_name,
                chunk_index=index,
                text=buf,
                char_start=start,
                char_end=start + len(buf),
            )
        )
        index += 1

    for para in paragraphs:
        para_with_sep = para + "\n\n"
        if len(para) > chunk_size:
            # Flush whatever is buffered, then hard-split the large paragraph.
            flush(buffer, buffer_start)
            buffer, buffer_start = "", cursor + len(para_with_sep)
            for piece, s, _e in _hard_split(para, chunk_size, chunk_overlap):
                flush(piece, cursor + s)
            cursor += len(para_with_sep)
            continue

        if len(buffer) + len(para) + 2 > chunk_size and buffer:
            flush(buffer, buffer_start)
            buffer, buffer_start = "", cursor

        if not buffer:
            buffer_start = cursor
        buffer += para_with_sep
        cursor += len(para_with_sep)

    flush(buffer, buffer_start)

    if not chunks:
        # Degenerate input (e.g. whitespace) — emit a single empty-safe chunk.
        flush(text, 0)

    logger.debug("Chunked %s into %d chunk(s)", document_name, len(chunks))
    return chunks
