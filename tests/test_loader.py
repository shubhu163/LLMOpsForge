"""Tests for the document loader."""

from __future__ import annotations

import pytest

from app.rag.loader import load_documents


def test_loads_sample_corpus(docs_dir):
    docs = load_documents(docs_dir)
    names = {d.name for d in docs}
    assert "refund_policy.md" in names
    assert len(docs) >= 5
    assert all(d.text for d in docs)
    assert all(d.content_hash for d in docs)


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError):
        load_documents("/nonexistent/path/xyz")


def test_skips_unsupported_and_errors_when_empty(tmp_path):
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    with pytest.raises(ValueError):
        load_documents(str(tmp_path))


def test_single_file(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\nSome content here.", encoding="utf-8")
    docs = load_documents(str(f))
    assert len(docs) == 1
    assert docs[0].name == "note.md"
