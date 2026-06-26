"""Citation extraction and tracking.

Citations link an answer back to the source document name and chunk id it drew
from. Two strategies are supported:

* Provider-reported (``used_chunk_ids`` in the response ``raw``) — used by the
  mock provider, which knows exactly which chunks it extracted from.
* Overlap-based fallback — for real providers, we match answer tokens against
  each retrieved chunk and cite those with meaningful overlap.

The prompt ``always_cite`` behavior flag controls breadth: when off (v1) only the
single strongest source is cited; when on (v2) every used source is cited.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from app.providers.base import RetrievedContext

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass
class Citation:
    chunk_id: str
    document_name: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def extract_citations(
    *,
    answer: str,
    contexts: list[RetrievedContext],
    used_chunk_ids: list[str] | None,
    require_citations: bool,
    always_cite: bool,
) -> list[Citation]:
    """Compute citations for an answer given retrieved context."""
    if not require_citations or not contexts:
        return []

    by_id = {c.chunk_id: c for c in contexts}
    candidates: list[Citation] = []

    if used_chunk_ids:
        for cid in used_chunk_ids:
            ctx = by_id.get(cid)
            if ctx is not None:
                candidates.append(
                    Citation(
                        chunk_id=ctx.chunk_id, document_name=ctx.document_name, score=ctx.score
                    )
                )
    else:
        # Overlap-based fallback for providers that do not report used chunks.
        ans_tokens = _tokens(answer)
        scored = []
        for ctx in contexts:
            overlap = len(ans_tokens & _tokens(ctx.text))
            if overlap > 0:
                scored.append((overlap, ctx))
        scored.sort(key=lambda x: -x[0])
        candidates = [
            Citation(chunk_id=c.chunk_id, document_name=c.document_name, score=float(o))
            for o, c in scored
        ]

    if not candidates:
        return []
    return candidates if always_cite else candidates[:1]
