"""Lexical retrieval over concept pages.

Reuses the same term-frequency weighting as records.retrieve — no new
ranker, just adapted field weights for ConceptPage inputs.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from spotlights_engine.module_knowledge.concepts.schemas import ConceptPage, QueryResult

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*(?:[./][A-Za-z0-9_+-]+)*")


def query_concepts(
    pages: Iterable[ConceptPage],
    q: str,
    *,
    top_k: int = 5,
    kinds: list[str] | None = None,
) -> list[QueryResult]:
    """Return ranked concept pages for a lexical query."""
    query_terms = Counter(_tokens(q))
    required_kinds = set(kinds) if kinds else None
    hits: list[QueryResult] = []
    for page in pages:
        if required_kinds and page.kind not in required_kinds:
            continue
        score, matched = _score_page(page, query_terms)
        if score <= 0:
            continue
        hits.append(
            QueryResult(
                page=page,
                score=_normalize_score(score, query_terms),
                matched_terms=matched,
            )
        )
    return sorted(hits, key=lambda r: (-r.score, r.page.kind, r.page.slug))[:top_k]


def _score_page(page: ConceptPage, query_terms: Counter[str]) -> tuple[float, list[str]]:
    tag_text = " ".join(
        page.links.techniques
        + page.links.entities
        + ([page.subject_system] if page.subject_system else [])
        + page.subject_systems_seen
    )
    weighted_parts = [
        (4.0, page.title),
        (3.0, page.slug.replace("-", " ").replace("/", " ")),
        (2.0, page.kind),
        (2.0, tag_text),
        (1.0, page.body),
    ]
    score = 0.0
    matched: set[str] = set()
    for weight, text in weighted_parts:
        counts = Counter(_tokens(text))
        for term, query_count in query_terms.items():
            if term in counts:
                score += weight * min(counts[term], 3) * query_count
                matched.add(term)
    return score, sorted(matched)


def _normalize_score(score: float, query_terms: Counter[str]) -> float:
    denom = max(1.0, len(query_terms) * 8.0)
    return round(min(1.0, math.log1p(score) / math.log1p(denom)), 4)


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


__all__ = ["query_concepts"]
