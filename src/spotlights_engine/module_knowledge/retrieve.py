"""Local lexical Retrieve API for module knowledge records."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from spotlights_engine.module_knowledge.schemas import (
    KnowledgeRecord,
    RetrievedItem,
    RetrieveRequest,
)

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]*(?:[./][A-Za-z0-9_+-]+)*")


def retrieve_records(
    records: Iterable[KnowledgeRecord],
    request: RetrieveRequest | str,
    *,
    top_k: int | None = None,
) -> list[RetrievedItem]:
    """Return ranked local matches for a query.

    The implementation is deliberately transparent and dependency-free for v1:
    weighted lexical overlap over title, tags, source metadata, and body text.
    Embeddings can be added later behind this function without changing callers.
    """
    active_request = RetrieveRequest(query=request) if isinstance(request, str) else request
    if top_k is not None:
        active_request = RetrieveRequest.model_validate(
            active_request.model_copy(update={"top_k": top_k}).model_dump()
        )
    required_source_types = set(active_request.source_types)
    required_tags = set(active_request.tags)
    query_terms = Counter(_tokens(active_request.query))
    hits: list[RetrievedItem] = []

    for record in records:
        if required_source_types and record.source_type not in required_source_types:
            continue
        if required_tags and not required_tags.issubset(set(record.tags)):
            continue
        score, matched = _score_record(record, query_terms)
        if score <= 0:
            continue
        hits.append(
            RetrievedItem(
                record=record,
                score=_normalize_score(score, query_terms),
                matched_terms=matched,
                why_returned="Matched query terms in the local module-knowledge archive.",
            )
        )

    return sorted(
        hits,
        key=lambda item: (-item.score, item.record.source_type, item.record.record_id),
    )[: active_request.top_k]


def _score_record(record: KnowledgeRecord, query_terms: Counter[str]) -> tuple[float, list[str]]:
    weighted_parts = [
        (4.0, record.title),
        (3.0, " ".join(record.tags)),
        (2.0, record.source_type),
        (2.0, record.source.title),
        (1.5, record.provenance.locator),
        (1.0, record.text),
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
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text or "")]


__all__ = ["retrieve_records"]
