"""Identifier normalization for expanded module deep research."""

from __future__ import annotations

import re

_MODERN_ARXIV_RE = re.compile(r"(?:arxiv[:/\s]|abs/|pdf/)?(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)


def normalize_arxiv_id(value: str | None) -> str | None:
    """Return a versionless modern arXiv ID, or None for malformed IDs."""
    if not value:
        return None
    match = _MODERN_ARXIV_RE.search(value.strip())
    if not match:
        return None
    candidate = match.group(1)
    return candidate if _has_valid_arxiv_month(candidate) else None


def extract_arxiv_id(text: str | None) -> str | None:
    """Extract a valid versionless arXiv identifier from arbitrary text or URL."""
    return normalize_arxiv_id(text)


def extract_doi(text: str | None) -> str | None:
    """Extract a lowercase DOI from arbitrary text or URL."""
    if not text:
        return None
    match = _DOI_RE.search(text)
    return match.group(0).lower().rstrip(".") if match else None


def _has_valid_arxiv_month(arxiv_id: str) -> bool:
    """Modern arXiv IDs start with YYMM; MM must be 01 through 12."""
    month = int(arxiv_id[2:4])
    return 1 <= month <= 12
