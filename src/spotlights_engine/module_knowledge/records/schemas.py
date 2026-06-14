"""Schemas for the Spotlights module-knowledge memory layer.

The archive is intentionally small and stable: callers store evidence-bearing
records, retrieve ranked records by query, and render a generated Markdown wiki.
JSONL is the source of truth; the wiki is a derived view.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION: Literal["module_knowledge.v1"] = "module_knowledge.v1"

KnowledgeSourceType = Literal[
    "paper",
    "blog",
    "docs",
    "issue",
    "pr",
    "talk",
    "codebase",
    "module_map",
    "experiment",
    "candidate",
    "finding",
    "proposal",
    "note",
    "other",
]

TrustTier = Literal["primary", "credible", "informal", "internal", "unknown"]


class SourceRef(BaseModel):
    """Where a knowledge record came from."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    published_at: str | None = None
    trust_tier: TrustTier = "unknown"


class Provenance(BaseModel):
    """Replay/debug information for a knowledge record."""

    model_config = ConfigDict(extra="forbid")

    locator: str = Field(min_length=1)
    extractor: str | None = None
    artifact_path: str | None = None
    content_hash: str | None = None
    created_at: str | None = None


class KnowledgeRecord(BaseModel):
    """One durable memory item in the local knowledge archive."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["module_knowledge.v1"] = SCHEMA_VERSION
    record_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.:-]+$")
    source_type: KnowledgeSourceType
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source: SourceRef
    provenance: Provenance
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, values: list[str]) -> list[str]:
        return sorted({tag.strip().lower() for tag in values if tag.strip()})


class RetrieveRequest(BaseModel):
    """Local retrieval request over archived knowledge records."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1)
    source_types: list[KnowledgeSourceType] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags")
    @classmethod
    def _normalize_tags(cls, values: list[str]) -> list[str]:
        return sorted({tag.strip().lower() for tag in values if tag.strip()})


class RetrievedItem(BaseModel):
    """Ranked retrieval hit returned by the Retrieve API."""

    model_config = ConfigDict(extra="forbid")

    record: KnowledgeRecord
    score: float = Field(ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)
    why_returned: str = ""


class ArchiveResult(BaseModel):
    """Archive write summary."""

    model_config = ConfigDict(extra="forbid")

    path: str
    records_written: int
    inserted: int
    updated: int
    record_ids: list[str] = Field(default_factory=list)


class WikiRenderResult(BaseModel):
    """Generated wiki output summary."""

    model_config = ConfigDict(extra="forbid")

    wiki_dir: str
    pages_written: int
    source_hash: str


class WikiVerificationIssue(BaseModel):
    """One generated-wiki verification problem."""

    model_config = ConfigDict(extra="forbid")

    page_path: str
    severity: Literal["error", "warning"] = "error"
    message: str


class WikiVerificationReport(BaseModel):
    """Verification result for generated wiki pages."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    checked_pages: int
    issues: list[WikiVerificationIssue] = Field(default_factory=list)


__all__ = [
    "SCHEMA_VERSION",
    "ArchiveResult",
    "KnowledgeRecord",
    "KnowledgeSourceType",
    "Provenance",
    "RetrieveRequest",
    "RetrievedItem",
    "SourceRef",
    "TrustTier",
    "WikiRenderResult",
    "WikiVerificationIssue",
    "WikiVerificationReport",
]
