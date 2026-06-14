"""Schemas for the concepts layer of module-knowledge."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

CONCEPTS_SCHEMA_VERSION: Literal["concepts.v1"] = "concepts.v1"

ConceptKind = Literal["technique", "entity"]


class RecordSourceRef(BaseModel):
    """Source reference in records-and-concepts mode."""

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(min_length=1)
    kind: str


class LightSourceRef(BaseModel):
    """Source reference in concepts-only mode."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    id: str = Field(min_length=1)
    title: str
    url: str | None = None


class ConceptLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    techniques: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)


class ConceptPage(BaseModel):
    """A parsed concept page: YAML frontmatter fields plus the markdown body."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["concepts.v1"] = CONCEPTS_SCHEMA_VERSION
    kind: ConceptKind
    slug: str = Field(min_length=1)
    title: str = Field(min_length=1)
    as_of_run: str = Field(min_length=1)
    subject_system: str | None = None
    subject_systems_seen: list[str] = Field(default_factory=list)
    objectives_seen: list[str] = Field(default_factory=list)
    sources: list[RecordSourceRef | LightSourceRef] = Field(default_factory=list)
    links: ConceptLinks = Field(default_factory=ConceptLinks)
    content_hash: str | None = None
    body: str = ""


class InlineSource(BaseModel):
    """Evidence carried inline in concepts-only mode."""

    model_config = ConfigDict(extra="forbid")

    kind: str
    id: str = Field(min_length=1)
    title: str
    body: str
    url: str | None = None
    module_qn: str | None = None


class RecordsBackedIngestRequest(BaseModel):
    """Ingest input in records-and-concepts mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["records-and-concepts"] = "records-and-concepts"
    record_ids: list[str]
    subject_system: str
    objective: str | None = None
    run_id: str


class InlineIngestRequest(BaseModel):
    """Ingest input in concepts-only mode."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["concepts-only"] = "concepts-only"
    sources: list[InlineSource]
    subject_system: str
    objective: str | None = None
    run_id: str


IngestRequest = Annotated[
    RecordsBackedIngestRequest | InlineIngestRequest,
    Field(discriminator="mode"),
]


class IngestReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages_updated: list[str] = Field(default_factory=list)
    pages_created: list[str] = Field(default_factory=list)
    llm_calls: int = 0
    cost_usd: float = 0.0
    duration_s: float = 0.0


class QueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: ConceptPage
    score: float = Field(ge=0.0)
    matched_terms: list[str] = Field(default_factory=list)


class ConceptVerifyIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_path: str
    severity: Literal["error", "warning"] = "error"
    message: str


class ConceptVerifyReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    checked_pages: int
    issues: list[ConceptVerifyIssue] = Field(default_factory=list)


class LintReport(BaseModel):
    """Stub for the deferred lint pass (see design plan §4.6)."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    checked_pages: int
    issues: list[str] = Field(default_factory=list)


__all__ = [
    "CONCEPTS_SCHEMA_VERSION",
    "ConceptKind",
    "ConceptLinks",
    "ConceptPage",
    "ConceptVerifyIssue",
    "ConceptVerifyReport",
    "IngestReport",
    "IngestRequest",
    "InlineIngestRequest",
    "InlineSource",
    "LightSourceRef",
    "LintReport",
    "QueryResult",
    "RecordSourceRef",
    "RecordsBackedIngestRequest",
]
