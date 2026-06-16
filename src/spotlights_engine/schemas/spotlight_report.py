"""Unified cross-pipeline output schema (`SpotlightReport`).

Both pipelines (deep-research and signal) emit one of these alongside their
existing pipeline-internal results. The shape is flat — top-level
`candidates`, `findings`, and `anomalies` lists; consumers that want a
per-module view group client-side using `module_qualified_name`. Per-pipeline
adapters fold native artifacts into this shape; nothing existing is replaced.

See `docs/_review-notes/2026-06-15_spotlight_report_schema.md` for the full
spec, naming map, and worked example.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import (
    CandidateKind,
    CandidateState,
    EstimatedImpact,
)
from spotlights_engine.schemas.common import SpotlightContext, StepIssue
from spotlights_engine.schemas.finding import FindingSourceType
from spotlights_engine.schemas.project import ProjectTree


# ── Location / CodeSpan ─────────────────────────────────────────────────────


class CodeSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)


class Location(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    spans: list[CodeSpan] = Field(min_length=1)


# ── SpotlightProposal ───────────────────────────────────────────────────────

ProposalType = Literal[
    "prefetch", "reorder", "replace", "tune", "add_cache", "fuse", "other"
]
ProposalSource = Literal[
    "research_finding", "agent_knowledge", "telemetry_anomaly"
]


class SpotlightProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^prop-\d{4}$")

    source: ProposalSource
    source_refs: list[str] = Field(default_factory=list)
    author: str | None = None

    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    proposal_type: ProposalType | None = None
    mechanism: str | None = None
    required_changes: str | None = None
    expected_effect: str | None = None
    evaluation_metric: str | None = None


# ── SpotlightCandidate ──────────────────────────────────────────────────────

CandidateOrigin = Literal["telemetry_anomaly", "code_agent"]


class SpotlightCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")

    module_qualified_name: str | None = None
    origin: CandidateOrigin

    locations: list[Location] = Field(min_length=1)

    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)
    state: CandidateState = "DISCOVERED"

    proposals: list[SpotlightProposal] = Field(default_factory=list)


# ── SpotlightFinding ────────────────────────────────────────────────────────


class SpotlightFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^find-\d{4}$")
    module_qualified_name: str = Field(min_length=1)

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""


# ── SpotlightAnomaly ────────────────────────────────────────────────────────


class SpotlightAnomaly(BaseModel):
    model_config = ConfigDict(extra="allow")

    anomaly_id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    description: str = ""


# ── RunInfo ─────────────────────────────────────────────────────────────────


class RunInfo(BaseModel):
    """Info about the run that produced this report."""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["deep_research", "signal"]
    run_id: str = Field(min_length=1)
    started_at: str
    finished_at: str | None = None
    model: str | None = None
    cost_usd: float | None = None
    duration_s: float | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


# ── SpotlightReport (top-level) ─────────────────────────────────────────────


class SpotlightReport(BaseModel):
    """Cross-pipeline output. Flat lists; group by `module_qualified_name`
    client-side."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"

    project_tree: ProjectTree
    context: SpotlightContext

    candidates: list[SpotlightCandidate] = Field(default_factory=list)
    findings: list[SpotlightFinding] = Field(default_factory=list)
    anomalies: list[SpotlightAnomaly] = Field(default_factory=list)

    run: RunInfo
    issues: list[StepIssue] = Field(default_factory=list)


__all__ = [
    "CandidateOrigin",
    "CodeSpan",
    "Location",
    "ProposalSource",
    "ProposalType",
    "RunInfo",
    "SpotlightAnomaly",
    "SpotlightCandidate",
    "SpotlightFinding",
    "SpotlightProposal",
    "SpotlightReport",
]
