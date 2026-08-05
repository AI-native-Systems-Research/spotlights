"""Per-step pipeline I/O contracts for the spotlights deep-research path.

Mirrors the input/output classes shown in
`docs/architecture/spotlights_deep_research_path_architecture.md`. The data
types they reference live in `schemas.project`, `schemas.common`,
`schemas.finding`, and `schemas.candidate`; this module is purely the
boundary contracts so the manager and per-step modules can talk to one
another with typed payloads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.anomaly import Anomaly
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import (
    ModuleRunStatus,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.schemas.search import SearchQueryLog


class ModulesExtractorInput(BaseModel):
    """Input contract for step 1 (`modules_extractor`).

    `SpotlightContext` is intentionally omitted: the structural map must not
    be biased by objective so its output is reusable across runs.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path


class CandidateDiscoveryInput(BaseModel):
    """Input contract for step 2 (`candidate_discovery`)."""

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    module_qualified_name: str = Field(min_length=1)
    context: SpotlightContext


class ModuleDeepResearchInput(BaseModel):
    """Input contract for step 3 (`module_deep_research`).

    Step 3 surveys the literature **per candidate** (not once per module): it
    iterates `candidates` and builds a candidate-focused prompt for each, so the
    survey is about that candidate's technique/problem. `candidates` is
    therefore the iteration set, not advisory hints.
    """

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    module_qualified_name: str = Field(min_length=1)
    context: SpotlightContext
    repo_path: Path
    # Per-candidate cap (decision D4). The effective cap inside step 3 is
    # `max_findings_per_candidate * len(runners)` per candidate (a union cap
    # across runners), applied at renumber time.
    max_findings_per_candidate: int = Field(default=10, ge=0)

    # The candidates surveyed for this module (decision D2). Each gets its own
    # literature/web survey; findings carry the `candidate_id` they were
    # surveyed for.
    candidates: list[Candidate] = Field(default_factory=list)

    # Controls the default step-3 runner fan-out. False (default) → Codex only;
    # True → Codex + Claude.
    enable_claude_search: bool = False


class ModuleDeepResearchOutput(BaseModel):
    """Output contract for step 3 (`module_deep_research`)."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)
    # Advisory search-query log, one entry per query each runner reported,
    # tagged with the issuing agent. Defaults to [] so older sidecars without
    # it still validate on resume.
    search_queries: list[SearchQueryLog] = Field(default_factory=list)


class ProposalFromFindingCreatorInput(BaseModel):
    """Input contract for step 4 (`proposal_from_finding_creator`).

    Step 4 groups the flat `findings` list by `finding.candidate_id` and judges
    each candidate only against *its own* findings — the pair set is
    `Σ|F_c|`, not the full `|C|×|F|` cartesian product. The target module is
    read off `candidates.module_qualified_name`.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    findings: list[Finding]
    context: SpotlightContext


class ProposalFromFindingCreatorOutput(BaseModel):
    """Output contract for step 4.

    `candidates.state` advances to `FINDING_PROPOSALS_CREATED`. Empty
    `deep_research_proposals` on a candidate is valid.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    issues: list[StepIssue] = Field(default_factory=list)


class AgentProposalsInput(BaseModel):
    """Input contract for step 5 (`agent_proposals`)."""

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    candidates: Candidates
    context: SpotlightContext


class AgentProposalsOutput(BaseModel):
    """Output contract for step 5. State advances to `AGENT_PROPOSALS_CREATED`."""

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    issues: list[StepIssue] = Field(default_factory=list)


class ModuleRun(BaseModel):
    """Per-target-module run record assembled by `SpotlightsManager`.

    `candidates` is `None` only when the module failed before discovery had
    a chance to produce a `Candidates` object.
    """

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(min_length=1)
    status: ModuleRunStatus
    candidates: Candidates | None = None
    findings: list[Finding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)


class SpotlightsManagerInput(BaseModel):
    """Input contract for the top-level `SpotlightsManager`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path
    repo_url: str | None = None
    context: SpotlightContext
    max_findings_per_candidate: int = Field(default=10, ge=0)
    enable_claude_search: bool = False
    continue_on_module_failure: bool = True


class RunInfo(BaseModel):
    """Info about the run that produced a `SpotlightReport`."""

    model_config = ConfigDict(extra="forbid")

    pipeline: Literal["deep_research", "signal"]

    run_id: str = Field(min_length=1)
    started_at: str
    finished_at: str | None = None
    cost_usd: float | None = None


class SpotlightReport(BaseModel):
    """Cross-pipeline output emitted by both pipelines."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"

    project_tree: ProjectTree
    context: SpotlightContext

    candidates: list[Candidate] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    anomalies: list[Anomaly] = Field(default_factory=list)

    run: RunInfo
    issues: list[StepIssue] = Field(default_factory=list)


__all__ = [
    "AgentProposalsInput",
    "AgentProposalsOutput",
    "CandidateDiscoveryInput",
    "ModuleDeepResearchInput",
    "ModuleDeepResearchOutput",
    "ModuleRun",
    "ModulesExtractorInput",
    "ProposalFromFindingCreatorInput",
    "ProposalFromFindingCreatorOutput",
    "RunInfo",
    "SpotlightReport",
    "SpotlightsManagerInput",
]
