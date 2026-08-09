"""Per-step pipeline I/O contracts for the spotlights deep-research path.

Mirrors the input/output classes shown in
`docs/architecture/spotlights_deep_research_path_architecture.md`. The data
types they reference live in `schemas.project`, `schemas.common`,
`schemas.finding`, and `schemas.candidate`; this module is purely the
boundary contracts so the manager and per-step modules can talk to one
another with typed payloads.
"""

from __future__ import annotations

from datetime import date
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
    """Input contract for step 3 (`module_deep_research`)."""

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    module_qualified_name: str = Field(min_length=1)
    context: SpotlightContext
    repo_path: Path
    max_findings_per_module: int = Field(default=30, ge=0)

    # Hot spots surfaced by step 2 (`candidate_discovery`) for this module.
    # When `include_candidate_hotspots` is true and this list is non-empty,
    # the prompt shows them to the research agent so its search can be steered
    # toward the symbols the discovery step already flagged as worth evolving.
    candidates: list[Candidate] = Field(default_factory=list)
    include_candidate_hotspots: bool = True

    # Controls the default step-3 runner fan-out. False (default) → Codex only;
    # True → Codex + Claude.
    enable_claude_search: bool = False

    # When set, restricts findings to sources first published or released
    # strictly before this date. Enforced in two ways: (a) the DR prompt
    # instructs the agent to honor the cutoff and populate `publication_date`
    # on every finding, and (b) the merge step drops any finding whose
    # self-reported `publication_date` falls on or after the cutoff. When
    # None (the default), the DR step behaves identically to runs that never
    # knew about the field. Useful for historical baselining and "would the
    # agent derive X without seeing later-published solutions" tests.
    source_cutoff_date: date | None = None

    # Directory into which per-runner raw stdout / stderr / final-message /
    # summary files are written, one set per CLI runner. When `None` (the
    # default), no per-runner artifacts are written — preserves the historical
    # behavior for standalone DR callers. The manager always supplies one so
    # runner failures (empty envelope, transport errors, quota timeouts) can
    # be diagnosed after the fact.
    runner_artifacts_dir: Path | None = None


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

    Step 4 considers every `(candidate, finding)` pair directly: there is no
    separate mapping step. The target module is read off
    `candidates.module_qualified_name`.
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
    """Input contract for the top-level `SpotlightsManager`.

    `dr_source_cutoff_date` is threaded through to every per-module
    `ModuleDeepResearchInput.source_cutoff_date`; see that field's docstring
    for semantics. Unset (None) means no cutoff, which is the historical
    behavior.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path
    repo_url: str | None = None
    context: SpotlightContext
    max_findings_per_module: int = Field(default=30, ge=0)
    include_candidate_hotspots: bool = True
    enable_claude_search: bool = False
    continue_on_module_failure: bool = True
    dr_source_cutoff_date: date | None = None


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
