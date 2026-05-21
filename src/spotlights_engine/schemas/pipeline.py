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

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import (
    ModuleRunStatus,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.project import ProjectTree


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
    max_findings_per_module: int = Field(default=10, ge=0)


class ModuleDeepResearchOutput(BaseModel):
    """Output contract for step 3 (`module_deep_research`)."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)


class FindingToCandidatesMapperInput(BaseModel):
    """Input contract for step 4 (`finding_to_candidates_mapper`)."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    candidates: Candidates
    context: SpotlightContext


class FindingToCandidatesMapperOutput(BaseModel):
    """Output contract for step 4. `candidates.state` is `FINDINGS_MAPPED`."""

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    issues: list[StepIssue] = Field(default_factory=list)


class ProposalFromFindingCreatorInput(BaseModel):
    """Input contract for step 5 (`proposal_from_finding_creator`)."""

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    context: SpotlightContext


class ProposalFromFindingCreatorOutput(BaseModel):
    """Output contract for step 5.

    `candidates.state` advances to `FINDING_PROPOSALS_CREATED`. Empty
    `deep_research_proposals` on a candidate is valid.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: Candidates
    issues: list[StepIssue] = Field(default_factory=list)


class AgentProposalsInput(BaseModel):
    """Input contract for step 6 (`agent_proposals`)."""

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    candidates: Candidates
    context: SpotlightContext


class AgentProposalsOutput(BaseModel):
    """Output contract for step 6. State advances to `AGENT_PROPOSALS_CREATED`."""

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
    issues: list[StepIssue] = Field(default_factory=list)


class SpotlightsManagerInput(BaseModel):
    """Input contract for the top-level `SpotlightsManager`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path
    context: SpotlightContext
    max_findings_per_module: int = Field(default=10, ge=0)
    continue_on_module_failure: bool = True


class SpotlightsResult(BaseModel):
    """Top-level result echoed back to the caller.

    `context` is echoed verbatim from the input so a result is self-describing
    for audit and repro. `module_runs` is keyed by module qualified name.
    """

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    context: SpotlightContext
    module_runs: dict[str, ModuleRun] = Field(default_factory=dict)


__all__ = [
    "AgentProposalsInput",
    "AgentProposalsOutput",
    "CandidateDiscoveryInput",
    "FindingToCandidatesMapperInput",
    "FindingToCandidatesMapperOutput",
    "ModuleDeepResearchInput",
    "ModuleDeepResearchOutput",
    "ModuleRun",
    "ProposalFromFindingCreatorInput",
    "ProposalFromFindingCreatorOutput",
    "SpotlightsManagerInput",
    "SpotlightsResult",
]
