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

    # When True, run deep research once PER candidate (each research call is
    # scoped to a single hot spot) instead of one call spanning the whole module
    # with every candidate listed as a hot spot. Findings from all per-candidate
    # runs are merged/deduped/capped together. Applies to every runner
    # (Codex/Claude/OpenAlex). No-op when there are no candidates.
    per_candidate_deep_research: bool = False

    # Controls the default step-3 runner fan-out. False (default) → Codex only;
    # True → Codex + Claude.
    enable_claude_search: bool = False

    # Opt-in OpenAlex runner. When True the default fan-out adds an OpenAlex
    # runner alongside Codex (and Claude, if enabled). Uses the OPENALEX_API_KEY
    # env var by default; `openalex_model` overrides the SDK default model.
    enable_openalex: bool = False
    openalex_model: str | None = None
    # How the OpenAlex runner builds its OpenAlex query from the prompt:
    # "regex" (deterministic) or "codex" (Codex query-writer + regex fallback).
    openalex_query_mode: str = "codex"


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
    """Input contract for the top-level `SpotlightsManager`."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    repo_path: Path
    repo_url: str | None = None
    context: SpotlightContext
    max_findings_per_module: int = Field(default=30, ge=0)
    include_candidate_hotspots: bool = True
    # Run step 3 once per candidate (scoped research) instead of once per module.
    per_candidate_deep_research: bool = False
    enable_claude_search: bool = False
    enable_openalex: bool = False
    openalex_model: str | None = None
    openalex_query_mode: str = "codex"
    # Determinism stress test: run step 3 (module_deep_research) this many times
    # per module. 1 (default) = normal single pass. When >1 the FIRST pass is the
    # canonical result used by every downstream step; each extra pass is written
    # to its own indexed sidecar (module_deep_research.{i}.json, i=1..N-1) purely
    # so the repeated outputs can be diffed for run-to-run determinism. Applies to
    # whatever step-3 runner set is active (Codex/Claude/OpenAlex). Each extra
    # pass issues real agent/API calls, so N>1 multiplies step-3 cost.
    deep_research_repeat: int = Field(default=1, ge=1)
    # False -> step 3 (module_deep_research) is not run for any module; the
    # manager substitutes an empty ModuleDeepResearchOutput and step 4 takes
    # its zero-findings short-circuit.
    enable_deep_research: bool = True
    # False -> step 4 (proposal_from_finding_creator) pairing is skipped: step 3
    # findings are kept, but no per-pair Claude session runs. Every candidate
    # advances with zero finding-derived proposals (same synthetic empty output
    # as the zero-findings short-circuit). Steps 1, 2, 3, 5 still run.
    enable_proposals_from_findings: bool = True
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
