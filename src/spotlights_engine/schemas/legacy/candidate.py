"""The `Candidate` and `Candidates` schemas.

Stage 1 (candidate_discovery) emits a `Candidates` object whose entries are at
state `DISCOVERED` with empty proposal lists. Steps 4 and 5 progressively
populate `deep_research_proposals` (see
`schemas.proposals.DeepResearchProposal`) and `agent_proposals` (see
`schemas.proposals.AgentProposal`) and advance `state`. The shape constraints
captured here are the contract: they are exported via `model_json_schema()`
and consumed by downstream agents.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.schemas.legacy.proposals import AgentProposal, DeepResearchProposal

CandidateKind = Literal[
    "function",
    "method",
    "loop",
    "region",
    "kernel",
    "config_block",
    "plugin_seam",
]
EstimatedImpact = Literal["high", "medium", "low"]
CandidateState = Literal[
    "DISCOVERED",
    "FINDING_PROPOSALS_CREATED",
    "AGENT_PROPOSALS_CREATED",
]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)
    # IDs of `Anomaly` records (from `Signals.anomalies`) that motivated
    # this candidate. Free-form strings to keep `Candidate` decoupled from
    # the Bundle A schema. Empty list when the candidate is technique-driven
    # or otherwise not anomaly-rooted.
    anomaly_refs: list[str] = Field(default_factory=list)
    state: CandidateState = "DISCOVERED"
    deep_research_proposals: list[DeepResearchProposal] = Field(default_factory=list)
    agent_proposals: list[AgentProposal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_range(self) -> Candidate:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Candidates(BaseModel):
    # `extra="forbid"` emits `additionalProperties: false`, which OpenAI's
    # structured-output (used by codex `--output-schema`) requires on every
    # object level. Without it codex returns invalid_json_schema (400).
    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str
    candidates: list[Candidate] = Field(default_factory=list)


__all__ = [
    "Candidate",
    "CandidateKind",
    "CandidateState",
    "Candidates",
    "EstimatedImpact",
]
