"""The unified `Candidate` schema and its location helpers.

`Candidate` is the consumer-facing record carried in `SpotlightReport.candidates`.
The legacy `Candidates` wrapper is retained for the per-step DR pipeline
contracts in `schemas.pipeline` until those are repaired.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

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

CodeKind = Literal[
    "function",
    "method",
    "loop",
    "region",
    "kernel",
    "config_block",
    "plugin_seam",
]


class CodeSpan(BaseModel):
    """One labelled chunk of code at a (line_start, line_end) range within a file."""

    model_config = ConfigDict(extra="forbid")

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    kind: CodeKind


class CodeLocation(BaseModel):
    """A file the candidate touches plus the spans within it."""

    model_config = ConfigDict(extra="forbid")

    file: str = Field(min_length=1)
    spans: list[CodeSpan] = Field(min_length=1)


CandidateOrigin = Literal["telemetry_anomaly", "code_agent"]


class Candidate(BaseModel):
    """A candidate-shaped record in `SpotlightReport.candidates`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")

    module_qualified_name: str | None = None

    origin: CandidateOrigin

    locations: list[CodeLocation] = Field(min_length=1)

    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)

    proposals: list["Proposal"] = Field(default_factory=list)


class Candidates(BaseModel):
    # `extra="forbid"` emits `additionalProperties: false`, which OpenAI's
    # structured-output (used by codex `--output-schema`) requires on every
    # object level. Without it codex returns invalid_json_schema (400).
    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str
    candidates: list[Candidate] = Field(default_factory=list)


from spotlights_engine.schemas.proposal import Proposal  # noqa: E402

Candidate.model_rebuild()


__all__ = [
    "Candidate",
    "CandidateKind",
    "CandidateOrigin",
    "CandidateState",
    "Candidates",
    "CodeKind",
    "CodeLocation",
    "CodeSpan",
    "EstimatedImpact",
]
