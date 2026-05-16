"""The `Candidate` and `Candidates` schemas.

Stage 1 of the candidate research proposer emits a `Candidates` JSON object as
its final artifact. The shape constraints captured here are the contract: they
are exported via `model_json_schema()` and handed to both subprocess agents so
the same validation runs model-side and orchestrator-side.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CandidateKind = Literal[
    "function",
    "method",
    "loop",
    "region",
    "kernel",
    "config_block",
    "plugin_seam",
]
MetricDirection = Literal["minimize", "maximize"]
EstimatedImpact = Literal["high", "medium", "low"]


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    direction: MetricDirection
    # Required (not optional) so OpenAI strict structured-outputs accepts the
    # schema: it demands every property appear in `required`. Nullable still
    # lets the model emit `null` when no baseline is known.
    target_or_baseline: str | None = Field(...)


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
    metrics: list[Metric] = Field(min_length=1)
    estimated_impact: EstimatedImpact

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
    candidates: list[Candidate] = Field(min_length=1)


__all__ = ["Candidate", "Candidates", "Metric"]
