"""Cross-cutting types used across multiple pipeline steps.

`SpotlightContext` is the caller-supplied run context threaded into the
discovery, deep-research, and proposal steps. `StepIssue`, `PipelineStep`,
and `ModuleRunStatus` are the run-status vocabulary the manager and per-step
modules share when reporting recoverable or fatal problems.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

PipelineStep = Literal[
    "modules_extractor",
    "candidate_discovery",
    "module_deep_research",
    "proposal_from_finding_creator",
    "agent_proposals",
]

ModuleRunStatus = Literal["SUCCEEDED", "DEGRADED", "SKIPPED", "FAILED"]


class SpotlightContext(BaseModel):
    """Caller-supplied run context threaded into downstream judgement steps."""

    model_config = ConfigDict(extra="forbid")

    objective: str = Field(min_length=1)
    workload_hints: list[str] = Field(default_factory=list)
    validation_plan: list[str] = Field(default_factory=list)


class StepIssue(BaseModel):
    """Recoverable or fatal issue observed while running a pipeline step."""

    model_config = ConfigDict(extra="forbid")

    step: PipelineStep
    severity: Literal["warning", "error"]
    message: str = Field(min_length=1)
    recoverable: bool = True


__all__ = [
    "ModuleRunStatus",
    "PipelineStep",
    "SpotlightContext",
    "StepIssue",
]
