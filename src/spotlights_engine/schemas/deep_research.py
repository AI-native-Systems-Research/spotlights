"""Schemas for the module deep-research step."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.modules import ProjectTree

ProjectModules = ProjectTree

FindingSourceType = Literal[
    "paper",
    "blog",
    "docs",
    "issue",
    "pr",
    "talk",
    "codebase",
    "other",
]

PipelineStep = Literal[
    "modules_extractor",
    "candidate_discovery",
    "module_deep_research",
    "finding_to_candidates_mapper",
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


class Finding(BaseModel):
    """One relevant source-backed finding for a target module."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^find-\d{4}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""


class StepIssue(BaseModel):
    """Recoverable or fatal issue observed while running a pipeline step."""

    model_config = ConfigDict(extra="forbid")

    step: PipelineStep
    severity: Literal["warning", "error"]
    message: str = Field(min_length=1)
    recoverable: bool = True


class ModuleDeepResearchInput(BaseModel):
    """Input contract for the per-module literature and web survey step."""

    model_config = ConfigDict(extra="forbid")

    project_tree: ProjectTree
    module_qualified_name: str = Field(min_length=1)
    context: SpotlightContext
    repo_path: Path
    max_findings_per_module: int = Field(default=10, ge=0)


class ModuleDeepResearchOutput(BaseModel):
    """Output contract for the per-module literature and web survey step."""

    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)


__all__ = [
    "Finding",
    "FindingSourceType",
    "ModuleDeepResearchInput",
    "ModuleDeepResearchOutput",
    "ModuleRunStatus",
    "PipelineStep",
    "ProjectModules",
    "SpotlightContext",
    "StepIssue",
]
