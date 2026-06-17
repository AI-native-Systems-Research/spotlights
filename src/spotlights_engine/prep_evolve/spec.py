"""`EvolveSpec` — the evolver-agnostic intermediate.

Pydantic models so we get free JSON (de)serialization for `evolve_spec.json`
and round-trip re-render. Field names here are the contract the adapters bind
to; see the implementation plan §3.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _SpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RepoInfo(_SpecModel):
    repo_name: str
    repo_path: str
    repo_summary: str = ""


class SourceRevision(_SpecModel):
    git_commit: str | None
    dirty: bool | None
    captured_at: str


class Objective(_SpecModel):
    goal: str
    workload_hints: list[str] = Field(default_factory=list)
    direction: Literal["minimize", "maximize"]


class MainFile(_SpecModel):
    path: str
    role: str = ""


class ModuleInfo(_SpecModel):
    qualified_name: str
    tree_qualified_name: str
    name: str
    path: str
    description: str = ""
    main_files: list[MainFile] = Field(default_factory=list)


class Oracles(_SpecModel):
    correctness: list[str] = Field(default_factory=list)
    performance: str | None = None


class Target(_SpecModel):
    scope_kind: Literal["candidate", "module_main_file"] = "candidate"
    candidate_id: str | None = None
    file: str
    role: str = ""
    symbol: str = ""
    kind: str = "region"
    line_start: int | None = None
    line_end: int | None = None
    source_excerpt_sha256: str | None = None
    description: str = ""
    current_approach: str = ""
    evolve_rationale: str = ""
    estimated_impact: str = ""
    estimated_impact_explanation: str = ""
    oracles: Oracles = Field(default_factory=Oracles)


class FindingRef(_SpecModel):
    finding_id: str
    title: str
    url: str
    source_type: str
    technique_summary: str
    supporting_evidence: str = ""


class ProposalRef(_SpecModel):
    origin: Literal["deep_research", "agent"]
    agent: str
    title: str
    detailed_description: str
    finding_id: str | None
    rationale: str


class EvolveSpec(_SpecModel):
    run: RepoInfo
    source_revision: SourceRevision
    objective: Objective
    module: ModuleInfo
    targets: list[Target]
    findings: list[FindingRef]
    proposals: list[ProposalRef]
    spec_version: str = "1"


__all__ = [
    "EvolveSpec",
    "FindingRef",
    "MainFile",
    "ModuleInfo",
    "Objective",
    "Oracles",
    "ProposalRef",
    "RepoInfo",
    "SourceRevision",
    "Target",
]
