"""Canonical shared types for the Spotlight Engine.

This module is the only part of `spotlights_engine` that other repos may import.
Everything else in the package is internal and may change without notice.

Layout:
    common     — SpotlightContext, StepIssue, PipelineStep, ModuleRunStatus
    project    — File, Module, Repository, ProjectTree
    finding    — Finding, FindingSourceType
    proposals  — DeepResearchProposal, AgentProposal
    candidate  — Candidate, Candidates, CandidateKind, CandidateState, EstimatedImpact
    pipeline   — Per-step I/O contracts + ModuleRun + SpotlightsManagerInput/Result
"""

from __future__ import annotations

from spotlights_engine.schemas.candidate import (
    Candidate,
    CandidateKind,
    CandidateState,
    Candidates,
    EstimatedImpact,
)
from spotlights_engine.schemas.common import (
    ModuleRunStatus,
    PipelineStep,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.finding import (
    Finding,
    FindingSourceType,
)
from spotlights_engine.schemas.pipeline import (
    AgentProposalsInput,
    AgentProposalsOutput,
    CandidateDiscoveryInput,
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
    ModuleRun,
    ProposalFromFindingCreatorInput,
    ProposalFromFindingCreatorOutput,
    SpotlightsManagerInput,
    SpotlightsResult,
)
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository
from spotlights_engine.schemas.proposals import AgentProposal, DeepResearchProposal

__all__ = [
    "AgentProposal",
    "AgentProposalsInput",
    "AgentProposalsOutput",
    "Candidate",
    "CandidateDiscoveryInput",
    "CandidateKind",
    "CandidateState",
    "Candidates",
    "DeepResearchProposal",
    "EstimatedImpact",
    "File",
    "Finding",
    "FindingSourceType",
    "Module",
    "ModuleDeepResearchInput",
    "ModuleDeepResearchOutput",
    "ModuleRun",
    "ModuleRunStatus",
    "PipelineStep",
    "ProjectTree",
    "ProposalFromFindingCreatorInput",
    "ProposalFromFindingCreatorOutput",
    "Repository",
    "SpotlightContext",
    "SpotlightsManagerInput",
    "SpotlightsResult",
    "StepIssue",
]
