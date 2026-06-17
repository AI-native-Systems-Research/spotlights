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

from spotlights_engine.schemas.legacy.candidate import (
    Candidate,
    CandidateKind,
    CandidateState,
    Candidates,
    EstimatedImpact,
)
from spotlights_engine.schemas.legacy.common import (
    ModuleRunStatus,
    PipelineStep,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.legacy.finding import (
    Finding,
    FindingSourceType,
)
from spotlights_engine.schemas.legacy.pipeline import (
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
from spotlights_engine.schemas.legacy.project import File, Module, ProjectTree, Repository
from spotlights_engine.schemas.legacy.proposals import AgentProposal, DeepResearchProposal
from spotlights_engine.objectives.schemas import Objective, ObjectiveIntent

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
    "Objective",
    "ObjectiveIntent",
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
