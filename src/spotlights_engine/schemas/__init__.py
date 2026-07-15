"""Canonical shared types for the Spotlight Engine.

This module is the only part of `spotlights_engine` that other repos may import.
Everything else in the package is internal and may change without notice.

Layout:
    common     — SpotlightContext, StepIssue, PipelineStep, ModuleRunStatus
    project    — File, Module, Repository, ProjectTree
    finding    — Finding, FindingSourceType
    search     — SearchResult, SearchQueryLog (deep-research search log)
    proposals  — DeepResearchProposal, AgentProposal (DR pipeline-internal)
    proposal   — Proposal, ProposalSource (unified report shape)
    candidate  — Candidate, Candidates, CandidateKind, CandidateState,
                 EstimatedImpact, CodeKind, CodeSpan, CodeLocation,
                 CandidateOrigin
    pipeline   — Per-step DR I/O contracts + ModuleRun + SpotlightsManagerInput;
                 plus the cross-pipeline `SpotlightReport` and `RunInfo`.
"""

from __future__ import annotations

from spotlights_engine.objectives.schemas import Objective, ObjectiveIntent
from spotlights_engine.schemas.anomaly import Anomaly, AnomalySeverity
from spotlights_engine.schemas.candidate import (
    Candidate,
    CandidateKind,
    CandidateOrigin,
    Candidates,
    CandidateState,
    CodeKind,
    CodeLocation,
    CodeSpan,
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
    RunInfo,
    SpotlightReport,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import File, Module, ProjectTree, Repository
from spotlights_engine.schemas.proposal import Proposal, ProposalSource
from spotlights_engine.schemas.proposals import AgentProposal, DeepResearchProposal
from spotlights_engine.schemas.search import SearchQueryLog, SearchResult

__all__ = [
    "AgentProposal",
    "AgentProposalsInput",
    "AgentProposalsOutput",
    "Anomaly",
    "AnomalySeverity",
    "Candidate",
    "CandidateDiscoveryInput",
    "CandidateKind",
    "CandidateOrigin",
    "CandidateState",
    "Candidates",
    "CodeKind",
    "CodeLocation",
    "CodeSpan",
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
    "Objective",
    "ObjectiveIntent",
    "PipelineStep",
    "ProjectTree",
    "Proposal",
    "ProposalFromFindingCreatorInput",
    "ProposalFromFindingCreatorOutput",
    "ProposalSource",
    "Repository",
    "RunInfo",
    "SearchQueryLog",
    "SearchResult",
    "SpotlightContext",
    "SpotlightReport",
    "SpotlightsManagerInput",
    "StepIssue",
]
