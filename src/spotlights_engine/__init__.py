"""Spotlight Engine: spine for schemas, orchestration, and integration."""

# ruff: noqa: E402 - configure package logger before importing submodules.

import logging as _logging

# Library default: silence package-level logs unless the application opts in.
# Avoids stderr noise from Python's `lastResort` handler for WARNING+ records
# when the host application has not configured logging.
_logging.getLogger("spotlights_engine").addHandler(_logging.NullHandler())

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    AgentProposalsSetupError,
    AgentProposalsValidationError,
    create_agent_proposals,
    create_agent_proposals_with_telemetry,
)
from spotlights_engine.module_knowledge import (
    KnowledgeBase,
    KnowledgeRecord,
    RetrievedItem,
    RetrieveRequest,
)
from spotlights_engine.proposal_from_candidate_finding_creator import (
    ProposalFromCandidateFindingConfig,
)
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
    create_proposals,
    create_proposals_with_telemetry,
)
from spotlights_engine.spotlights_manager import (
    ManagerSetupError,
    ModuleFilter,
    ModuleTelemetry,
    ResumeMismatchError,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
    run,
    run_with_telemetry,
)

__version__ = "0.1.0"

__all__ = [
    "AgentProposalsConfig",
    "AgentProposalsSetupError",
    "AgentProposalsValidationError",
    "KnowledgeBase",
    "KnowledgeRecord",
    "RetrieveRequest",
    "RetrievedItem",
    "ManagerSetupError",
    "ModuleFilter",
    "ModuleTelemetry",
    "ProposalFromCandidateFindingConfig",
    "ProposalFromFindingConfig",
    "ProposalFromFindingSetupError",
    "ProposalFromFindingValidationError",
    "ResumeMismatchError",
    "SpotlightsManagerConfig",
    "SpotlightsManagerResult",
    "create_agent_proposals",
    "create_agent_proposals_with_telemetry",
    "create_proposals",
    "create_proposals_with_telemetry",
    "run",
    "run_with_telemetry",
]
