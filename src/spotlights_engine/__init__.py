"""Spotlight Engine: spine for schemas, orchestration, and integration."""

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    AgentProposalsSetupError,
    AgentProposalsValidationError,
    create_agent_proposals,
    create_agent_proposals_with_telemetry,
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

__version__ = "0.0.0"

__all__ = [
    "AgentProposalsConfig",
    "AgentProposalsSetupError",
    "AgentProposalsValidationError",
    "ManagerSetupError",
    "ModuleFilter",
    "ModuleTelemetry",
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
