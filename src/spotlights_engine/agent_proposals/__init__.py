"""Step 5 of the deep-research pipeline: `agent_proposals`.

Public surface:

- `create_agent_proposals(input, *, config)` — architecture-shaped entrypoint
  (`AgentProposalsInput` -> `AgentProposalsOutput`).
- `create_agent_proposals_with_telemetry(input, *, config)` — runtime-rich
  variant returning per-candidate / per-agent durations alongside the output.
- `AgentProposalsConfig` — runtime/infra knobs (paths, parallelism, agent
  knobs, debug flag).
- `AgentProposalsSetupError` / `AgentProposalsValidationError` —
  setup/validation error types.
"""

from __future__ import annotations

from spotlights_engine.agent_proposals.api import (
    AgentProposalsConfig,
    AgentProposalsResult,
    create_agent_proposals,
    create_agent_proposals_with_telemetry,
)
from spotlights_engine.agent_proposals.errors import (
    AgentProposalsSetupError,
    AgentProposalsValidationError,
)

__all__ = [
    "AgentProposalsConfig",
    "AgentProposalsResult",
    "AgentProposalsSetupError",
    "AgentProposalsValidationError",
    "create_agent_proposals",
    "create_agent_proposals_with_telemetry",
]
