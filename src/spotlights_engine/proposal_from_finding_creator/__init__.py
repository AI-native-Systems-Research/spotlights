"""Step 4 of the deep-research pipeline: `proposal_from_finding_creator`.

Public surface mirrors the schemas-only cross-repo convention plus the two
entrypoints other components import directly:

- `create_proposals(input, *, config)` — architecture-shaped entrypoint
  (`ProposalFromFindingCreatorInput` -> `ProposalFromFindingCreatorOutput`).
- `create_proposals_with_telemetry(input, *, config)` — runtime-rich variant
  returning per-pair durations alongside the output (used by the manager).
- `ProposalFromFindingConfig` — runtime/infra knobs (paths, parallelism,
  Claude knobs, debug flag).
- `ProposalFromFindingSetupError` / `ProposalFromFindingValidationError` —
  setup/validation error types.
"""

from __future__ import annotations

from spotlights_engine.proposal_from_finding_creator.api import (
    ProposalFromFindingConfig,
    ProposalFromFindingCreatorResult,
    create_proposals,
    create_proposals_with_telemetry,
)
from spotlights_engine.proposal_from_finding_creator.errors import (
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
)

__all__ = [
    "ProposalFromFindingConfig",
    "ProposalFromFindingCreatorResult",
    "ProposalFromFindingSetupError",
    "ProposalFromFindingValidationError",
    "create_proposals",
    "create_proposals_with_telemetry",
]
