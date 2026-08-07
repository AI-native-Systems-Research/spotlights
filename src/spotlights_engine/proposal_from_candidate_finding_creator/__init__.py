"""Step 4 of the deep-research pipeline in candidate mode.

Sibling of `proposal_from_finding_creator`: each candidate is paired only with
the findings whose `Finding.candidate_id` names it, and the emitted proposals
carry `mechanism` / `required_changes` / `expected_effect` /
`evaluation_metric`.

Public surface mirrors the existing package:

- `create_proposals(input, *, config)` — architecture-shaped entrypoint
  (`ProposalFromFindingCreatorInput` -> `ProposalFromFindingCreatorOutput`).
- `create_proposals_with_telemetry(input, *, config)` — runtime-rich variant
  returning per-pair durations alongside the output (used by the manager).
- `ProposalFromCandidateFindingConfig` — runtime/infra knobs (paths,
  parallelism, Claude knobs, debug flag).

The setup/validation error types are shared with
`proposal_from_finding_creator` and re-exported here for convenience.
"""

from __future__ import annotations

from spotlights_engine.proposal_from_candidate_finding_creator.api import (
    ProposalFromCandidateFindingConfig,
    ProposalFromCandidateFindingCreatorResult,
    create_proposals,
    create_proposals_with_telemetry,
)
from spotlights_engine.proposal_from_finding_creator.errors import (
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
)

__all__ = [
    "ProposalFromCandidateFindingConfig",
    "ProposalFromCandidateFindingCreatorResult",
    "ProposalFromFindingSetupError",
    "ProposalFromFindingValidationError",
    "create_proposals",
    "create_proposals_with_telemetry",
]
