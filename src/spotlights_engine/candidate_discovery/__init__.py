"""Stage 1 of candidate_research_proposer: candidate discovery.

This package is the *second* public surface of `spotlights_engine` beyond
`schemas/` — the architecture doc otherwise designates `schemas/` as the only
contract surface other repos may import. The exception is intentional per
spec §1: `discover_candidates` (the architecture-shaped entrypoint matching
`docs/architecture/spotlights_deep_research_path_architecture.md` step 2)
and `discover` (the runtime-rich entrypoint with telemetry / tunable knobs)
must be importable cross-repo.

A future reviewer expecting the schemas-only cross-repo contract should not
mistake this re-export for a regression.
"""

from __future__ import annotations

from spotlights_engine.candidate_discovery.api import (
    DiscoveryConfig,
    DiscoveryResult,
    IterationTelemetry,
    discover,
    discover_candidates,
)
from spotlights_engine.candidate_discovery.errors import (
    DiscoveryMutationError,
    DiscoverySetupError,
    DiscoveryValidationError,
)

__all__ = [
    "DiscoveryConfig",
    "DiscoveryMutationError",
    "DiscoveryResult",
    "DiscoverySetupError",
    "DiscoveryValidationError",
    "IterationTelemetry",
    "discover",
    "discover_candidates",
]
