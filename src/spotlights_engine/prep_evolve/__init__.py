"""prep-evolve: turn a selected Spotlights candidate into a ready-to-run evolve
bundle for skydiscover / CORAL / Nous.

See `design/integration_with_evolvers_impl_plan.md` for the contract.
"""

from __future__ import annotations

from spotlights_engine.prep_evolve.api import (
    BundleResult,
    PrepEvolveConfig,
    PrepEvolveInput,
    PrepEvolveResult,
    SkippedEvolver,
    prep_evolve,
)
from spotlights_engine.prep_evolve.errors import (
    GeneratedPathError,
    PrepEvolveError,
    RepoResolutionError,
    ScopeError,
    SelectionError,
    StalenessError,
    UnsupportedEvolverError,
)
from spotlights_engine.prep_evolve.spec import EvolveSpec

__all__ = [
    "BundleResult",
    "EvolveSpec",
    "GeneratedPathError",
    "PrepEvolveConfig",
    "PrepEvolveError",
    "PrepEvolveInput",
    "PrepEvolveResult",
    "RepoResolutionError",
    "ScopeError",
    "SelectionError",
    "SkippedEvolver",
    "StalenessError",
    "UnsupportedEvolverError",
    "prep_evolve",
]
