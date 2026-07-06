"""spotlights-validation — public API."""
from __future__ import annotations

from pathlib import Path

from spotlights_engine.validation.schemas import (
    PreparationRun,
    ValidationExecutionResult,
    ValidationPreparation,
    ValidationResult,
    ValidationRun,
    ValidationStatus,
)

__all__ = [
    "discover_validation_plan",
    "specialize_validation_plan",
    "get_validation_status",
    "PreparationRun",
    "ValidationExecutionResult",
    "ValidationPreparation",
    "ValidationRun",
    "ValidationResult",
    "ValidationStatus",
]


async def discover_validation_plan(
    source_tree: Path,
    artifacts_dir: Path | None = None,
    candidate: object | None = None,
    change: object | None = None,
) -> PreparationRun:
    """Phase 1: discover test harness and build validation plan.

    In MVP, pass artifacts_dir to load pre-built artifacts from disk
    instead of running live discovery.

    At least one of *candidate* or *change* must be provided to build a
    prioritized validation plan — the plan is scoped to the components
    affected by the candidate/change.
    """
    raise NotImplementedError("discover_validation_plan() is not yet implemented")


async def specialize_validation_plan(
    change: object,
    execution_result: ValidationExecutionResult,
    preparation: PreparationRun,
) -> ValidationRun:
    """Phase 2: execute the validation plan.

    Awaits the preparation internally if Phase 1 is still running.
    """
    raise NotImplementedError("specialize_validation_plan() is not yet implemented")


async def get_validation_status(run_id: str) -> ValidationStatus:
    """Return the current status of a validation run."""
    raise NotImplementedError("get_validation_status() is not yet implemented")
