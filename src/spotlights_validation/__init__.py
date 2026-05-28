"""spotlights-validation — public API."""
from __future__ import annotations

from pathlib import Path

from spotlights_validation.schemas import (
    ExecutionResult,
    PreparationRun,
    ValidationPreparation,
    ValidationResult,
    ValidationRun,
    ValidationStatus,
)

__all__ = [
    "prepare",
    "start_validation",
    "get_validation_status",
    "PreparationRun",
    "ValidationPreparation",
    "ValidationRun",
    "ValidationResult",
    "ValidationStatus",
    "ExecutionResult",
]


async def prepare(
    source_tree: Path,
    artifacts_dir: Path | None = None,
) -> PreparationRun:
    """Phase 1: discover test harness and build validation plan.

    In MVP, pass artifacts_dir to load pre-built artifacts from disk
    instead of running live discovery.
    """
    raise NotImplementedError("prepare() is not yet implemented")


async def start_validation(
    change: object,
    execution_result: ExecutionResult,
    preparation: PreparationRun,
) -> ValidationRun:
    """Phase 2: execute the validation plan.

    Awaits the preparation internally if Phase 1 is still running.
    """
    raise NotImplementedError("start_validation() is not yet implemented")


async def get_validation_status(run_id: str) -> ValidationStatus:
    """Return the current status of a validation run."""
    raise NotImplementedError("get_validation_status() is not yet implemented")
