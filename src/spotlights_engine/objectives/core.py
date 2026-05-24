"""Core functions for objective setting (Bundle G)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from spotlights_engine.objectives.schemas import (
    Objective,
    ObjectiveIntent,
    ObjectiveProposal,
)


def build_intent(
    *,
    target_metric: str,
    target_direction: str,
    workload_classes: list[str],
    priorities: list[str] | None = None,
    notes: str = "",
) -> ObjectiveIntent:
    """Validate inputs and construct an ObjectiveIntent."""
    return ObjectiveIntent(
        target_metric=target_metric,
        target_direction=target_direction,  # type: ignore[arg-type]
        workload_classes=workload_classes,
        priorities=priorities or [],
        notes=notes,
    )


def assemble_proposal(intent: ObjectiveIntent) -> ObjectiveProposal:
    """Produce an ObjectiveProposal from a validated intent."""
    direction_word = "reducing" if intent.target_direction == "minimize" else "increasing"
    workloads_str = ", ".join(intent.workload_classes)

    rationale = (
        f"This session will focus on {direction_word} {intent.target_metric} "
        f"for {workloads_str} workloads. "
        f"The system will determine which components to examine and what tests to run. "
        f"No metric may regress as a result of any change."
    )

    coverage_notes: list[str] = []
    if intent.priorities:
        priorities_str = ", ".join(intent.priorities)
        coverage_notes.append(
            f"Secondary priorities ({priorities_str}) will be considered when ranking "
            f"candidate optimizations."
        )

    return ObjectiveProposal(
        intent=intent,
        rationale=rationale,
        coverage_notes=coverage_notes,
    )


def finalize_objective(
    proposal: ObjectiveProposal,
    *,
    session_id: str,
    approved_by: str,
) -> Objective:
    """Produce the final frozen Objective from an approved proposal."""
    return Objective(
        objective_id=str(uuid.uuid4()),
        session_id=session_id,
        created_at=datetime.now(UTC),
        approved_by=approved_by,
        intent=proposal.intent,
        rationale=proposal.rationale,
    )
