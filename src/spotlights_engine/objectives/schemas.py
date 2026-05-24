"""Objective setting schemas (Bundle G)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ObjectiveIntent(BaseModel):
    """Structured intent produced from the PM interview."""

    target_metric: str = Field(min_length=1)
    target_direction: Literal["minimize", "maximize"]
    workload_classes: list[str] = Field(min_length=1)
    priorities: list[str] = Field(default_factory=list)
    notes: str = ""


class ObjectiveProposal(BaseModel):
    """Draft objective presented to the PM for review before finalization."""

    intent: ObjectiveIntent
    rationale: str = Field(min_length=1)
    coverage_notes: list[str] = Field(default_factory=list)
    requires_approval: bool = True


class Objective(BaseModel, frozen=True):
    """Final, immutable objective for a discovery session."""

    objective_id: str
    session_id: str
    created_at: datetime
    approved_by: str
    intent: ObjectiveIntent
    rationale: str
    frozen: bool = True
