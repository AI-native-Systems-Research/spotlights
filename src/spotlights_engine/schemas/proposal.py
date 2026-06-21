"""Unified `Proposal` shape carried under `Candidate.proposals`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ProposalSource = Literal["research_finding", "agent_knowledge", "telemetry_anomaly"]


class Proposal(BaseModel):
    """The unified proposal shape carried under each `Candidate`."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^prop-[A-Za-z0-9._-]+-\d{4}$")

    source: ProposalSource

    anomaly_ref_ids: list[str] | None = None
    finding_ref_id: str | None = None

    author: str | None = None

    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    mechanism: str | None = None
    required_changes: str | None = None
    expected_effect: str | None = None
    evaluation_metric: str | None = None


__all__ = ["Proposal", "ProposalSource"]
