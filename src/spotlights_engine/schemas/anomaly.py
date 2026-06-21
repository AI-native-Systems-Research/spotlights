"""`Anomaly` shape carried in `SpotlightReport.anomalies`.

Closed shape covering the fields signal-pipeline anomalies carry today
plus a categorical `estimated_severity` label for consumer-side
ordering. Additional fields are additive — no schema-version bump.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnomalySeverity = Literal["high", "medium", "low"]


class Anomaly(BaseModel):
    """A telemetry anomaly referenced by `Proposal.anomaly_ref_ids`."""

    model_config = ConfigDict(extra="forbid")

    anomaly_id: str = Field(pattern=r"^anom-[A-Za-z0-9._-]+-\d{4}$")
    type: str = Field(min_length=1)
    description: str = ""
    # Renamed from `severity` to mirror `Candidate.estimated_impact` —
    # both are agent-estimated categorical labels, not measured values.
    # Stage-01 prompt asks the agent to fill this; `None` means the
    # agent declined to estimate (e.g. instrumentation-gap anomalies).
    estimated_severity: AnomalySeverity | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_pointer: str = ""
    magnitude: str = ""


__all__ = ["Anomaly", "AnomalySeverity"]
