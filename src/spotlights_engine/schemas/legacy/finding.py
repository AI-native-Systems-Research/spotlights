"""Finding entity produced by the per-module deep-research step.

`Finding` is produced by step 3 (`module_deep_research`). Step 4
(`proposal_from_finding_creator`) consumes the findings list directly and
considers each `(candidate, finding)` pair when drafting proposals.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FindingSourceType = Literal[
    "paper",
    "blog",
    "docs",
    "issue",
    "pr",
    "talk",
    "codebase",
    "other",
]


class Finding(BaseModel):
    """One relevant source-backed finding for a target module."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^find-\d{4}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""


__all__ = [
    "Finding",
    "FindingSourceType",
]
