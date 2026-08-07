"""Finding entity produced by the per-module deep-research step.

`Finding` is produced by step 3 (`module_deep_research`). Step 4
(`proposal_from_finding_creator`) consumes the findings list directly and
considers each `(candidate, finding)` pair when drafting proposals.

`candidate_id` is populated only in `candidate` deep-research mode (step 3
`candidate_deep_research`), where each survey is scoped to one candidate. In
`module` mode the field is always `None`.
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

    finding_id: str = Field(pattern=r"^find-[A-Za-z0-9._-]+-\d{4}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""

    # Set only by `candidate_deep_research` (candidate mode); `None` in module
    # mode, where a survey covers the whole module rather than one candidate.
    candidate_id: str | None = Field(default=None, pattern=r"^cand-[A-Za-z0-9._-]+-\d{4}$")


__all__ = [
    "Finding",
    "FindingSourceType",
]
