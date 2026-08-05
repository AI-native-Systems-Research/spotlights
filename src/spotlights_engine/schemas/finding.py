"""Finding entity produced by the per-candidate deep-research step.

`Finding` is produced by step 3 (`module_deep_research`), which now surveys the
literature **once per candidate**: every finding carries the `candidate_id` it
was surveyed for. Step 4 (`proposal_from_finding_creator`) groups the findings
by `candidate_id` and only judges each candidate against its own findings when
drafting proposals.
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

    # The candidate this finding was surveyed for (decision D2). Optional on the
    # public schema so older `SpotlightReport`s / ad hoc sidecars still load;
    # every new step-3 run sets it. Step 4 groups findings by this id.
    candidate_id: str | None = Field(
        default=None, pattern=r"^cand-[A-Za-z0-9._-]+-\d{4}$"
    )


__all__ = [
    "Finding",
    "FindingSourceType",
]
