"""Finding entities and the candidate↔finding edge type.

`Finding` is produced by step 3 (`module_deep_research`). `FindingMatch` is the
edge attached to a `Candidate` by step 4 (`finding_to_candidates_mapper`); it
references `Finding` directly, which is why both live here.
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

MappingConfidence = Literal["high", "medium", "low"]


class Finding(BaseModel):
    """One relevant source-backed finding for a target module."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^find-\d{4}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""


class FindingMatch(BaseModel):
    """One mapped finding edge attached to a candidate.

    Filled in by step 4 (`finding_to_candidates_mapper`). The edge is explicit
    because applicability is a judgment, not a foreign-key join — it carries
    confidence, rationale, and mapper identity.
    """

    model_config = ConfigDict(extra="forbid")

    finding: Finding
    confidence: MappingConfidence
    rationale: str = Field(min_length=1)
    mapped_by: str = Field(min_length=1)


__all__ = [
    "Finding",
    "FindingMatch",
    "FindingSourceType",
    "MappingConfidence",
]
