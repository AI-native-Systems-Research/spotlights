"""Proposal kinds attached to a `Candidate` by the late pipeline steps.

`DeepResearchProposal` is filled in by step 4 — either
`proposal_from_finding_creator` (module mode) or
`proposal_from_candidate_finding_creator` (candidate mode) — and cites the
originating finding. `AgentProposal` is filled in by step 5 (`agent_proposals`)
for ideas that come from agent knowledge rather than the research findings.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DeepResearchProposal(BaseModel):
    """A research-backed proposal attached to a candidate.

    Filled in by step 4. Each proposal cites the originating `finding_id` so
    its evidence trail is preserved.

    The four structured fields (`mechanism`, `required_changes`,
    `expected_effect`, `evaluation_metric`) are emitted only by
    `proposal_from_candidate_finding_creator` (candidate mode); the module-mode
    agent schema does not reference them, so they stay `None` there.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    detailed_description: str = Field(min_length=1)
    finding_id: str = Field(pattern=r"^find-[A-Za-z0-9._-]+-\d{4}$")
    proposal_rationale: str = Field(min_length=1)
    created_by: str = Field(min_length=1)

    mechanism: str | None = Field(default=None, min_length=1)
    required_changes: str | None = Field(default=None, min_length=1)
    expected_effect: str | None = Field(default=None, min_length=1)
    evaluation_metric: str | None = Field(default=None, min_length=1)


class AgentProposal(BaseModel):
    """An agent-knowledge proposal not derived from research findings.

    Filled in by step 5 (`agent_proposals`). `novelty_rationale` records why
    this proposal is not already covered by `deep_research_proposals`.
    """

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    detailed_description: str = Field(min_length=1)
    agent_name: str = Field(min_length=1)
    novelty_rationale: str = Field(min_length=1)


__all__ = [
    "AgentProposal",
    "DeepResearchProposal",
]
