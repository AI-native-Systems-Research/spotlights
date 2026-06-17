"""Discovery-only output schema handed to the Claude / codex subprocess agents.

The full `Candidate` schema in `spotlights_engine.schemas.legacy.candidate` carries
fields (`state`, `deep_research_proposals`, `agent_proposals`) that are only
filled in by later pipeline steps. They default to sensible empty values,
which makes pydantic emit a JSON schema where those properties are *not* in
`required`. Codex's `--output-schema` forwards the schema to OpenAI
structured-output, whose strict mode rejects schemas with
`additionalProperties: false` plus optional properties.

To avoid that breakage we hand the agents a strict subset (`AgentCandidates`)
whose properties exactly match `required`. The orchestrator promotes each
parsed `AgentCandidate` into a full `Candidate` with `state="DISCOVERED"`
and empty proposal lists.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.schemas.legacy.candidate import (
    Candidate,
    CandidateKind,
    Candidates,
    EstimatedImpact,
)


class AgentCandidate(BaseModel):
    """Subset of `Candidate` that the discovery agents must emit verbatim."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_range(self) -> AgentCandidate:
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self

    def to_candidate(self) -> Candidate:
        return Candidate(
            id=self.id,
            file=self.file,
            line_start=self.line_start,
            line_end=self.line_end,
            symbol=self.symbol,
            kind=self.kind,
            description=self.description,
            current_approach=self.current_approach,
            evolve_rationale=self.evolve_rationale,
            estimated_impact=self.estimated_impact,
            estimated_impact_explanation=self.estimated_impact_explanation,
        )


class AgentCandidates(BaseModel):
    """Top-level shape the agents emit; promoted to `Candidates` after parsing.

    Both fields are required in the generated JSON schema (no defaults) so
    the strict-mode structured-output validator on the codex side accepts the
    schema. The agent may legitimately emit `"candidates": []` — empty is
    valid; missing is not.
    """

    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str = Field(min_length=1)
    candidates: list[AgentCandidate]

    def to_candidates(self) -> Candidates:
        return Candidates(
            module_qualified_name=self.module_qualified_name,
            candidates=[c.to_candidate() for c in self.candidates],
        )


__all__ = ["AgentCandidate", "AgentCandidates"]
