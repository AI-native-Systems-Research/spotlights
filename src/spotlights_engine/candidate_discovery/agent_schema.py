"""Discovery-only output schema handed to the Claude / codex subprocess agents.

The full `Candidate` schema in `spotlights_engine.schemas.candidate` carries
fields the agent should not emit: `locations` (nested file/span data) and
`proposals` (populated by later pipeline steps), plus a required `origin` and an
optional `module_qualified_name`. Handing the agents that schema would also trip
codex's `--output-schema` strict mode, which rejects `additionalProperties:
false` objects with optional properties.

To avoid that we hand the agents a strict, flat subset (`AgentCandidates`) whose
properties exactly match `required`. The orchestrator promotes each parsed
`AgentCandidate` into a full `Candidate` by wrapping the flat
`file/line_start/line_end/symbol/kind` fields into the nested
`locations=[CodeLocation(...)]` shape (decision D1) and setting
`origin="code_agent"`; `proposals` start empty and are filled by steps 4/5.

The agent keeps emitting **bare** `cand-NNNN` ids (decision D3, option A — the
agent is unaware of module slugs). Promotion to a schema `Candidate` therefore
also prefixes the bare id with the producing module's segment
(`cand-<segment>-NNNN`) via `prefix_local_id`, because the widened
`Candidate.id` pattern rejects bare ids. The prefix is idempotent, so re-running
promotion on an already-prefixed id is a no-op.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from spotlights_engine.schemas.candidate import (
    Candidate,
    CandidateKind,
    Candidates,
    EstimatedImpact,
)
from spotlights_engine.utils.id_helpers import prefix_local_id
from spotlights_engine.utils.schema_compat import make_location


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

    def to_candidate(
        self,
        *,
        module_qualified_name: str | None = None,
        segment: str,
    ) -> Candidate:
        return Candidate(
            id=prefix_local_id(self.id, expected_type="cand", segment=segment),
            module_qualified_name=module_qualified_name,
            origin="code_agent",
            locations=[
                make_location(
                    file=self.file,
                    line_start=self.line_start,
                    line_end=self.line_end,
                    symbol=self.symbol,
                    kind=self.kind,
                )
            ],
            description=self.description,
            current_approach=self.current_approach,
            evolve_rationale=self.evolve_rationale,
            estimated_impact=self.estimated_impact,
            estimated_impact_explanation=self.estimated_impact_explanation,
            proposals=[],
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

    def to_candidates(self, *, segment: str) -> Candidates:
        return Candidates(
            module_qualified_name=self.module_qualified_name,
            candidates=[
                c.to_candidate(
                    module_qualified_name=self.module_qualified_name,
                    segment=segment,
                )
                for c in self.candidates
            ],
        )


__all__ = ["AgentCandidate", "AgentCandidates"]
