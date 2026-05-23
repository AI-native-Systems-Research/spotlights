"""Validation of one per-candidate agent response into 0..1 AgentProposals.

The agent is prompted (and the JSON schema enforces) a wrapper object
`{"proposals": [...]}` with a 0..1-length array. This module turns whatever
came back into a `(proposals, warnings)` tuple — defensive because schema
enforcement at the agent edge can be loose, especially on fallback paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from spotlights_engine.schemas.proposals import AgentProposal


@dataclass
class PerAgentParseResult:
    """Outcome of validating one (candidate, agent) structured response."""

    proposals: list[AgentProposal]
    warnings: list[str]


def parse_candidate_payload(
    payload: Any,
    *,
    candidate_id: str,
    agent_name: str,
) -> PerAgentParseResult:
    """Coerce the parsed payload into 0..1 `AgentProposal` entries.

    Accepts either the schema's wrapper object (`{"proposals": [...]}`) or a
    bare list. Returns the surviving proposals plus a list of human-readable
    warnings to surface as recoverable `StepIssue`s.
    """
    warnings: list[str] = []
    cand_tag = f"(candidate_id={candidate_id}, agent={agent_name})"

    items: list | None = None
    if isinstance(payload, dict):
        proposals_field = payload.get("proposals")
        if isinstance(proposals_field, list):
            items = proposals_field
        else:
            return PerAgentParseResult(
                proposals=[],
                warnings=[
                    f"agent payload wrapper missing list 'proposals' {cand_tag}: "
                    f"got {type(proposals_field).__name__}"
                ],
            )
    elif isinstance(payload, list):
        items = payload
    else:
        return PerAgentParseResult(
            proposals=[],
            warnings=[
                f"agent payload was not an object or array {cand_tag}: "
                f"got {type(payload).__name__}"
            ],
        )

    if len(items) == 0:
        return PerAgentParseResult(proposals=[], warnings=[])

    if len(items) > 1:
        warnings.append(
            f"agent emitted {len(items)} proposals for one candidate {cand_tag}; "
            "keeping the first"
        )
        items = items[:1]

    raw = items[0]
    if not isinstance(raw, dict):
        return PerAgentParseResult(
            proposals=[],
            warnings=warnings
            + [
                f"agent payload entry was not an object {cand_tag}: "
                f"got {type(raw).__name__}"
            ],
        )

    raw = dict(raw)
    raw_agent_name = raw.get("agent_name")
    if raw_agent_name != agent_name:
        warnings.append(
            f"agent_name mismatch {cand_tag}: got {raw_agent_name!r}; "
            f"normalizing to {agent_name!r}"
        )
        raw["agent_name"] = agent_name

    try:
        proposal = AgentProposal.model_validate(raw)
    except ValidationError as exc:
        warnings.append(
            f"agent payload failed schema validation {cand_tag}: {exc}"
        )
        return PerAgentParseResult(proposals=[], warnings=warnings)

    return PerAgentParseResult(proposals=[proposal], warnings=warnings)


__all__ = ["PerAgentParseResult", "parse_candidate_payload"]
