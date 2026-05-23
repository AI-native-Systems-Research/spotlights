"""Validation of one per-pair Claude response into 0..1 proposals.

The agent is prompted (and the JSON schema enforces) a 0..1-length list of
`DeepResearchProposal` for the single (candidate, finding) pair. This module
turns whatever came back into a `(proposals, warnings)` tuple — defensive
because the schema is enforced by Claude, but the fallback text-path can
still drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from spotlights_engine.schemas.proposals import DeepResearchProposal


@dataclass
class PairParseResult:
    """Outcome of validating one pair's structured response."""

    proposals: list[DeepResearchProposal]
    warnings: list[str]


def parse_pair_payload(
    payload: Any,
    *,
    candidate_id: str,
    finding_id: str,
    created_by: str,
) -> PairParseResult:
    """Coerce the parsed payload into 0..1 `DeepResearchProposal` entries.

    `payload` is whatever came back on `structured_output` (already JSON-
    decoded). Returns the surviving proposals plus a list of human-readable
    warnings to surface as recoverable `StepIssue`s.
    """
    warnings: list[str] = []
    pair_tag = f"(candidate_id={candidate_id}, finding_id={finding_id})"

    if not isinstance(payload, list):
        return PairParseResult(
            proposals=[],
            warnings=[
                f"agent payload was not a JSON array {pair_tag}: "
                f"got {type(payload).__name__}"
            ],
        )

    if len(payload) == 0:
        return PairParseResult(proposals=[], warnings=[])

    if len(payload) > 1:
        warnings.append(
            f"agent emitted {len(payload)} proposals for one pair {pair_tag}; "
            "keeping the first"
        )
        payload = payload[:1]

    raw = payload[0]
    if not isinstance(raw, dict):
        return PairParseResult(
            proposals=[],
            warnings=warnings
            + [
                f"agent payload entry was not an object {pair_tag}: "
                f"got {type(raw).__name__}"
            ],
        )

    # Coerce mismatched / fallback-path values back to the canonical ones so
    # downstream attribution stays deterministic.
    raw = dict(raw)
    raw_finding_id = raw.get("finding_id")
    if raw_finding_id != finding_id:
        warnings.append(
            f"agent finding_id mismatch {pair_tag}: got {raw_finding_id!r}; "
            "dropping proposal"
        )
        return PairParseResult(proposals=[], warnings=warnings)

    raw_created_by = raw.get("created_by")
    if raw_created_by != created_by:
        warnings.append(
            f"agent created_by mismatch {pair_tag}: got {raw_created_by!r}; "
            f"normalizing to {created_by!r}"
        )
        raw["created_by"] = created_by

    try:
        proposal = DeepResearchProposal.model_validate(raw)
    except ValidationError as exc:
        warnings.append(
            f"agent payload failed schema validation {pair_tag}: {exc}"
        )
        return PairParseResult(proposals=[], warnings=warnings)

    return PairParseResult(proposals=[proposal], warnings=warnings)


__all__ = ["PairParseResult", "parse_pair_payload"]
