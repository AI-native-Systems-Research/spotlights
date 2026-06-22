"""Compatibility accessors bridging the old flat `Candidate` shape to the new one.

The new `Candidate` (see `schemas.candidate`) nests its location data under
`locations[].spans[]` and merges the former `deep_research_proposals` /
`agent_proposals` lists into a single `proposals: list[Proposal]` discriminated
by `Proposal.source`. These helpers keep the resulting indexing assumptions in
exactly one place:

- `primary_span` / `primary_file` encode decision **D1** (single-span wrap):
  consumers read the *first span of the first location*. When multi-location
  support lands this is the single site to revisit.
- `make_location` builds the nested shape from flat fields at the discovery
  promotion boundary (and enforces the `CodeSpan` range invariant early).
- `proposals_from` filters the unified list by `source` so the source literals
  don't get copy-pasted across consumers.
- `mint_proposal_ids` mints `prop-<segment>-NNNN` ids from a caller-supplied
  start and the module's id segment (decision **D3**) — the slug makes the ids
  globally unique, so the counter is a plain per-(module, session) sequence
  rather than a disjoint numeric block.
"""

from __future__ import annotations

from spotlights_engine.schemas.candidate import (
    Candidate,
    CodeKind,
    CodeLocation,
    CodeSpan,
)
from spotlights_engine.schemas.proposal import Proposal, ProposalSource

# The counter stays four digits, but it is now a per-(module, session) sequence;
# uniqueness across modules comes from the slug segment, not the number. A
# single module session emitting >9999 proposals is still an error worth
# raising, so the width ceiling remains.
MAX_FOUR_DIGIT_ID = 9999


def primary_span(c: Candidate) -> CodeSpan:
    """The first span of the candidate's first location (decision D1).

    Both `locations` (min_length=1) and `spans` (min_length=1) are guaranteed
    non-empty by the schema, so this never raises on a validated `Candidate`.
    """
    return c.locations[0].spans[0]


def primary_file(c: Candidate) -> str:
    """The file of the candidate's first location (decision D1)."""
    return c.locations[0].file


def make_location(
    file: str,
    line_start: int,
    line_end: int,
    symbol: str,
    kind: CodeKind,
) -> CodeLocation:
    """Wrap flat discovery fields into the nested single-span shape (D1).

    Constructing the `CodeSpan` here applies its `line_end >= line_start`
    validator at the promotion boundary, mirroring the old `Candidate._check_range`.
    """
    return CodeLocation(
        file=file,
        spans=[
            CodeSpan(
                line_start=line_start,
                line_end=line_end,
                symbol=symbol,
                kind=kind,
            )
        ],
    )


def proposals_from(c: Candidate, source: ProposalSource) -> list[Proposal]:
    """Proposals on `c` whose `source` matches, in list order."""
    return [p for p in c.proposals if p.source == source]


def mint_proposal_ids(start: int, new_count: int, *, segment: str) -> list[str]:
    """Return `new_count` sequential `prop-<segment>-NNNN` ids from `start`.

    `start` is the next free number in the module session's proposal counter
    (decision D3), advancing across steps 4 -> 5; `segment` is the module's id
    segment (`module_segment`). `mint_proposal_ids(7, 3, segment="auth_login")`
    yields `["prop-auth_login-0007", "prop-auth_login-0008",
    "prop-auth_login-0009"]`. The four-digit ceiling is now a per-module-session
    limit, not a run-wide cap.
    """
    if start < 1:
        raise ValueError(f"proposal id start must be >= 1, got {start}")
    if new_count < 0:
        raise ValueError(f"new_count must be >= 0, got {new_count}")
    if not segment:
        raise ValueError("proposal id segment must be non-empty")
    end = start + new_count - 1
    if end > MAX_FOUR_DIGIT_ID:
        raise ValueError(
            f"proposal id allocation [{start}, {end}] exceeds the four-digit "
            f"counter ceiling {MAX_FOUR_DIGIT_ID} for module session {segment!r}"
        )
    return [f"prop-{segment}-{n:04d}" for n in range(start, start + new_count)]


__all__ = [
    "MAX_FOUR_DIGIT_ID",
    "make_location",
    "mint_proposal_ids",
    "primary_file",
    "primary_span",
    "proposals_from",
]
