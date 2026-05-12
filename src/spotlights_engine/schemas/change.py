"""The Change schema.

Stub. Field definitions are to be drawn verbatim from `discovery_engine_proposal.md` §4.1.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Change:
    """A realized change applied to the system, as defined in the proposal §4.1.

    TODO: populate fields from `discovery_engine_proposal.md` §4.1.
    """

    id: str
