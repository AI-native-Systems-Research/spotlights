"""Pipeline-internal candidate state (decision D2).

The schema `Candidate` no longer carries a `state` field. To preserve the
step-4 / step-5 guard checks (step 4 expects `DISCOVERED`, step 5 expects
`FINDING_PROPOSALS_CREATED`) without weakening them, the manager keeps a
**parallel** per-module state map, keyed by candidate id. It lives only inside
the manager/step plumbing and is never persisted or part of any exported schema.

On resume the map is reconstructed from which manager sidecar is present:
discovery candidates → `DISCOVERED`; proposal-from-finding output →
`FINDING_PROPOSALS_CREATED`; agent-proposals output → `AGENT_PROPOSALS_CREATED`.
"""

from __future__ import annotations

from typing import Literal

from spotlights_engine.schemas.candidate import Candidates

CandidatePipelineState = Literal[
    "DISCOVERED",
    "FINDING_PROPOSALS_CREATED",
    "AGENT_PROPOSALS_CREATED",
]

CandidateStateMap = dict[str, CandidatePipelineState]


def state_map_for(
    candidates: Candidates, state: CandidatePipelineState
) -> CandidateStateMap:
    """Build a per-module state map assigning `state` to every candidate id.

    Keyed by the candidate ids in the supplied (manager-promoted) `Candidates`
    wrapper. After discovery promotion those ids are already global; the map
    stays per-module to avoid accidental cross-module coupling.
    """
    return {c.id: state for c in candidates.candidates}


__all__ = [
    "CandidatePipelineState",
    "CandidateStateMap",
    "state_map_for",
]
