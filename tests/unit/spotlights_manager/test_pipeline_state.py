"""Unit tests for the pipeline-internal candidate state map (decision D2)."""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.spotlights_manager.pipeline_state import state_map_for
from spotlights_engine.utils.schema_compat import make_location


def _cand(n: int) -> Candidate:
    return Candidate(
        id=f"cand-m-{n:04d}",
        origin="code_agent",
        locations=[
            make_location(
                file="src/a.py",
                line_start=1,
                line_end=2,
                symbol=f"s{n}",
                kind="function",
            )
        ],
        description="d",
        current_approach="c",
        evolve_rationale="e",
        estimated_impact="low",
        estimated_impact_explanation="x",
    )


def test_state_map_for_keys_every_candidate_id() -> None:
    cands = Candidates(
        module_qualified_name="m", candidates=[_cand(1), _cand(2)]
    )
    assert state_map_for(cands, "DISCOVERED") == {
        "cand-m-0001": "DISCOVERED",
        "cand-m-0002": "DISCOVERED",
    }


def test_state_map_for_empty_is_empty() -> None:
    cands = Candidates(module_qualified_name="m", candidates=[])
    assert state_map_for(cands, "FINDING_PROPOSALS_CREATED") == {}
