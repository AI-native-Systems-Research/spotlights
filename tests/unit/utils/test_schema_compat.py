"""Unit tests for the shared schema-compat accessors (decisions D1 / D3)."""

from __future__ import annotations

import pytest

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.proposal import Proposal
from spotlights_engine.utils.schema_compat import (
    MAX_FOUR_DIGIT_ID,
    make_location,
    mint_proposal_ids,
    primary_file,
    primary_span,
    proposals_from,
)


def _candidate(proposals: list[Proposal] | None = None) -> Candidate:
    return Candidate(
        id="cand-mod-0001",
        origin="code_agent",
        locations=[
            make_location(
                file="src/a.py",
                line_start=3,
                line_end=9,
                symbol="foo",
                kind="function",
            )
        ],
        description="d",
        current_approach="c",
        evolve_rationale="e",
        estimated_impact="high",
        estimated_impact_explanation="x",
        proposals=list(proposals or []),
    )


def test_primary_accessors_read_first_span_of_first_location() -> None:
    c = _candidate()
    assert primary_file(c) == "src/a.py"
    span = primary_span(c)
    assert (span.line_start, span.line_end, span.symbol, span.kind) == (
        3,
        9,
        "foo",
        "function",
    )


def test_make_location_enforces_codespan_range() -> None:
    with pytest.raises(ValueError):
        make_location(
            file="src/a.py",
            line_start=10,
            line_end=2,
            symbol="foo",
            kind="function",
        )


def test_proposals_from_filters_by_source() -> None:
    research = Proposal(
        id="prop-mod-0001",
        source="research_finding",
        finding_ref_id="find-mod-0001",
        title="t",
        description="d",
        rationale="r",
    )
    agent = Proposal(
        id="prop-mod-0002",
        source="agent_knowledge",
        title="t",
        description="d",
        rationale="r",
    )
    c = _candidate([research, agent])
    assert proposals_from(c, "research_finding") == [research]
    assert proposals_from(c, "agent_knowledge") == [agent]
    assert proposals_from(c, "telemetry_anomaly") == []


def test_mint_proposal_ids_sequential_from_start() -> None:
    assert mint_proposal_ids(7, 3, segment="mod") == [
        "prop-mod-0007",
        "prop-mod-0008",
        "prop-mod-0009",
    ]
    assert mint_proposal_ids(1, 0, segment="mod") == []


def test_mint_proposal_ids_embeds_segment() -> None:
    # The slug segment (which may contain '-') makes ids globally unique; the
    # counter is a plain per-module-session sequence.
    assert mint_proposal_ids(1, 1, segment="a-b_c.s2") == ["prop-a-b_c.s2-0001"]


def test_mint_proposal_ids_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        mint_proposal_ids(0, 1, segment="mod")
    with pytest.raises(ValueError):
        mint_proposal_ids(MAX_FOUR_DIGIT_ID, 2, segment="mod")
    with pytest.raises(ValueError):
        mint_proposal_ids(1, 1, segment="")
