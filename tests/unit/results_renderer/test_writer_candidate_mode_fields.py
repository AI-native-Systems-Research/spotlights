"""The four candidate-mode proposal fields render only when present.

Module-mode proposals leave them `None`, so their Markdown must stay
byte-identical to the pre-change output.
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.results_renderer.aggregator import ModulePageView
from spotlights_engine.results_renderer.api import RendererConfig
from spotlights_engine.results_renderer.writer import _render_candidate_page
from spotlights_engine.schemas.proposal import Proposal
from tests.unit.results_renderer._fixtures import (
    make_candidate,
    make_finding,
    make_tree,
)

_HEADINGS = (
    "**Mechanism.**",
    "**Required changes.**",
    "**Expected effect.**",
    "**Evaluation metric.**",
)


def _proposal(**overrides: Any) -> Proposal:
    kwargs: dict[str, Any] = dict(
        id="prop-mod-0001",
        source="research_finding",
        finding_ref_id="find-mod-0001",
        author="proposal_from_finding_creator",
        title="Parallelize hot_1",
        description="Use a thread pool.",
        rationale="Hot path.",
    )
    kwargs.update(overrides)
    return Proposal(**kwargs)


def _page(proposal: Proposal) -> str:
    cand = make_candidate(1, deep_proposals=[proposal])
    finding = make_finding(1)
    view = ModulePageView(
        qualified_name="v1/kv_offload",
        module=make_tree().resolve("v1/kv_offload"),
        status="SUCCEEDED",
        candidates_sorted=[cand],
        findings=[finding],
    )
    return _render_candidate_page(view, cand, RendererConfig(), {finding.finding_id: finding})


def test_module_mode_proposal_renders_none_of_the_new_headings() -> None:
    page = _page(_proposal())

    for heading in _HEADINGS:
        assert heading not in page
    # The historical sections are still there.
    assert "**Detailed description.**" in page
    assert "**Proposal rationale.**" in page


def test_candidate_mode_proposal_renders_all_four_fields() -> None:
    page = _page(
        _proposal(
            author="proposal_from_candidate_finding_creator",
            mechanism="Pool amortizes spawn cost.",
            required_changes="Replace the loop in core.py hot_1.",
            expected_effect="2x on the hot path.",
            evaluation_metric="p99 latency.",
        )
    )

    for heading in _HEADINGS:
        assert heading in page
    assert "Pool amortizes spawn cost." in page
    assert "Replace the loop in core.py hot_1." in page
    assert "2x on the hot path." in page
    assert "p99 latency." in page
    # Order is stable and follows the proposal's own field order.
    positions = [page.index(h) for h in _HEADINGS]
    assert positions == sorted(positions)
    assert page.index("**Proposal rationale.**") < positions[0]


def test_partially_filled_proposal_renders_only_the_present_fields() -> None:
    page = _page(_proposal(mechanism="Pool amortizes spawn cost.", expected_effect="2x."))

    assert "**Mechanism.**" in page
    assert "**Expected effect.**" in page
    assert "**Required changes.**" not in page
    assert "**Evaluation metric.**" not in page
