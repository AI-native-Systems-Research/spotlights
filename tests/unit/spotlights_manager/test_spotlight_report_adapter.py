"""Unit tests for the deep-research → SpotlightReport adapter."""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleRun, SpotlightsResult
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.schemas.proposals import AgentProposal, DeepResearchProposal
from spotlights_engine.schemas.spotlight_report import RunInfo
from spotlights_engine.spotlights_manager.spotlight_report_adapter import (
    to_spotlight_report,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


def _project_tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="vllm", summary="vllm subset"),
        modules=[
            Module(name="alpha", path="vllm/alpha", description=""),
            Module(name="bravo", path="vllm/bravo", description=""),
        ],
    )


def _context() -> SpotlightContext:
    return SpotlightContext(objective="reduce TTFT")


def _run_info() -> RunInfo:
    return RunInfo(
        pipeline="deep_research",
        run_id="r-1",
        started_at="2026-06-14T00:00:00Z",
        finished_at="2026-06-14T00:05:00Z",
        cost_usd=1.23,
        duration_s=300.0,
        parameters={
            "max_findings_per_module": 30,
            "continue_on_module_failure": True,
        },
    )


def _finding(idx: int, *, title: str = "t") -> Finding:
    return Finding(
        finding_id=f"find-{idx:04d}",
        title=f"{title}-{idx}",
        url=f"https://example.com/{idx}",
        source_type="paper",
        technique_summary="ts",
    )


def _candidate(
    idx: int,
    *,
    file: str,
    symbol: str,
    drps: list[DeepResearchProposal] = (),
    aps: list[AgentProposal] = (),
) -> Candidate:
    return Candidate(
        id=f"cand-{idx:04d}",
        file=file,
        line_start=10,
        line_end=20,
        symbol=symbol,
        kind="function",
        description="d",
        current_approach="ca",
        evolve_rationale="er",
        estimated_impact="medium",
        estimated_impact_explanation="eie",
        state="AGENT_PROPOSALS_CREATED",
        deep_research_proposals=list(drps),
        agent_proposals=list(aps),
    )


def _drp(finding_id: str, *, title: str = "drp") -> DeepResearchProposal:
    return DeepResearchProposal(
        title=title,
        detailed_description="dd",
        finding_id=finding_id,
        proposal_rationale="pr",
        created_by="proposal_from_finding_creator",
    )


def _ap(agent: str, *, title: str = "ap") -> AgentProposal:
    return AgentProposal(
        title=title,
        detailed_description="dd",
        agent_name=agent,
        novelty_rationale="nr",
    )


# ── tests ────────────────────────────────────────────────────────────────────


def test_module_runs_walked_alphabetically() -> None:
    """Walk order is by sorted module_qualified_name, not insertion order."""
    # Insertion order is (bravo, alpha). Adapter must sort.
    module_runs: dict[str, ModuleRun] = {
        "vllm.bravo": ModuleRun(
            module_qualified_name="vllm.bravo",
            status="SUCCEEDED",
            candidates=Candidates(
                module_qualified_name="vllm.bravo",
                candidates=[_candidate(1, file="vllm/bravo/x.py", symbol="b_one")],
            ),
            findings=[_finding(1, title="b")],
        ),
        "vllm.alpha": ModuleRun(
            module_qualified_name="vllm.alpha",
            status="SUCCEEDED",
            candidates=Candidates(
                module_qualified_name="vllm.alpha",
                candidates=[_candidate(1, file="vllm/alpha/y.py", symbol="a_one")],
            ),
            findings=[_finding(1, title="a")],
        ),
    }
    result = SpotlightsResult(
        project_tree=_project_tree(),
        context=_context(),
        module_runs=module_runs,
    )
    report = to_spotlight_report(result, run=_run_info())

    # Findings: alpha first (find-0001), bravo second (find-0002).
    assert [f.finding_id for f in report.findings] == ["find-0001", "find-0002"]
    assert [f.module_qualified_name for f in report.findings] == [
        "vllm.alpha",
        "vllm.bravo",
    ]
    # Candidates: alpha first (cand-0001), bravo second (cand-0002).
    assert [c.id for c in report.candidates] == ["cand-0001", "cand-0002"]
    assert [c.module_qualified_name for c in report.candidates] == [
        "vllm.alpha",
        "vllm.bravo",
    ]


def test_origin_is_code_agent() -> None:
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[_candidate(1, file="a.py", symbol="f")],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    assert all(c.origin == "code_agent" for c in report.candidates)
    assert report.anomalies == []


def test_findings_renumbered_globally_and_source_refs_rewritten() -> None:
    """Per-module find-0001 ids collide; adapter must renumber globally and
    rewrite each DR proposal's source_refs to the new id."""
    # Both modules have a `find-0001`. Each module has a candidate with a
    # DR proposal that references its module's local find-0001.
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            findings=[_finding(1, title="alpha-finding")],
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[
                    _candidate(
                        1,
                        file="alpha/a.py",
                        symbol="a",
                        drps=[_drp("find-0001", title="drp-alpha")],
                    )
                ],
            ),
        ),
        "bravo": ModuleRun(
            module_qualified_name="bravo",
            status="SUCCEEDED",
            findings=[_finding(1, title="bravo-finding")],
            candidates=Candidates(
                module_qualified_name="bravo",
                candidates=[
                    _candidate(
                        1,
                        file="bravo/b.py",
                        symbol="b",
                        drps=[_drp("find-0001", title="drp-bravo")],
                    )
                ],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    assert [f.finding_id for f in report.findings] == ["find-0001", "find-0002"]
    assert report.findings[0].title == "alpha-finding-1"
    assert report.findings[1].title == "bravo-finding-1"

    # alpha's DR proposal points at the global find-0001;
    # bravo's points at the global find-0002.
    alpha_props = report.candidates[0].proposals
    bravo_props = report.candidates[1].proposals
    assert len(alpha_props) == 1 and alpha_props[0].source_refs == ["find-0001"]
    assert len(bravo_props) == 1 and bravo_props[0].source_refs == ["find-0002"]


def test_agent_proposals_carry_author() -> None:
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[
                    _candidate(
                        1,
                        file="alpha/a.py",
                        symbol="a",
                        aps=[_ap("claude"), _ap("codex")],
                    )
                ],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    props = report.candidates[0].proposals
    assert [p.source for p in props] == ["agent_knowledge", "agent_knowledge"]
    assert [p.author for p in props] == ["claude", "codex"]
    assert all(p.source_refs == [] for p in props)


def test_proposal_type_pass_through_is_none_for_dr_path() -> None:
    """DR adapter leaves the structured fields (proposal_type, mechanism, …)
    None — those are set only by the signal adapter from `Change`."""
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            findings=[_finding(1)],
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[
                    _candidate(
                        1,
                        file="alpha/a.py",
                        symbol="a",
                        drps=[_drp("find-0001")],
                        aps=[_ap("claude")],
                    )
                ],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    for prop in report.candidates[0].proposals:
        assert prop.proposal_type is None
        assert prop.mechanism is None
        assert prop.required_changes is None
        assert prop.expected_effect is None
        assert prop.evaluation_metric is None


def test_locations_wrap_single_file_single_span() -> None:
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[_candidate(1, file="alpha/a.py", symbol="hot")],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    locs = report.candidates[0].locations
    assert len(locs) == 1
    assert locs[0].file == "alpha/a.py"
    assert len(locs[0].spans) == 1
    assert locs[0].spans[0].symbol == "hot"


def test_proposal_ordering_dr_then_agent_within_a_candidate() -> None:
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            findings=[_finding(1), _finding(2)],
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[
                    _candidate(
                        1,
                        file="alpha/a.py",
                        symbol="a",
                        drps=[_drp("find-0001", title="d1"), _drp("find-0002", title="d2")],
                        aps=[_ap("claude", title="agent-1")],
                    )
                ],
            ),
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    sources = [p.source for p in report.candidates[0].proposals]
    assert sources == ["research_finding", "research_finding", "agent_knowledge"]
    assert [p.id for p in report.candidates[0].proposals] == [
        "prop-0001",
        "prop-0002",
        "prop-0003",
    ]


def test_pure_function_same_input_same_output() -> None:
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="SUCCEEDED",
            findings=[_finding(1)],
            candidates=Candidates(
                module_qualified_name="alpha",
                candidates=[
                    _candidate(
                        1,
                        file="alpha/a.py",
                        symbol="a",
                        drps=[_drp("find-0001")],
                        aps=[_ap("claude")],
                    )
                ],
            ),
        ),
    }
    result = SpotlightsResult(
        project_tree=_project_tree(),
        context=_context(),
        module_runs=module_runs,
    )
    report_a = to_spotlight_report(result, run=_run_info())
    report_b = to_spotlight_report(result, run=_run_info())
    assert report_a.model_dump_json() == report_b.model_dump_json()


def test_candidates_none_module_skipped_but_findings_emitted() -> None:
    """A module that failed before discovery has `candidates=None` but may
    still have findings on its run record. Adapter must emit the findings
    and skip candidates without crashing."""
    module_runs = {
        "alpha": ModuleRun(
            module_qualified_name="alpha",
            status="FAILED",
            candidates=None,
            findings=[_finding(1)],
        ),
    }
    report = to_spotlight_report(
        SpotlightsResult(
            project_tree=_project_tree(),
            context=_context(),
            module_runs=module_runs,
        ),
        run=_run_info(),
    )
    assert len(report.findings) == 1
    assert report.candidates == []
