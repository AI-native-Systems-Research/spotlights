"""Unit tests for the signal-pipeline → SpotlightReport adapter."""

from __future__ import annotations

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.schemas.spotlight_report import RunInfo
from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    Change,
    Signals,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.spotlight_report_adapter import (
    to_spotlight_report,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


def _project_tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="vllm", summary="vllm subset"),
        modules=[
            Module(
                name="v1",
                path="vllm/v1",
                description="",
                submodules=[
                    Module(name="kv_offload", path="vllm/v1/kv_offload"),
                    Module(name="attention", path="vllm/v1/attention"),
                ],
            ),
            Module(name="utils", path="vllm/utils"),
        ],
    )


def _context() -> SpotlightContext:
    return SpotlightContext(objective="reduce TTFT")


def _signals(*anomalies: AnomalyLite) -> Signals:
    return Signals(
        workload=WorkloadProfileLite(workload_id="wl-1", description=""),
        traces=[],
        anomalies=list(anomalies),
    )


def _candidate(
    idx: int,
    *,
    file: str,
    symbol: str = "f",
    anomaly_refs: list[str] = (),
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
        state="DISCOVERED",
        anomaly_refs=list(anomaly_refs),
    )


def _change(cand_id: str, *, change_type: str = "tune") -> Change:
    return Change(
        change_id=f"chg-{cand_id}",
        candidate_ref=cand_id,
        change_type=change_type,  # type: ignore[arg-type]
        mechanism="m",
        expected_effect="ee",
        required_changes="rc",
        evaluation_metric="em",
    )


def _run_info() -> RunInfo:
    return RunInfo(
        pipeline="signal",
        run_id="r-1",
        started_at="2026-06-14T00:00:00Z",
    )


# ── tests ────────────────────────────────────────────────────────────────────


def test_origin_is_telemetry_anomaly_and_findings_empty() -> None:
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=[_candidate(1, file="vllm/v1/kv_offload/x.py")],
        changes={},
        context=_context(),
        run=_run_info(),
    )
    assert report.findings == []
    assert all(c.origin == "telemetry_anomaly" for c in report.candidates)


def test_anomalies_passed_through_with_ids_preserved() -> None:
    report = to_spotlight_report(
        signals=_signals(
            AnomalyLite(anomaly_id="a-1", type="hot", description="x"),
            AnomalyLite(anomaly_id="a-2", type="slow"),
        ),
        project_tree=_project_tree(),
        candidates=[],
        changes={},
        context=_context(),
        run=_run_info(),
    )
    assert [a.anomaly_id for a in report.anomalies] == ["a-1", "a-2"]


def test_module_assignment_deepest_prefix_match() -> None:
    """File matched against tree.modules[*].path; deepest path wins."""
    candidates = [
        _candidate(1, file="vllm/v1/kv_offload/cpu/manager.py"),
        _candidate(2, file="vllm/v1/attention/backend.py"),
        _candidate(3, file="vllm/utils/helpers.py"),
    ]
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=candidates,
        changes={},
        context=_context(),
        run=_run_info(),
    )
    assigned = [c.module_qualified_name for c in report.candidates]
    # `kv_offload` (depth 3) wins over `v1` (depth 2) for cand-1.
    assert assigned[0] == "v1.kv_offload"
    assert assigned[1] == "v1.attention"
    assert assigned[2] == "utils"


def test_module_assignment_none_for_unmatched_file() -> None:
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=[_candidate(1, file="elsewhere/orphan.py")],
        changes={},
        context=_context(),
        run=_run_info(),
    )
    assert report.candidates[0].module_qualified_name is None


def test_change_populates_structured_fields() -> None:
    cand = _candidate(
        1, file="vllm/v1/kv_offload/x.py", symbol="hot_fn", anomaly_refs=["a-1"]
    )
    report = to_spotlight_report(
        signals=_signals(AnomalyLite(anomaly_id="a-1", type="hot")),
        project_tree=_project_tree(),
        candidates=[cand],
        changes={cand.id: _change(cand.id, change_type="prefetch")},
        context=_context(),
        run=_run_info(),
    )
    assert len(report.candidates[0].proposals) == 1
    p = report.candidates[0].proposals[0]
    assert p.source == "telemetry_anomaly"
    assert p.source_refs == ["a-1"]
    assert p.proposal_type == "prefetch"
    assert p.mechanism == "m"
    assert p.required_changes == "rc"
    assert p.expected_effect == "ee"
    assert p.evaluation_metric == "em"
    assert "Prefetch" in p.title and "hot_fn" in p.title


def test_candidate_without_change_has_empty_proposals() -> None:
    """Stage 04 may be partial; missing change → empty proposal list."""
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=[_candidate(1, file="vllm/utils/x.py")],
        changes={},  # no change for this candidate
        context=_context(),
        run=_run_info(),
    )
    assert report.candidates[0].proposals == []


def test_candidates_renumbered_globally_in_input_order() -> None:
    candidates = [
        _candidate(7, file="vllm/utils/a.py"),
        _candidate(2, file="vllm/v1/attention/b.py"),
    ]
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=candidates,
        changes={
            candidates[0].id: _change(candidates[0].id),
            candidates[1].id: _change(candidates[1].id),
        },
        context=_context(),
        run=_run_info(),
    )
    # Renumbered to cand-0001, cand-0002 in input order (stage 03 stored order).
    assert [c.id for c in report.candidates] == ["cand-0001", "cand-0002"]
    # Proposal ids globally unique across the report.
    prop_ids = [
        p.id for c in report.candidates for p in c.proposals
    ]
    assert prop_ids == ["prop-0001", "prop-0002"]


def test_locations_wrap_each_candidate_into_one_location_one_span() -> None:
    cand = _candidate(1, file="vllm/v1/kv_offload/x.py", symbol="hot")
    report = to_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        candidates=[cand],
        changes={},
        context=_context(),
        run=_run_info(),
    )
    locs = report.candidates[0].locations
    assert len(locs) == 1
    assert locs[0].file == "vllm/v1/kv_offload/x.py"
    assert len(locs[0].spans) == 1
    assert locs[0].spans[0].symbol == "hot"


def test_pure_function_same_input_same_output() -> None:
    cand = _candidate(1, file="vllm/utils/x.py")
    args = dict(
        signals=_signals(AnomalyLite(anomaly_id="a-1", type="t")),
        project_tree=_project_tree(),
        candidates=[cand],
        changes={cand.id: _change(cand.id)},
        context=_context(),
        run=_run_info(),
    )
    a = to_spotlight_report(**args)
    b = to_spotlight_report(**args)
    assert a.model_dump_json() == b.model_dump_json()
