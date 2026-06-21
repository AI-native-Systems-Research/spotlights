"""Tests for the signal-pipeline `SpotlightReport` builder.

Covers the boundary translation that turns
`Signals.anomalies + list[CandidateDraft] + dict[id, Change]` into the
unified `SpotlightReport` schema. End-to-end runner emission is exercised
indirectly by `test_runner.py::test_fresh_run_writes_canonical_layout`
(it asserts `spotlight_report.json` lands in the run dir).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import RunInfo, SpotlightReport
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline.layout import RunDirLayout
from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    CandidateDraft,
    Change,
    Signals,
    TraceSummaryLite,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.spotlight_report import (
    _resolve_module,
    build_spotlight_report,
    emit_spotlight_report,
)


# ---- Fixtures ---------------------------------------------------------------


def _signals(*, with_anomaly: bool = True) -> Signals:
    anomalies = []
    if with_anomaly:
        anomalies.append(
            AnomalyLite(
                anomaly_id="anom-1",
                type="latency",
                description="p99 high",
                magnitude="19x p50",
                confidence=0.6,
                evidence_pointer="trace=t1 span=llm_request",
            )
        )
    return Signals(
        workload=WorkloadProfileLite(
            workload_id="wl-test", description="multi-turn agentic"
        ),
        traces=[TraceSummaryLite(trace_id="t1", summary="summary")],
        anomalies=anomalies,
    )


def _project_tree() -> ProjectTree:
    """Tree where module `inference/attention` sits under `src/inference/attention`,
    and `inference` is its parent module. Exercises deepest-prefix module
    assignment with the slash-separated qualified-name encoding."""
    return ProjectTree(
        repository=Repository(
            name="subject", summary="test subject", source_root="src"
        ),
        modules=[
            Module(
                name="inference",
                path="src/inference",
                description="inference top",
                submodules=[
                    Module(
                        name="attention",
                        path="src/inference/attention",
                        description="attention layer",
                    ),
                ],
            ),
        ],
    )


def _draft(
    *, id_: str = "cand-0001", file: str = "src/inference/attention/kernel.py",
    anomaly_refs: list[str] | None = None,
) -> CandidateDraft:
    return CandidateDraft(
        id=id_,
        file=file,
        line_start=10,
        line_end=20,
        symbol="kernel_fn",
        kind="function",
        description="d",
        current_approach="ca",
        evolve_rationale="er",
        estimated_impact="high",
        estimated_impact_explanation="ee",
        anomaly_refs=anomaly_refs or [],
    )


def _change(candidate_id: str, *, mech: str = "swap A for B") -> Change:
    return Change(
        change_id=f"chg-{candidate_id}",
        candidate_ref=candidate_id,
        change_type="tune",
        mechanism=mech,
        expected_effect="latency p99 down 30%",
        required_changes="kernel.py:10-20",
        evaluation_metric="bench-mix p99",
    )


def _run_info() -> RunInfo:
    return RunInfo(
        pipeline="signal",
        run_id="run-xyz",
        started_at="2026-06-19T00:00:00+00:00",
        finished_at="2026-06-19T00:01:00+00:00",
        cost_usd=0.42,
    )


def _context() -> SpotlightContext:
    return SpotlightContext(
        objective="reduce p99 latency", workload_hints=[], validation_plan=[]
    )


# ---- _resolve_module --------------------------------------------------------


def test_resolve_module_picks_deepest_prefix() -> None:
    """File under `src/inference/attention/...` resolves to the deepest
    matching module, not the parent `inference`."""
    qn = _resolve_module("src/inference/attention/kernel.py", _project_tree())
    assert qn == "inference/attention"


def test_resolve_module_falls_back_to_parent_when_deepest_doesnt_match() -> None:
    """File under `src/inference/foo.py` (not under `attention/`) resolves
    to the parent `inference` module."""
    qn = _resolve_module("src/inference/foo.py", _project_tree())
    assert qn == "inference"


def test_resolve_module_returns_none_when_no_match() -> None:
    """File outside any module's path is left unassigned (None) rather than
    matching a sentinel string."""
    qn = _resolve_module("docs/readme.md", _project_tree())
    assert qn is None


# ---- build_spotlight_report -------------------------------------------------


def test_build_assigns_global_cand_and_prop_ids() -> None:
    """Two drafts + two changes produce cand-signal-0001/cand-signal-0002 and
    prop-signal-0001/prop-signal-0002 in deterministic walk order."""
    drafts = [_draft(id_="cand-0001"), _draft(id_="cand-0002", file="src/inference/foo.py")]
    changes = {"cand-0001": _change("cand-0001"), "cand-0002": _change("cand-0002")}
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes=changes,
        context=_context(),
        run_info=_run_info(),
    )
    assert [c.id for c in report.candidates] == ["cand-signal-0001", "cand-signal-0002"]
    proposal_ids = [p.id for c in report.candidates for p in c.proposals]
    assert proposal_ids == ["prop-signal-0001", "prop-signal-0002"]


def test_build_skips_proposal_when_change_missing() -> None:
    """A draft without a corresponding stage-04 change produces a candidate
    with an empty `proposals` list — partial run support."""
    drafts = [_draft(id_="cand-0001"), _draft(id_="cand-0002")]
    changes = {"cand-0001": _change("cand-0001")}  # cand-0002 omitted
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes=changes,
        context=_context(),
        run_info=_run_info(),
    )
    assert len(report.candidates[0].proposals) == 1
    assert report.candidates[1].proposals == []


def test_build_origin_is_telemetry_anomaly() -> None:
    """Every signal-pipeline candidate carries `origin="telemetry_anomaly"`."""
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert all(c.origin == "telemetry_anomaly" for c in report.candidates)


def test_build_proposal_source_is_telemetry_anomaly_with_ref_ids() -> None:
    """Proposals carry `source="telemetry_anomaly"` and the upstream anomaly
    ids in `anomaly_ref_ids`. `finding_ref_id` and `author` stay None."""
    drafts = [_draft(anomaly_refs=["anom-1"])]
    changes = {"cand-0001": _change("cand-0001")}
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes=changes,
        context=_context(),
        run_info=_run_info(),
    )
    p = report.candidates[0].proposals[0]
    assert p.source == "telemetry_anomaly"
    assert p.anomaly_ref_ids == ["anom-1"]
    assert p.finding_ref_id is None
    assert p.author is None


def test_build_proposal_anomaly_ref_ids_is_none_when_empty() -> None:
    """An empty `anomaly_refs` on the draft serializes as None on the
    proposal — the spec uses None to indicate "no upstream anomaly"."""
    drafts = [_draft(anomaly_refs=[])]
    changes = {"cand-0001": _change("cand-0001")}
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes=changes,
        context=_context(),
        run_info=_run_info(),
    )
    assert report.candidates[0].proposals[0].anomaly_ref_ids is None


def test_build_locations_wraps_flat_fields_into_one_codespan() -> None:
    """Single-location candidates today get one `CodeLocation` containing
    one `CodeSpan` per the spec section 4.12 / section 5 PR 2 step 4."""
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    locs = report.candidates[0].locations
    assert len(locs) == 1
    assert locs[0].file == "src/inference/attention/kernel.py"
    assert len(locs[0].spans) == 1
    span = locs[0].spans[0]
    assert (span.line_start, span.line_end) == (10, 20)
    assert span.symbol == "kernel_fn"
    assert span.kind == "function"


def test_build_module_qualified_name_resolved_against_tree() -> None:
    drafts = [_draft(file="src/inference/attention/kernel.py")]
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert report.candidates[0].module_qualified_name == "inference/attention"


def test_build_module_qualified_name_none_for_unmapped_file() -> None:
    drafts = [_draft(file="docs/x.md")]
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert report.candidates[0].module_qualified_name is None


def test_build_anomalies_translate_into_closed_shape() -> None:
    """`AnomalyLite` (extra=allow) translates to closed `Anomaly` with
    confidence/magnitude/evidence_pointer pulled from the production fields."""
    report = build_spotlight_report(
        signals=_signals(with_anomaly=True),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert len(report.anomalies) == 1
    a = report.anomalies[0]
    assert a.anomaly_id == "anom-1"
    assert a.type == "latency"
    assert a.confidence == 0.6
    assert a.magnitude == "19x p50"
    assert a.evidence_pointer == "trace=t1 span=llm_request"
    assert a.severity is None  # no upstream populator


def test_build_findings_is_empty() -> None:
    """Signal pipeline never produces literature findings; the list is
    always empty in a signal-driven report."""
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert report.findings == []


def test_build_run_pipeline_is_signal() -> None:
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
        context=_context(),
        run_info=_run_info(),
    )
    assert report.run.pipeline == "signal"


def test_build_proposal_drops_change_type_from_unified_shape() -> None:
    """`Change.change_type` is intentionally not on the unified `Proposal`
    (spec section 4.10). Verify it doesn't sneak into a structured field."""
    change = _change("cand-0001", mech="reorder kernels")
    drafts = [_draft()]
    report = build_spotlight_report(
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=drafts,
        changes={"cand-0001": change},
        context=_context(),
        run_info=_run_info(),
    )
    p = report.candidates[0].proposals[0]
    # Mechanism + structured fields carry over.
    assert p.mechanism == "reorder kernels"
    assert p.required_changes == "kernel.py:10-20"
    assert p.expected_effect == "latency p99 down 30%"
    assert p.evaluation_metric == "bench-mix p99"
    # change_type isn't represented as a structured field.
    assert "change_type" not in p.model_dump()


# ---- emit_spotlight_report (disk I/O) ---------------------------------------


def _layout(tmp_path: Path) -> RunDirLayout:
    layout = RunDirLayout(tmp_path / "run")
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


def _write_artifacts(
    layout: RunDirLayout,
    *,
    signals: Signals,
    project_tree: ProjectTree,
    drafts: list[CandidateDraft],
    changes: dict[str, Change],
) -> None:
    layout.stage_artifact("01", shape="single").write_text(
        signals.model_dump_json(indent=2), encoding="utf-8"
    )
    layout.stage_artifact("02", shape="single").write_text(
        project_tree.model_dump_json(indent=2), encoding="utf-8"
    )
    layout.stage_artifact("03", shape="single").write_text(
        json.dumps([d.model_dump(mode="json") for d in drafts], indent=2),
        encoding="utf-8",
    )
    fanout_dir = layout.stage_artifact("04", shape="fanout")
    fanout_dir.mkdir(parents=True, exist_ok=True)
    for cid, change in changes.items():
        (fanout_dir / f"{cid}.json").write_text(
            change.model_dump_json(indent=2), encoding="utf-8"
        )


def test_emit_writes_spotlight_report_json(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    _write_artifacts(
        layout,
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft(anomaly_refs=["anom-1"])],
        changes={"cand-0001": _change("cand-0001")},
    )
    report = emit_spotlight_report(
        layout,
        run_id="run-xyz",
        started_at="2026-06-19T00:00:00+00:00",
        finished_at="2026-06-19T00:01:00+00:00",
        cost_usd=0.42,
    )
    assert report is not None
    target = layout.root / "spotlight_report.json"
    assert target.exists()
    parsed = SpotlightReport.model_validate_json(target.read_text(encoding="utf-8"))
    assert parsed.run.pipeline == "signal"
    assert parsed.run.run_id == "run-xyz"
    assert len(parsed.candidates) == 1
    assert parsed.candidates[0].proposals[0].anomaly_ref_ids == ["anom-1"]


def test_emit_returns_none_when_signals_missing(tmp_path: Path) -> None:
    """Partial run that didn't reach stage 01 → no signals → no report."""
    layout = _layout(tmp_path)
    report = emit_spotlight_report(
        layout, run_id="r", started_at="2026-06-19T00:00:00+00:00"
    )
    assert report is None
    assert not (layout.root / "spotlight_report.json").exists()


def test_emit_works_with_partial_changes(tmp_path: Path) -> None:
    """Stage 03 done, stage 04 partial — report still emitted; candidates
    without a change get an empty proposals list."""
    layout = _layout(tmp_path)
    _write_artifacts(
        layout,
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft(id_="cand-0001"), _draft(id_="cand-0002")],
        changes={"cand-0001": _change("cand-0001")},
    )
    report = emit_spotlight_report(
        layout, run_id="r", started_at="2026-06-19T00:00:00+00:00"
    )
    assert report is not None
    assert len(report.candidates) == 2
    assert len(report.candidates[0].proposals) == 1
    assert report.candidates[1].proposals == []


def test_emit_uses_default_context_when_none_supplied(tmp_path: Path) -> None:
    """Signal pipeline has no caller-supplied objective; the builder
    populates `SpotlightContext.objective` with a fixed general string
    so the closed `min_length=1` constraint stays valid. Future runs
    will supply real objectives like 'Reduce TTFT'."""
    layout = _layout(tmp_path)
    _write_artifacts(
        layout,
        signals=_signals(),
        project_tree=_project_tree(),
        drafts=[_draft()],
        changes={},
    )
    report = emit_spotlight_report(
        layout, run_id="r", started_at="2026-06-19T00:00:00+00:00"
    )
    assert report is not None
    assert report.context.objective == (
        "Address performance issues surfaced by telemetry signals"
    )
    assert "wl-test" in report.context.workload_hints
