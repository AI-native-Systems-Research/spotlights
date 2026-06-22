"""End-to-end tests for `unified_runner.run_unified` with mocked sub-pipelines.

Covers:
- mode=both runs extraction once, runs both sub-pipelines concurrently,
  produces a merged `SpotlightReport` with `pipeline="unified"`.
- mode=signal skips DR entirely; mode=dr skips signal entirely.
- Signal is forced to `--to-stage 04` (the unified runner never opts into s05).
- DR's per-pipeline report is persisted to its sub-run-dir for symmetry.
- Resume / fingerprint mismatch policy at the unified layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.modules_extractor.extractor import ExtractorResult
from spotlights_engine.schemas.candidate import Candidate, CodeLocation, CodeSpan
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import (
    RunInfo,
    SpotlightReport,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline.layout import RunDirLayout
from spotlights_engine.unified_runner import (
    UnifiedConfig,
    UnifiedInput,
    UnifiedResumeMismatchError,
    run_unified,
)
from spotlights_engine.unified_runner import runner as unified_runner
from spotlights_engine.unified_runner import shared_extraction as se


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(
            name="demo", summary="x", source_root="", external_dependencies=[]
        ),
        modules=[Module(name="core", path="src/core")],
    )


def _invocation() -> ExtractionInvocation:
    return ExtractionInvocation(
        session_id="sess-1",
        duration_s=1.0,
        cost_usd=0.0,
        input_tokens=10,
        output_tokens=5,
    )


def _candidate(*, id: str, origin: str = "code_agent") -> Candidate:
    return Candidate(
        id=id,
        module_qualified_name="core",
        origin=origin,  # type: ignore[arg-type]
        locations=[
            CodeLocation(
                file="src/core/x.py",
                spans=[CodeSpan(line_start=1, line_end=10, symbol="f", kind="function")],
            )
        ],
        description="d",
        current_approach="ca",
        evolve_rationale="er",
        estimated_impact="high",
        estimated_impact_explanation="x",
        proposals=[],
    )


def _report(*, pipeline: str, candidates, ctx: SpotlightContext, cost: float) -> SpotlightReport:
    return SpotlightReport(
        project_tree=_tree(),
        context=ctx,
        candidates=candidates,
        run=RunInfo(
            pipeline=pipeline,  # type: ignore[arg-type]
            run_id=f"run-{pipeline}",
            started_at="2026-06-22T10:00:00+00:00",
            finished_at="2026-06-22T10:30:00+00:00",
            cost_usd=cost,
        ),
        issues=[],
    )


@pytest.fixture
def patched_extractor(monkeypatch):
    """Stub `extract_with_telemetry` so tests don't fork a real claude subprocess."""
    calls: list[Path] = []

    def _fake_extract(input, *, config=None, on_event=None):
        calls.append(input.repo_path)
        return ExtractorResult(project_tree=_tree(), invocation=_invocation())

    monkeypatch.setattr(se, "extract_with_telemetry", _fake_extract)
    return calls


@pytest.fixture
def patched_signal(monkeypatch):
    """Stub the signal pipeline to write the expected `spotlight_report.json`
    (matching what `emit_spotlight_report` would have produced)."""
    captured: dict = {}

    def _fake_run_pipeline(input, *, run_dir, stages=None, resume=True, **_):
        captured["input"] = input
        captured["stages"] = stages
        captured["resume"] = resume
        layout = RunDirLayout(run_dir.resolve())
        # Imitate emit_spotlight_report: drop a report at the canonical path.
        report = _report(
            pipeline="signal",
            candidates=[_candidate(id="cand-signal-0001", origin="telemetry_anomaly")],
            ctx=input.context if input.context is not None else SpotlightContext(objective="o"),
            cost=1.0,
        )
        (layout.root / "spotlight_report.json").write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        from spotlights_engine.signal_pipeline.schemas import SignalPipelineResult

        return SignalPipelineResult(
            run_dir=layout.root,
            output_folder=layout.root / "report",
            completed_stages=["01", "02", "03", "04"],
            skipped_stages=[],
            issues=[],
        )

    monkeypatch.setattr(unified_runner, "run_signal_pipeline", _fake_run_pipeline)
    return captured


@pytest.fixture
def patched_dr(monkeypatch):
    """Stub DR's `_run_async` so tests don't drive the real orchestrator."""
    captured: dict = {}

    async def _fake_run_async(input: SpotlightsManagerInput, *, config):
        captured["input"] = input
        captured["config"] = config
        from spotlights_engine.spotlights_manager.api import SpotlightsManagerResult

        report = _report(
            pipeline="deep_research",
            candidates=[_candidate(id="cand-core-0001")],
            ctx=input.context,
            cost=2.0,
        )
        return SpotlightsManagerResult(
            report=report,
            module_runs={},
            extractor_invocation=_invocation(),
            per_module_telemetry={},
            manager_issues=[],
            renderer_result=None,
        )

    # Patch on the orchestrator module so the runner's late import picks it up.
    import spotlights_engine.spotlights_manager.orchestrator as orchestrator

    monkeypatch.setattr(orchestrator, "_run_async", _fake_run_async)
    return captured


def _make_input(tmp_path, mode="both") -> UnifiedInput:
    return UnifiedInput(
        repo_path=tmp_path / "repo",
        context=SpotlightContext(objective="reduce p99", workload_hints=["batch=8"]),
        mode=mode,  # type: ignore[arg-type]
    )


def _make_config(tmp_path) -> UnifiedConfig:
    return UnifiedConfig(
        artifacts_dir=tmp_path / "artifacts",
        output_folder=tmp_path / "output",
    )


# Tests --------------------------------------------------------------------


def test_run_both_extracts_once_and_emits_unified_report(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)

    result = run_unified(inp, config=cfg)

    assert len(patched_extractor) == 1, "extractor should run exactly once"
    assert result.report.run.pipeline == "unified"
    assert {c.id for c in result.report.candidates} == {
        "cand-signal-0001",
        "cand-core-0001",
    }
    assert result.report.run.cost_usd == pytest.approx(3.0)


def test_run_both_writes_top_level_report_and_summary(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)

    result = run_unified(inp, config=cfg)

    top = result.run_dir / "spotlight_report.json"
    assert top.exists()
    SpotlightReport.model_validate_json(top.read_text(encoding="utf-8"))

    summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "both"


def test_run_both_signal_locked_to_stage_04(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    run_unified(inp, config=cfg)

    sel = patched_signal["stages"]
    assert sel.from_stage == "01"
    assert sel.to_stage == "04", "stage 05 must be locked off in mode=both"
    assert patched_signal["resume"] is True


def test_run_both_threads_context_into_signal(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    run_unified(inp, config=cfg)

    sig_input = patched_signal["input"]
    assert sig_input.context is not None
    assert sig_input.context.objective == "reduce p99"
    assert sig_input.context.workload_hints == ["batch=8"]


def test_run_both_persists_dr_report_for_symmetry(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    result = run_unified(inp, config=cfg)

    dr_report = result.run_dir / "deep_research" / "spotlight_report.json"
    assert dr_report.exists()
    SpotlightReport.model_validate_json(dr_report.read_text(encoding="utf-8"))


def test_run_signal_only_skips_dr(
    tmp_path, patched_extractor, patched_signal, monkeypatch
):
    # Patch DR's _run_async to raise so we know it's NOT called.
    import spotlights_engine.spotlights_manager.orchestrator as orchestrator

    async def _should_not_be_called(*a, **kw):
        raise AssertionError("DR _run_async called in mode=signal")

    monkeypatch.setattr(orchestrator, "_run_async", _should_not_be_called)

    inp = _make_input(tmp_path, mode="signal")
    cfg = _make_config(tmp_path)
    result = run_unified(inp, config=cfg)

    assert result.report.run.pipeline == "unified"
    assert {c.id for c in result.report.candidates} == {"cand-signal-0001"}
    assert result.summary.dr_artifacts_dir is None


def test_run_dr_only_skips_signal(tmp_path, patched_extractor, patched_dr, monkeypatch):
    def _should_not_be_called(*a, **kw):
        raise AssertionError("signal pipeline called in mode=dr")

    monkeypatch.setattr(unified_runner, "run_signal_pipeline", _should_not_be_called)

    inp = _make_input(tmp_path, mode="dr")
    cfg = _make_config(tmp_path)
    result = run_unified(inp, config=cfg)

    assert result.report.run.pipeline == "unified"
    assert {c.id for c in result.report.candidates} == {"cand-core-0001"}
    assert result.summary.signal_run_dir is None


def test_resume_mismatch_raises_when_resume_true(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    result = run_unified(inp, config=cfg)

    # Tamper with the manifest so the next run sees a fingerprint mismatch.
    manifest_path = result.run_dir / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fingerprint"]["mode"] = "signal"
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    # Same UnifiedInput → same run_id → same run_dir on second invocation.
    with pytest.raises(UnifiedResumeMismatchError):
        run_unified(inp, config=cfg)


def test_no_resume_clears_run_dir_on_mismatch(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    result1 = run_unified(inp, config=cfg)

    # Tamper with the manifest then re-run with resume=False.
    payload = json.loads((result1.run_dir / "manifest.json").read_text(encoding="utf-8"))
    payload["fingerprint"]["mode"] = "signal"
    (result1.run_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    cfg2 = _make_config(tmp_path).model_copy(update={"resume": False})
    result2 = run_unified(inp, config=cfg2)
    assert result2.run_dir == result1.run_dir
    assert result2.report.run.pipeline == "unified"


def test_clean_resume_reuses_extractor(
    tmp_path, patched_extractor, patched_signal, patched_dr
):
    """A second invocation with the same fingerprint should not re-extract."""
    inp = _make_input(tmp_path, mode="both")
    cfg = _make_config(tmp_path)
    run_unified(inp, config=cfg)
    assert len(patched_extractor) == 1

    run_unified(inp, config=cfg)  # second run with matching fingerprint
    assert len(patched_extractor) == 1, "extractor should be cached on resume"
