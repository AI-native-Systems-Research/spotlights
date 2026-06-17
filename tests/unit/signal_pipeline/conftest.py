"""Test fixtures for the signal pipeline.

The runner state-machine tests don't care about stage internals — they
care about resume gating, fan-out manifests, inject validation. As real
stage bodies replace stubs (steps 3+), each one needs a per-test default
that bypasses external calls (claude subprocess, network, sibling repos).

`_default_stubs` is autouse so any test that doesn't explicitly opt out
gets safe fallbacks. Tests that *want* to exercise the real path (e.g.
`test_s02.py::test_extract_is_called_with_correct_args`) override the
specific fixture they need with their own monkeypatch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.schemas.legacy.project import Module, ProjectTree, Repository


def pytest_configure(config):
    """Register the `no_stub_stages` marker so a test can opt out of the
    autouse stub fixture and exercise the real stage helpers."""
    config.addinivalue_line(
        "markers",
        "no_stub_stages: skip the conftest autouse fixture that stubs LLM-backed stages",
    )


@pytest.fixture(autouse=True)
def _default_stage_stubs(request, monkeypatch):
    """Pin LLM-backed stage entry points to deterministic placeholders.

    Centralized here so adding a real stage in steps 3+ doesn't ripple
    through the existing 30 state-machine tests. Each per-stage helper
    function (`_extract_project_tree`, the future `_run_candidate_gen`,
    etc.) is the monkeypatch target, not the stage's `run` itself —
    that way the runner's plumbing (`run` invocation, ctx wiring,
    artifact write, status update) is still exercised end-to-end.

    Tests that need the real helper (e.g. wiring tests that mock one
    layer deeper) mark themselves with `@pytest.mark.no_stub_stages`.
    """
    if request.node.get_closest_marker("no_stub_stages"):
        return

    # ── Stage 01 ─────────────────────────────────────────────────────
    # Only the agent-extraction branch needs stubbing; the pre-cooked
    # JSON branch and synthetic-fallback branch don't fire claude. Tests
    # that pass `--telemetry-from <dir-with-traces.jsonl>` would hit
    # the agent path, so stub it defensively.
    from spotlights_engine.signal_pipeline.schemas import (
        AnomalyLite,
        Signals,
        TraceSummaryLite,
        WorkloadProfileLite,
    )
    from spotlights_engine.signal_pipeline.stages import s01_signal_extraction

    def _fake_extract_signals(target_dir, log_dir, on_event=None, model=None):
        return Signals(
            workload=WorkloadProfileLite(
                workload_id="conftest-stub-workload",
                description="conftest stub for runner state-machine tests",
            ),
            traces=[
                TraceSummaryLite(
                    trace_id="conftest-trace",
                    summary="conftest stub trace",
                )
            ],
            anomalies=[
                AnomalyLite(
                    anomaly_id="conftest-anomaly",
                    type="stub",
                    description="conftest stub anomaly",
                )
            ],
        )

    monkeypatch.setattr(
        s01_signal_extraction, "_extract_signals_via_claude", _fake_extract_signals
    )

    # ── Stage 02 ─────────────────────────────────────────────────────
    from spotlights_engine.signal_pipeline.stages import s02_projecttree

    def _fake_extract(
        subject_root: Path, log_dir: Path, on_event=None, use_cache: bool = True
    ) -> ProjectTree:
        # Tests don't depend on this exact shape — only that it
        # round-trips through `02_projecttree.json` and parses back.
        return ProjectTree(
            repository=Repository(
                name="stub-repo",
                summary="conftest stub for runner state-machine tests",
            ),
            modules=[
                Module(
                    name="stub_module",
                    path="stub_module",
                    description="conftest stub module",
                )
            ],
        )

    monkeypatch.setattr(s02_projecttree, "_extract_project_tree", _fake_extract)

    # ── Stage 03 ─────────────────────────────────────────────────────
    from spotlights_engine.schemas.legacy.candidate import Candidate
    from spotlights_engine.signal_pipeline.stages import s03_candidate_generation

    def _fake_candidates(
        *, signals, project_tree, subject_root, log_dir,
        on_event=None, model=None, max_candidates=None,
    ):
        return [
            Candidate(
                id="cand-0001",
                file="stub_module/example.py",
                line_start=1,
                line_end=10,
                symbol="stub_function_a",
                kind="function",
                description="Phase 1 stub candidate.",
                current_approach="Stubbed.",
                evolve_rationale="Stubbed.",
                estimated_impact="medium",
                estimated_impact_explanation="Stubbed.",
            ),
            Candidate(
                id="cand-0002",
                file="stub_module/example.py",
                line_start=20,
                line_end=30,
                symbol="stub_function_b",
                kind="function",
                description="Phase 1 stub candidate.",
                current_approach="Stubbed.",
                evolve_rationale="Stubbed.",
                estimated_impact="low",
                estimated_impact_explanation="Stubbed.",
            ),
        ]

    monkeypatch.setattr(
        s03_candidate_generation, "_run_candidate_generation", _fake_candidates
    )

    # ── Stage 04 ─────────────────────────────────────────────────────
    from spotlights_engine.signal_pipeline.schemas import Change
    from spotlights_engine.signal_pipeline.stages import s04_change_generation

    def _fake_change(candidate, subject_root, log_dir, on_event=None, model=None):
        return Change(
            change_id=f"chg-{candidate.id}",
            candidate_ref=candidate.id,
            change_type="other",
            mechanism="conftest stub change",
            expected_effect="stub effect",
            required_changes="stub changes",
            evaluation_metric="stub metric",
        )

    monkeypatch.setattr(s04_change_generation, "_generate_change", _fake_change)

    # ── Stage 05 ─────────────────────────────────────────────────────
    from spotlights_engine.signal_pipeline.schemas import ExecutionResult
    from spotlights_engine.signal_pipeline.stages import s05_execution

    def _fake_execute(
        *, change, subject_root, log_dir, backend_id, on_event=None, model=None
    ):
        return ExecutionResult(
            result_id=f"res-{change.candidate_ref}",
            change_ref=change.change_id,
            backend_id=backend_id,
            status="applied",
            file_edits=[],
            rationale="conftest stub execution",
            artifacts={},
        )

    monkeypatch.setattr(s05_execution, "_execute_change", _fake_execute)
