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

from spotlights_engine.schemas.project import Module, ProjectTree, Repository


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

    # ── Stage 02 ─────────────────────────────────────────────────────
    from spotlights_engine.signal_pipeline.stages import s02_projecttree

    def _fake_extract(subject_root: Path, log_dir: Path) -> ProjectTree:
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
