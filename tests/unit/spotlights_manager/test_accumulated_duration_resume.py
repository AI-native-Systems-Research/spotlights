"""Resume regression for `timing.accumulated_duration_s`.

The public run manifest's `accumulated_duration_s` is reconstructed by summing
the durably-persisted per-step durations, so a crash + resume (all steps cached,
nothing re-executed) must report the *same* number as the original run — proving
it comes from disk, not the last-leg process wall clock. See
design/accumulated_duration.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.spotlights_manager import persistence as P
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_research_output,
    make_tree,
    patch_agent_proposals,
    patch_proposal_from_finding,
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    return r


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    a = tmp_path / "artifacts"
    a.mkdir()
    return a


def _read_accumulated(paths: P.ManagerPaths) -> float:
    payload = json.loads(paths.run_manifest_path.read_text(encoding="utf-8"))
    return payload["timing"]["accumulated_duration_s"]


def test_accumulated_duration_is_resume_stable(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()

    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None, **_kw: make_research_output(n_findings=1),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )

    run_with_telemetry(make_input(repo), config=cfg)
    paths = P.ManagerPaths(artifacts)
    first = _read_accumulated(paths)

    # The fake discovery sidecar duration is nonzero, so the accumulated total
    # must be positive; otherwise the assertion below would pass trivially on
    # two zeros.
    assert first > 0.0

    # Re-run with resume: every step is cached, so nothing re-executes. Make
    # re-execution loud in case caching regresses.
    def _boom(*_a, **_k):  # pragma: no cover - asserted not called
        raise AssertionError("no step should re-run on a fully cached resume")

    monkeypatch.setattr(orch, "extract_with_telemetry", _boom)
    monkeypatch.setattr(orch, "discover", _boom)
    monkeypatch.setattr(orch, "research_module", _boom)
    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _boom)
    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _boom)

    run_with_telemetry(make_input(repo), config=cfg)
    second = _read_accumulated(paths)

    assert second == first
