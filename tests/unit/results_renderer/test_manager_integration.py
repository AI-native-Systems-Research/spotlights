"""End-to-end manager integration: `run_with_telemetry` invokes the renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.schemas.pipeline import SpotlightReport

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


@pytest.fixture
def output(tmp_path: Path) -> Path:
    return tmp_path / "output"


def _patch_pipeline(monkeypatch) -> None:
    tree = make_tree()

    def _fake_extract(inp, *, config):
        return make_extractor_result(tree)

    monkeypatch.setattr(orch, "extract_with_telemetry", _fake_extract)
    monkeypatch.setattr(
        orch,
        "discover",
        lambda inp, *, config: make_discovery_result(inp.module_qualified_name),
    )
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None: make_research_output(n_findings=1),
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)


def test_manager_renders_index_md(
    monkeypatch, repo: Path, artifacts: Path, output: Path
) -> None:
    _patch_pipeline(monkeypatch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=output,
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    assert result.renderer_result is not None
    assert result.renderer_result.index_path == output / "index.md"
    assert (output / "index.md").exists()
    assert (output / "modules" / "v1_kv_offload.md").exists()
    assert result.manager_issues == []


def test_run_returns_spotlight_report(
    monkeypatch, repo: Path, artifacts: Path, output: Path
) -> None:
    _patch_pipeline(monkeypatch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=output,
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    report = run(make_input(repo), config=cfg)
    assert isinstance(report, SpotlightReport)
    assert report.run.pipeline == "deep_research"
    assert report.run.run_id
    # The kv_offload module's candidates are flattened into the report; their
    # slug-segmented ids stay globally unique.
    assert report.candidates
    assert all(c.id.startswith("cand-v1_kv_offload") for c in report.candidates)


def test_renderer_failure_recorded_as_manager_issue(
    monkeypatch, repo: Path, artifacts: Path, output: Path
) -> None:
    _patch_pipeline(monkeypatch)

    def _boom(input, *, config=None):
        raise RuntimeError("renderer asploded")

    # Patch the symbol the orchestrator imports lazily.
    import spotlights_engine.results_renderer as rr

    monkeypatch.setattr(rr, "render", _boom)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=output,
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    assert result.renderer_result is None
    assert any(
        iss.step == "results_renderer" for iss in result.manager_issues
    )
    # The pipeline did not fail just because the renderer did.
    assert "v1/kv_offload" in result.module_runs
