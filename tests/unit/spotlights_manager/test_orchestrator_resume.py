"""Resume rules per plan §7.4 / §7.5."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.spotlights_manager import (
    ManagerSetupError,
    ModuleFilter,
    ResumeMismatchError,
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


def _wire_step_doubles(monkeypatch, *, tree, discover_calls, research_calls) -> None:
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )

    def _discover(inp, *, config):
        discover_calls.append(inp.module_qualified_name)
        return make_discovery_result(inp.module_qualified_name)

    def _research(inp, options=None):
        research_calls.append(inp.module_qualified_name)
        return make_research_output(n_findings=1)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)


def test_resume_skips_completed_module(monkeypatch, repo: Path, artifacts: Path) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == ["v1.kv_offload"]
    assert research_calls == ["v1.kv_offload"]

    # Re-run: extractor + module fully cached.
    discover_calls.clear()
    research_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == []
    assert research_calls == []


def test_resume_reruns_only_step3_when_research_artifact_missing(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    # Delete only the research output and downgrade checkpoint to DEEP_RESEARCHED.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    mp.deep_research_path.unlink()
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(update={"status": "DEEP_RESEARCHED", "issues": []})
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    discover_calls.clear()
    research_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == []
    assert research_calls == ["v1.kv_offload"]


def test_resume_reruns_step2_when_candidates_missing(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    mp.candidates_path.unlink()
    mp.deep_research_path.unlink()
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(update={"status": "DISCOVERED", "issues": []})
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    discover_calls.clear()
    research_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == ["v1.kv_offload"]
    assert research_calls == ["v1.kv_offload"]


def test_redoing_step2_clears_stale_research_output(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )

    discovery_candidate_counts = [1, 0]
    research_calls: list[str] = []

    def _discover(inp, *, config):
        return make_discovery_result(
            inp.module_qualified_name,
            n_candidates=discovery_candidate_counts.pop(0),
        )

    def _research(inp, options=None):
        research_calls.append(inp.module_qualified_name)
        return make_research_output(n_findings=1)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    mp.candidates_path.unlink()
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(update={"status": "DISCOVERED", "issues": []})
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    result = run_with_telemetry(make_input(repo), config=cfg)
    mr = result.module_runs["v1.kv_offload"]
    assert mr.status == "SKIPPED"
    assert mr.findings == []
    assert research_calls == ["v1.kv_offload"]


def test_resume_false_refuses_existing_run_dir(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    cfg2 = cfg.model_copy(update={"resume": False})
    with pytest.raises(ManagerSetupError):
        run_with_telemetry(make_input(repo), config=cfg2)


def test_resume_mismatch_on_changed_input(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _wire_step_doubles(
        monkeypatch, tree=tree, discover_calls=[], research_calls=[]
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    inp2 = make_input(repo).model_copy(update={"max_findings_per_module": 99})
    with pytest.raises(ResumeMismatchError):
        run_with_telemetry(inp2, config=cfg)
