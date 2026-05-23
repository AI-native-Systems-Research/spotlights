"""Step-5 resume paths and pre-step-5 manifest compatibility."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    ResumeMismatchError,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from spotlights_engine.spotlights_manager import persistence as P
from tests.unit.spotlights_manager._fakes import (
    make_agent_proposals_result,
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_proposal_from_finding_result,
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


def _wire(
    monkeypatch,
    *,
    tree,
    discover_calls: list[str],
    research_calls: list[str],
    step4_calls: list[str],
    step5_calls: list[str],
) -> None:
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

    def _step4(inp, *, config, runner=None):
        qn = inp.candidates.module_qualified_name
        step4_calls.append(qn)
        return make_proposal_from_finding_result(
            qn,
            n_candidates=len(inp.candidates.candidates),
        )

    def _step5(inp, *, config, claude_runner=None, codex_runner=None):
        qn = inp.candidates.module_qualified_name
        step5_calls.append(qn)
        return make_agent_proposals_result(inp.candidates)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)
    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _step4)
    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _step5)


def _load_checkpoint(mp: P.ModulePaths) -> P.ModuleCheckpoint:
    return P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )


def _write_checkpoint(mp: P.ModulePaths, cp: P.ModuleCheckpoint, **updates) -> None:
    P.write_checkpoint(mp, cp.model_copy(update=updates))


def test_resume_runs_only_step5_from_finding_proposals_created(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    step4_calls: list[str] = []
    step5_calls: list[str] = []
    _wire(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
        step4_calls=step4_calls,
        step5_calls=step5_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    mp.agent_proposals_path.unlink()
    cp = _load_checkpoint(mp)
    _write_checkpoint(
        mp,
        cp,
        status="FINDING_PROPOSALS_CREATED",
        last_step="proposal_from_finding_creator",
        issues=[],
    )

    discover_calls.clear()
    research_calls.clear()
    step4_calls.clear()
    step5_calls.clear()
    result = run_with_telemetry(make_input(repo), config=cfg)

    assert discover_calls == []
    assert research_calls == []
    assert step4_calls == []
    assert step5_calls == ["v1.kv_offload"]
    assert result.module_runs["v1.kv_offload"].status == "SUCCEEDED"
    assert mp.agent_proposals_path.exists()


def test_legacy_terminal_step4_resume_runs_only_step5(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    step4_calls: list[str] = []
    step5_calls: list[str] = []
    _wire(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
        step4_calls=step4_calls,
        step5_calls=step5_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    mp.agent_proposals_path.unlink()
    cp = _load_checkpoint(mp)
    _write_checkpoint(
        mp,
        cp,
        status="SUCCEEDED",
        last_step="proposal_from_finding_creator",
        issues=[],
    )

    discover_calls.clear()
    research_calls.clear()
    step4_calls.clear()
    step5_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)

    assert discover_calls == []
    assert research_calls == []
    assert step4_calls == []
    assert step5_calls == ["v1.kv_offload"]


def test_pre_step5_manifest_migrates_for_default_agent_config(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    step4_calls: list[str] = []
    step5_calls: list[str] = []
    _wire(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
        step4_calls=step4_calls,
        step5_calls=step5_calls,
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    manifest = P.read_manifest(paths)
    assert manifest is not None
    manifest["config_fingerprint"].pop("agent_proposals_hash")
    P.write_manifest(paths, manifest)

    mp = paths.for_module("v1.kv_offload")
    mp.agent_proposals_path.unlink()
    cp = _load_checkpoint(mp)
    _write_checkpoint(
        mp,
        cp,
        status="SUCCEEDED",
        last_step="proposal_from_finding_creator",
        issues=[],
    )

    step5_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)

    migrated = P.read_manifest(paths)
    assert migrated is not None
    assert "agent_proposals_hash" in migrated["config_fingerprint"]
    assert step5_calls == ["v1.kv_offload"]


def test_pre_step5_manifest_rejects_non_default_agent_config(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    _wire(
        monkeypatch,
        tree=tree,
        discover_calls=[],
        research_calls=[],
        step4_calls=[],
        step5_calls=[],
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    manifest = P.read_manifest(paths)
    assert manifest is not None
    manifest["config_fingerprint"].pop("agent_proposals_hash")
    P.write_manifest(paths, manifest)

    non_default = cfg.model_copy(
        update={"agent_proposals": AgentProposalsConfig(max_parallel_candidates=2)}
    )
    with pytest.raises(ResumeMismatchError):
        run_with_telemetry(make_input(repo), config=non_default)
