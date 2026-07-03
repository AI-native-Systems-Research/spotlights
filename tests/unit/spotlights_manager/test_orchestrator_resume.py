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


def _wire_step_doubles(monkeypatch, *, tree, discover_calls, research_calls) -> None:
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )

    def _discover(inp, *, config):
        discover_calls.append(inp.module_qualified_name)
        return make_discovery_result(inp.module_qualified_name)

    def _research(inp, options=None, **_kw):
        research_calls.append(inp.module_qualified_name)
        return make_research_output(n_findings=1)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)


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
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == ["v1/kv_offload"]
    assert research_calls == ["v1/kv_offload"]

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
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    # Delete only the research output and downgrade checkpoint to DEEP_RESEARCHED.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
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
    assert research_calls == ["v1/kv_offload"]


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
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
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
    assert discover_calls == ["v1/kv_offload"]
    assert research_calls == ["v1/kv_offload"]


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

    def _research(inp, options=None, **_kw):
        research_calls.append(inp.module_qualified_name)
        return make_research_output(n_findings=1)

    monkeypatch.setattr(orch, "discover", _discover)
    monkeypatch.setattr(orch, "research_module", _research)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
    assert mp.agent_proposals_path.exists()
    mp.candidates_path.unlink()
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(update={"status": "DISCOVERED", "issues": []})
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    result = run_with_telemetry(make_input(repo), config=cfg)
    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SKIPPED"
    assert mr.findings == []
    assert not mp.agent_proposals_path.exists()
    assert research_calls == ["v1/kv_offload"]


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
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
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
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    inp2 = make_input(repo).model_copy(update={"max_findings_per_module": 99})
    with pytest.raises(ResumeMismatchError):
        run_with_telemetry(inp2, config=cfg)


def test_resume_rejects_pre_change_schema_version(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """A run dir written under the old (pre-source-root) qualified-name layout
    carries `schema_version: 1` and must be rejected cleanly on resume rather
    than silently reloaded with mismatched `module_runs` keys."""
    tree = make_tree()
    _wire_step_doubles(
        monkeypatch, tree=tree, discover_calls=[], research_calls=[]
    )

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    # Simulate a pre-change manifest by downgrading the on-disk schema_version.
    paths = P.ManagerPaths(artifacts)
    manifest = P.read_manifest(paths)
    assert manifest is not None
    manifest["schema_version"] = 1
    P.write_manifest(paths, manifest)

    with pytest.raises(ResumeMismatchError, match="schema_version"):
        run_with_telemetry(make_input(repo), config=cfg)


def test_resume_rejects_manifest_predating_arxiv_hash(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """A run dir whose config fingerprint predates `arxiv_search_hash` must fail
    resume with an explicit ResumeMismatchError — no silent migration (§1.2)."""
    tree = make_tree()
    _wire_step_doubles(monkeypatch, tree=tree, discover_calls=[], research_calls=[])

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    # Simulate a pre-arXiv manifest by dropping the new fingerprint key.
    paths = P.ManagerPaths(artifacts)
    manifest = P.read_manifest(paths)
    assert manifest is not None
    fp = dict(manifest["config_fingerprint"])
    fp.pop("arxiv_search_hash", None)
    manifest["config_fingerprint"] = fp
    P.write_manifest(paths, manifest)

    with pytest.raises(ResumeMismatchError, match="config fingerprint"):
        run_with_telemetry(make_input(repo), config=cfg)


def test_resume_reruns_only_step4_when_proposal_artifact_missing(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    pf_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    # Track step-4 calls in addition to the default no-op patch.
    from tests.unit.spotlights_manager._fakes import (
        make_proposal_from_finding_result,
    )

    def _capture(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        pf_calls.append(inp.candidates.module_qualified_name)
        return make_proposal_from_finding_result(
            inp.candidates.module_qualified_name,
            n_candidates=len(inp.candidates.candidates),
        )

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _capture)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)
    assert pf_calls == ["v1/kv_offload"]

    # Delete only the step-4 sidecar and downgrade checkpoint to DEEP_RESEARCHED.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
    mp.proposal_from_finding_path.unlink()
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(update={"status": "DEEP_RESEARCHED", "issues": []})
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    discover_calls.clear()
    research_calls.clear()
    pf_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == []
    assert research_calls == []
    assert pf_calls == ["v1/kv_offload"]


def test_resume_skips_step4_when_finding_proposals_created_intact(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    pf_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    from tests.unit.spotlights_manager._fakes import (
        make_proposal_from_finding_result,
    )

    def _capture(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        pf_calls.append(inp.candidates.module_qualified_name)
        return make_proposal_from_finding_result(
            inp.candidates.module_qualified_name,
            n_candidates=len(inp.candidates.candidates),
        )

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _capture)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)
    assert pf_calls == ["v1/kv_offload"]

    # Re-run: everything cached, no further work.
    discover_calls.clear()
    research_calls.clear()
    pf_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == []
    assert research_calls == []
    assert pf_calls == []


def test_resume_redoes_step4_only_when_failed_step_is_step4(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    tree = make_tree()
    discover_calls: list[str] = []
    research_calls: list[str] = []
    pf_calls: list[str] = []
    _wire_step_doubles(
        monkeypatch,
        tree=tree,
        discover_calls=discover_calls,
        research_calls=research_calls,
    )

    from tests.unit.spotlights_manager._fakes import (
        make_proposal_from_finding_result,
    )

    def _capture(
        inp,
        *,
        config,
        runner=None,
        candidate_states=None,
        proposal_id_start=1,
        segment=None,
    ):
        pf_calls.append(inp.candidates.module_qualified_name)
        return make_proposal_from_finding_result(
            inp.candidates.module_qualified_name,
            n_candidates=len(inp.candidates.candidates),
        )

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _capture)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    # Edit the checkpoint to FAILED+retryable on step 4. Sidecar still on disk.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
    cp = P.ModuleCheckpoint.model_validate_json(
        mp.status_path.read_text(encoding="utf-8")
    )
    cp = cp.model_copy(
        update={
            "status": "FAILED",
            "failed_step": "proposal_from_finding_creator",
            "retryable": True,
            "issues": [],
            "error": "transient",
        }
    )
    mp.status_path.write_text(cp.model_dump_json(indent=2), encoding="utf-8")

    discover_calls.clear()
    research_calls.clear()
    pf_calls.clear()
    run_with_telemetry(make_input(repo), config=cfg)
    assert discover_calls == []
    assert research_calls == []
    assert pf_calls == ["v1/kv_offload"]
