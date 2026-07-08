"""Tests for `unified_runner.shared_extraction`.

Covers:
- DR pre-population matches what `_ensure_resume_compatible` /
  `_run_extractor_if_needed` expect (so DR's resume sees the cached state).
- Signal pre-population writes both the artifact AND a status mark stage 02
  done (the dual condition `_is_complete` checks).
- The ProjectTree round-trips through both pre-population helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput
from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline.layout import RunDirLayout
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.api import SpotlightsManagerConfig
from spotlights_engine.unified_runner.shared_extraction import (
    populate_dr_run_dir,
    populate_signal_run_dir,
)


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
        duration_s=12.5,
        cost_usd=0.0,
        input_tokens=100,
        output_tokens=50,
    )


def test_populate_signal_writes_artifact_and_status(tmp_path):
    tree = _tree()
    signal_dir = tmp_path / "signal"
    populate_signal_run_dir(signal_run_dir=signal_dir, project_tree=tree)

    layout = RunDirLayout(signal_dir.resolve())
    artifact = layout.stage_artifact("02", shape="single")
    assert artifact.exists()
    raw = json.loads(artifact.read_text(encoding="utf-8"))
    assert ProjectTree.model_validate(raw) == tree

    status = json.loads(layout.status_path.read_text(encoding="utf-8"))
    assert status["stages"]["02"]["state"] == "done"
    assert status["stages"]["02"]["started_at"]
    assert status["stages"]["02"]["ended_at"]


def test_populate_signal_preserves_other_stage_status(tmp_path):
    """Re-populating stage 02 must not clobber any pre-existing stage status."""
    signal_dir = tmp_path / "signal"
    signal_dir.mkdir()
    layout = RunDirLayout(signal_dir.resolve())
    layout.status_path.write_text(
        json.dumps(
            {
                "stages": {
                    "01": {
                        "state": "done",
                        "started_at": "2026-06-22T08:00:00Z",
                        "ended_at": "2026-06-22T08:01:00Z",
                        "error": None,
                        "issues": [],
                        "model": None,
                        "cost_usd": None,
                        "duration_s": None,
                    }
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    populate_signal_run_dir(signal_run_dir=signal_dir, project_tree=_tree())

    status = json.loads(layout.status_path.read_text(encoding="utf-8"))
    assert status["stages"]["01"]["state"] == "done"
    assert status["stages"]["02"]["state"] == "done"


def test_populate_dr_writes_extractor_outputs_and_manifest(tmp_path):
    tree = _tree()
    invocation = _invocation()
    dr_dir = tmp_path / "deep_research"

    dr_input = SpotlightsManagerInput(
        repo_path=Path("/tmp/repo"),
        context=SpotlightContext(objective="o"),
    )
    dr_config = SpotlightsManagerConfig(
        artifacts_dir=dr_dir,
        output_folder=dr_dir / "output",
    )

    populate_dr_run_dir(
        dr_artifacts_dir=dr_dir,
        dr_input=dr_input,
        dr_config=dr_config,
        project_tree=tree,
        invocation=invocation,
        extractor_duration_s=12.5,
    )

    paths = P.ManagerPaths(artifacts_dir=dr_dir)
    assert paths.project_tree_path.exists()
    assert paths.extractor_invocation_path.exists()
    assert paths.manifest_path.exists()

    on_disk_tree, on_disk_inv = P.read_extractor_outputs(paths)
    assert on_disk_tree == tree
    assert on_disk_inv is not None

    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
    assert manifest["extractor"]["completed"] is True
    assert manifest["extractor"]["duration_s"] == 12.5
    assert manifest["status"] == "RUNNING"


def test_populate_dr_fingerprint_matches_dr_helpers(tmp_path):
    """The manifest fingerprints must be identical to what DR computes
    natively, otherwise its `_ensure_resume_compatible` would reject."""
    tree = _tree()
    invocation = _invocation()
    dr_dir = tmp_path / "dr"
    ctx = SpotlightContext(objective="reduce p99", workload_hints=["batch=8"])

    dr_input = SpotlightsManagerInput(repo_path=Path("/tmp/repo"), context=ctx)
    dr_config = SpotlightsManagerConfig(
        artifacts_dir=dr_dir,
        output_folder=dr_dir / "out",
    )

    populate_dr_run_dir(
        dr_artifacts_dir=dr_dir,
        dr_input=dr_input,
        dr_config=dr_config,
        project_tree=tree,
        invocation=invocation,
        extractor_duration_s=1.0,
    )

    paths = P.ManagerPaths(artifacts_dir=dr_dir)
    manifest = json.loads(paths.manifest_path.read_text(encoding="utf-8"))

    expected_input_fp = P.build_input_fingerprint(
        repo_path=dr_input.repo_path,
        context=dr_input.context,
        max_findings_per_module=dr_input.max_findings_per_module,
        continue_on_module_failure=dr_input.continue_on_module_failure,
    )
    expected_config_fp = P.build_config_fingerprint(
        module_filter=dr_config.module_filter,
        extractor_cfg=dr_config.extractor,
        discovery_cfg=dr_config.discovery,
        deep_research_cfg=dr_config.deep_research,
        proposal_from_finding_cfg=dr_config.proposal_from_finding,
        agent_proposals_cfg=dr_config.agent_proposals,
    )
    assert manifest["input_fingerprint"] == expected_input_fp
    assert manifest["config_fingerprint"] == expected_config_fp


def test_extract_once_is_invoked_only_once(monkeypatch, tmp_path):
    """The unified runner runs extraction exactly once; smoke this at the
    helper level (the runner wires it up; here we just confirm the helper
    delegates correctly with no double-call surface)."""
    from spotlights_engine.unified_runner import shared_extraction as se

    calls = []

    class _FakeResult:
        def __init__(self):
            self.project_tree = _tree()
            self.invocation = _invocation()

    def _fake_extract(input, *, config=None, on_event=None):
        calls.append(input)
        return _FakeResult()

    monkeypatch.setattr(se, "extract_with_telemetry", _fake_extract)

    result = se.extract_once(repo_path=Path("/tmp/repo"), cache_dir=tmp_path / "cache")
    assert isinstance(result.project_tree, ProjectTree)
    assert len(calls) == 1
