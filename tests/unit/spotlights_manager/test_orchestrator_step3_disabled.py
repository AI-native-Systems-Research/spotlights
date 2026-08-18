"""`--no-deep-research` (`SpotlightsManagerInput.enable_deep_research=False`):
step 3 must not be entered at all, an empty `module_deep_research.json` sidecar
must still be written (so resume is a no-op), and disabling a step on purpose
must not degrade the module."""

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
from spotlights_engine.utils.schema_compat import proposals_from
from tests.unit.spotlights_manager._fakes import (
    make_discovery_result,
    make_extractor_result,
    make_input,
    make_tree,
    patch_agent_proposals,
)

QN = "v1/kv_offload"


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


def _patch_steps_1_2(monkeypatch) -> None:
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


def _forbid_research(monkeypatch) -> None:
    def _no_step3(*args, **kwargs):  # pragma: no cover - asserted not called
        raise AssertionError("step 3 must not run when deep research is disabled")

    monkeypatch.setattr(orch, "research_module", _no_step3)


def _cfg(artifacts: Path) -> SpotlightsManagerConfig:
    return SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=[QN]),
    )


def test_disabled_deep_research_never_calls_research_module(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _patch_steps_1_2(monkeypatch)
    _forbid_research(monkeypatch)
    patch_agent_proposals(monkeypatch, orch)

    result = run_with_telemetry(
        make_input(repo, enable_deep_research=False), config=_cfg(artifacts)
    )

    mr = result.module_runs[QN]
    # The run must *complete*, not merely produce no findings: without this the
    # assertion below is satisfied by a module that FAILED inside step 3 (the
    # orchestrator turns the AssertionError into a retryable FAILED checkpoint).
    assert mr.status == "SUCCEEDED"
    assert mr.findings == []
    assert mr.candidates is not None
    assert mr.candidates.candidates
    for c in mr.candidates.candidates:
        assert proposals_from(c, "research_finding") == []


def test_disabled_deep_research_writes_empty_sidecar(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _patch_steps_1_2(monkeypatch)
    _forbid_research(monkeypatch)
    patch_agent_proposals(monkeypatch, orch)

    statuses: list[str] = []
    real_write_checkpoint = P.write_checkpoint

    def _spy(module_paths, checkpoint):
        statuses.append(checkpoint.status)
        real_write_checkpoint(module_paths, checkpoint)

    monkeypatch.setattr(P, "write_checkpoint", _spy)

    run_with_telemetry(
        make_input(repo, enable_deep_research=False), config=_cfg(artifacts)
    )

    mp = P.ManagerPaths(artifacts).for_module(QN)
    assert mp.deep_research_path.exists()
    payload = json.loads(mp.deep_research_path.read_text(encoding="utf-8"))
    assert payload["output"]["findings"] == []
    assert payload["output"]["issues"] == []
    assert payload["duration_s"] == 0.0
    # The search log sits next to the JSON as usual, even with zero queries.
    assert mp.deep_research_search_log_path.exists()
    # No CLI session ran, so no usage records were minted.
    assert not (mp.dir / "module_deep_research.usage").exists()

    # The disabled branch still lands the DEEP_RESEARCHED checkpoint on the
    # ladder before steps 4/5 carry the module to SUCCEEDED.
    assert "DEEP_RESEARCHED" in statuses
    status = json.loads(mp.status_path.read_text(encoding="utf-8"))
    assert status["status"] == "SUCCEEDED"


def test_module_status_is_succeeded_when_disabled(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    _patch_steps_1_2(monkeypatch)
    _forbid_research(monkeypatch)
    patch_agent_proposals(monkeypatch, orch)

    result = run_with_telemetry(
        make_input(repo, enable_deep_research=False), config=_cfg(artifacts)
    )

    mr = result.module_runs[QN]
    # Disabling a step on purpose is not a degradation: no StepIssue is recorded.
    assert mr.status == "SUCCEEDED"
    assert mr.issues == []


def test_disabled_deep_research_resume_is_a_no_op(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """Regression test for the "write the empty sidecar" requirement: without
    it every resume would see `state.deep_research is None`, re-enter step 3 and
    (via `redo_step3`) clear + re-run steps 4 and 5."""
    _patch_steps_1_2(monkeypatch)
    _forbid_research(monkeypatch)
    patch_agent_proposals(monkeypatch, orch)

    inp = make_input(repo, enable_deep_research=False)
    run_with_telemetry(inp, config=_cfg(artifacts))

    # Second pass: every per-module step entry point raises.
    def _no_step(*args, **kwargs):  # pragma: no cover - asserted not called
        raise AssertionError("resume must not re-enter any module step")

    monkeypatch.setattr(orch, "discover", _no_step)
    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _no_step)
    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _no_step)

    result = run_with_telemetry(inp, config=_cfg(artifacts))

    mr = result.module_runs[QN]
    assert mr.status == "SUCCEEDED"
    assert mr.findings == []
