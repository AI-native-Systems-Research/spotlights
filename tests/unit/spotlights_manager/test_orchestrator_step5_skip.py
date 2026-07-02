"""`skip_agent_proposals=True` short-circuits step 5: agent_proposals must
not be invoked, the module still finalizes after step 4, and no
agent_proposals sidecar is written (so the checkpoint records step 4 as the
furthest completed step)."""

from __future__ import annotations

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


def test_skip_agent_proposals_finalizes_after_step4(
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

    def _no_step5(*args, **kwargs):  # pragma: no cover - asserted not called
        raise AssertionError("step 5 must not run when skip_agent_proposals")

    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _no_step5)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
        skip_agent_proposals=True,
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SUCCEEDED"

    # No agent_proposals sidecar; checkpoint records step 4 as furthest step.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
    assert mp.proposal_from_finding_path.exists()
    assert not mp.agent_proposals_path.exists()

    state = P.read_module_state(mp)
    assert state.checkpoint is not None
    assert state.checkpoint.last_step == "proposal_from_finding_creator"
    assert state.agent_proposals is None
