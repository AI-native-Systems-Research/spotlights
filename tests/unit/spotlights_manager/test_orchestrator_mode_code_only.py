"""`mode=code_only` skips deep-research (step 3) and proposal-from-finding
(step 4) entirely. Pipeline becomes
extractor → candidate-discovery → agent-proposals.

Asserts:
- `research_module` is NOT invoked.
- `create_proposals_with_telemetry` (step 4 runner) is NOT invoked.
- Stage 5 (agent_proposals) still runs over candidates.
- Per-candidate output has `deep_research_proposals=[]` and the
  expected agent_proposals attached.
- Findings on the module run are empty.
- Synthetic deep-research and proposal-from-finding sidecars are on
  disk so a resume sees consistent state.
"""

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
    make_tree,
    patch_agent_proposals,
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


def test_code_only_mode_skips_step3_and_step4_runs_step5(
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

    research_calls: list = []

    def _no_research(*args, **kwargs):  # pragma: no cover — asserted unused
        research_calls.append(args)
        raise AssertionError("step 3 must not run for mode=code_only")

    monkeypatch.setattr(orch, "research_module", _no_research)

    pf_calls: list = []

    def _no_step4(*args, **kwargs):  # pragma: no cover — asserted unused
        pf_calls.append(args)
        raise AssertionError("step 4 must not run for mode=code_only")

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _no_step4)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    result = run_with_telemetry(
        make_input(repo, mode="code_only"), config=cfg
    )

    assert research_calls == []
    assert pf_calls == []

    mr = result.module_runs["v1.kv_offload"]
    assert mr.status == "SUCCEEDED"
    assert mr.findings == []
    assert mr.candidates is not None
    for c in mr.candidates.candidates:
        assert c.state == "AGENT_PROPOSALS_CREATED"
        assert c.deep_research_proposals == []

    # Synthetic sidecars must be on disk so resume sees a consistent state.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1.kv_offload")
    assert mp.deep_research_path.exists()
    assert mp.proposal_from_finding_path.exists()


def test_full_mode_default_still_invokes_research(
    monkeypatch, repo: Path, artifacts: Path
) -> None:
    """Sanity check: with mode unset (default `full`), step 3 IS called."""
    from tests.unit.spotlights_manager._fakes import make_research_output

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

    research_calls: list = []

    def _record_research(inp, options=None):
        research_calls.append(inp)
        return make_research_output(n_findings=0)

    monkeypatch.setattr(orch, "research_module", _record_research)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)
    assert len(research_calls) == 1
