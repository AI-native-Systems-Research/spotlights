"""Zero-findings short-circuit: step 4 must not invoke `claude` when
research returned no findings, but every candidate must still advance to
`FINDING_PROPOSALS_CREATED` with `deep_research_proposals=[]`."""

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


def test_zero_findings_synthesizes_step4_without_invoking_claude(
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
        lambda inp, options=None: make_research_output(n_findings=0),
    )
    pf_calls: list = []

    def _no_step4(*args, **kwargs):  # pragma: no cover - asserted not called
        pf_calls.append(args)
        raise AssertionError("step 4 must not run for zero findings")

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _no_step4)
    patch_agent_proposals(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)
    assert pf_calls == []

    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SUCCEEDED"
    assert mr.findings == []
    assert mr.candidates is not None
    for c in mr.candidates.candidates:
        # Step 5 advances every candidate from FINDING_PROPOSALS_CREATED to
        # AGENT_PROPOSALS_CREATED; deep_research_proposals stays empty.
        assert c.state == "AGENT_PROPOSALS_CREATED"
        assert c.deep_research_proposals == []

    # The synthetic sidecar must be on disk so a resume sees the step-4 outputs.
    paths = P.ManagerPaths(artifacts)
    mp = paths.for_module("v1/kv_offload")
    assert mp.proposal_from_finding_path.exists()
