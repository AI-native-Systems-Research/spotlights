"""Step-4 config: the caller's `ProposalFromFindingConfig` is copied, never
mutated; `repo_path` and `artifacts_dir` are overridden per-module."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
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


def test_proposal_from_finding_options_per_module_override(
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
        lambda inp, options=None: make_research_output(n_findings=1),
    )

    seen_cfgs: list[ProposalFromFindingConfig] = []
    base_fake = None

    def _capture(inp, *, config, runner=None):
        seen_cfgs.append(config)
        return _fake_result(inp)

    def _fake_result(inp):
        # Reuse the helper inside patch_proposal_from_finding — but we need
        # the result directly because we're not delegating.
        from spotlights_engine.proposal_from_finding_creator.api import (
            ProposalFromFindingCreatorResult,
        )
        from spotlights_engine.schemas.candidate import Candidates
        from spotlights_engine.schemas.pipeline import (
            ProposalFromFindingCreatorOutput,
        )

        cands = inp.candidates
        advanced = Candidates(
            module_qualified_name=cands.module_qualified_name,
            candidates=[
                c.model_copy(
                    update={
                        "state": "FINDING_PROPOSALS_CREATED",
                        "deep_research_proposals": [],
                    }
                )
                for c in cands.candidates
            ],
        )
        return ProposalFromFindingCreatorResult(
            output=ProposalFromFindingCreatorOutput(
                candidates=advanced, issues=[]
            ),
            per_pair_durations_s={},
            total_duration_s=0.0,
        )

    monkeypatch.setattr(orch, "create_proposals_with_telemetry", _capture)

    caller_pf = ProposalFromFindingConfig(
        repo_path=Path("/tmp/will-be-overridden"),
        artifacts_dir=Path("/tmp/will-also-be-overridden"),
        max_parallel_pairs=3,
        claude_max_turns=42,
    )
    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        proposal_from_finding=caller_pf,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    assert len(seen_cfgs) == 1
    used = seen_cfgs[0]
    assert used.repo_path == repo
    assert used.artifacts_dir is not None
    assert "v1.kv_offload" in str(used.artifacts_dir)
    assert used.max_parallel_pairs == 3
    assert used.claude_max_turns == 42

    # Caller's config object is unchanged.
    assert caller_pf.repo_path == Path("/tmp/will-be-overridden")
    assert caller_pf.artifacts_dir == Path("/tmp/will-also-be-overridden")


def test_proposal_from_finding_default_when_caller_none(
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
        lambda inp, options=None: make_research_output(n_findings=1),
    )
    patch_proposal_from_finding(monkeypatch, orch)

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    # Should not raise — defaults are used.
    run_with_telemetry(make_input(repo), config=cfg)
