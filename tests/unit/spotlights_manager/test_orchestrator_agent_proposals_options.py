"""Step-5 config: caller config is copied and module-owned paths are applied."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.spotlights_manager import (
    ModuleFilter,
    SpotlightsManagerConfig,
    run_with_telemetry,
)
from spotlights_engine.spotlights_manager import orchestrator as orch
from tests.unit.spotlights_manager._fakes import (
    make_agent_proposals_result,
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


def test_agent_proposals_options_per_module_override(
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

    seen_cfgs: list[AgentProposalsConfig] = []

    def _capture(inp, *, config, claude_runner=None, codex_runner=None):
        seen_cfgs.append(config)
        return make_agent_proposals_result(
            inp.candidates,
            total_duration_s=1.25,
            per_candidate_durations_s={"cand-0001": {"claude": 0.5, "codex": 0.7}},
        )

    monkeypatch.setattr(orch, "create_agent_proposals_with_telemetry", _capture)

    caller_ap = AgentProposalsConfig(
        repo_path=Path("/tmp/will-be-overridden"),
        artifacts_dir=Path("/tmp/will-also-be-overridden"),
        max_parallel_candidates=3,
        claude_max_turns=42,
        claude_wallclock_s=123,
        codex_wallclock_s=456,
        codex_model="gpt-test",
        codex_reasoning_effort="high",
        debug_first_n_candidates=1,
        claude_agent_name="claude-a",
        codex_agent_name="codex-b",
    )
    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        agent_proposals=caller_ap,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    result = run_with_telemetry(make_input(repo), config=cfg)

    assert len(seen_cfgs) == 1
    used = seen_cfgs[0]
    assert used.repo_path == repo
    assert used.artifacts_dir is not None
    assert "v1.kv_offload" in str(used.artifacts_dir)
    assert used.max_parallel_candidates == 3
    assert used.claude_max_turns == 42
    assert used.claude_wallclock_s == 123
    assert used.codex_wallclock_s == 456
    assert used.codex_model == "gpt-test"
    assert used.codex_reasoning_effort == "high"
    assert used.debug_first_n_candidates == 1
    assert used.claude_agent_name == "claude-a"
    assert used.codex_agent_name == "codex-b"

    telemetry = result.per_module_telemetry["v1.kv_offload"]
    assert telemetry.agent_proposals_duration_s == 1.25
    assert telemetry.agent_proposals_per_candidate_durations_s == {
        "cand-0001": {"claude": 0.5, "codex": 0.7}
    }

    assert caller_ap.repo_path == Path("/tmp/will-be-overridden")
    assert caller_ap.artifacts_dir == Path("/tmp/will-also-be-overridden")
