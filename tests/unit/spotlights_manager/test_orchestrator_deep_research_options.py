"""Deep-research options: caller config is copied, never mutated; per-module
overrides for `cwd` and `output_last_message` are applied."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
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
    patch_agent_proposals,
    patch_proposal_from_finding,
)


def test_deep_research_options_per_module_override(tmp_path: Path, monkeypatch) -> None:
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

    seen_options: list[CodexExecOptions] = []

    def _research(inp, options=None):
        seen_options.append(options)
        return make_research_output()

    monkeypatch.setattr(orch, "research_module", _research)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    caller_options = CodexExecOptions(
        cwd=Path("/tmp/will-be-overridden"),
        codex_bin="my-codex",
        model="custom-model",
        output_last_message=Path("/tmp/will-also-be-overridden"),
    )
    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        deep_research=caller_options,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    assert len(seen_options) == 1
    used = seen_options[0]
    assert used.cwd == repo
    assert used.codex_bin == "my-codex"
    assert used.model == "custom-model"
    assert used.output_last_message is not None
    assert "v1.kv_offload" in str(used.output_last_message)

    # The caller's options object is unchanged.
    assert caller_options.cwd == Path("/tmp/will-be-overridden")
    assert caller_options.output_last_message == Path("/tmp/will-also-be-overridden")


def test_deep_research_options_default_when_caller_none(
    tmp_path: Path, monkeypatch
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

    seen: list[CodexExecOptions] = []
    monkeypatch.setattr(
        orch,
        "research_module",
        lambda inp, options=None: (seen.append(options), make_research_output())[1],
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    used = seen[0]
    assert used.cwd == repo
    assert used.output_last_message is not None
