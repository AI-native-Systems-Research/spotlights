"""Deep-research options: caller config is copied, never mutated; per-module
overrides for `cwd` and `output_last_message` are applied."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.costing.usage import AgentUsage, CliUsage
from spotlights_engine.module_deep_research.api import ModuleDeepResearchResult
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
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

    def _research(inp, options=None, **_kw):
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
        output_folder=artifacts.parent / "output",
        deep_research=caller_options,
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    assert len(seen_options) == 1
    used = seen_options[0]
    assert used.cwd == repo
    assert used.codex_bin == "my-codex"
    assert used.model == "custom-model"
    assert used.output_last_message is not None
    assert "v1_kv_offload" in str(used.output_last_message)

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
        lambda inp, options=None, **_kw: (seen.append(options), make_research_output())[1],
    )
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    used = seen[0]
    assert used.cwd == repo
    assert used.output_last_message is not None


def test_step3_usage_records_are_candidate_attributed(
    tmp_path: Path, monkeypatch
) -> None:
    """Decision D7: step-3 usage records must carry candidate-qualified
    invocation ids (`cand-...:codex`) and the per-(candidate, runner) wall time
    as the fallback duration — not one `deep-research:<cli>` record per runner
    each charged the whole step-3 duration.

    This exercises the manager's real usage-attribution code path by patching
    `research_module` with a fake that returns the telemetry-rich
    `ModuleDeepResearchResult` (per-candidate usages + durations), the way the
    real step 3 does. The default `make_research_output` fake returns the bare
    output and hits the manager's `{}` fallback, so it never covers this."""
    tree = make_tree()
    monkeypatch.setattr(
        orch,
        "extract_with_telemetry",
        lambda inp, *, config=None: make_extractor_result(tree),
    )
    monkeypatch.setattr(
        orch,
        "discover",
        # Two candidates so cross-attribution and per-candidate durations matter.
        lambda inp, *, config: make_discovery_result(
            inp.module_qualified_name, n_candidates=2
        ),
    )

    cand_a = "cand-v1_kv_offload-0001"
    cand_b = "cand-v1_kv_offload-0002"

    def _research(inp, options=None, **_kw):
        usage = AgentUsage(input=100, output=50, model="gpt-x")
        return ModuleDeepResearchResult(
            output=make_research_output(),
            usages_by_candidate={
                cand_a: [CliUsage(cli="codex", usage=usage)],
                cand_b: [CliUsage(cli="codex", usage=usage)],
            },
            per_candidate_durations_s={
                cand_a: {"codex": 1.25},
                cand_b: {"codex": 3.75},
            },
        )

    monkeypatch.setattr(orch, "research_module", _research)
    patch_proposal_from_finding(monkeypatch, orch)
    patch_agent_proposals(monkeypatch, orch)

    repo = tmp_path / "repo"
    repo.mkdir()
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()

    cfg = SpotlightsManagerConfig(
        artifacts_dir=artifacts,
        output_folder=artifacts.parent / "output",
        module_filter=ModuleFilter(include=["v1/kv_offload"]),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    module_paths = P.ManagerPaths(artifacts).for_module("v1/kv_offload")
    records, _notes = P.read_usage_records(module_paths, "module_deep_research")

    by_invocation = {r.invocation_id: r for r in records}
    # Candidate-qualified invocation ids, one per (candidate, runner) — never the
    # old shared `deep-research:codex`.
    assert set(by_invocation) == {f"{cand_a}:codex", f"{cand_b}:codex"}
    assert all(not r.invocation_id.startswith("deep-research:") for r in records)
    # Each record carries its own candidate/runner wall time, not a shared value.
    assert by_invocation[f"{cand_a}:codex"].api_time_s == 1.25
    assert by_invocation[f"{cand_b}:codex"].api_time_s == 3.75
