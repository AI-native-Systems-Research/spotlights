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


def _run_with_old_signature_research(
    tmp_path: Path, monkeypatch
) -> tuple[list[dict], object]:
    """Drive a full run whose step-3 runner predates the `claude_model` kwarg.

    Returns the kwargs each call saw, plus the run result.
    """
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

    seen: list[dict] = []

    # The pre-change signature: `segment` but no `claude_model`, and no
    # `**kwargs` to absorb one. The in-repo fakes all use `**_kw`, which is why
    # they cannot catch a kwarg passed unconditionally.
    def _old_research(inp, options=None, *, runner=None, runners=None, segment=None):
        seen.append({"segment": segment})
        return make_research_output()

    monkeypatch.setattr(orch, "research_module", _old_research)
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
    return seen, run_with_telemetry(make_input(repo), config=cfg)


def test_step3_runner_predating_the_model_argument_still_works(
    tmp_path: Path, monkeypatch
) -> None:
    """`research_module` is a documented monkeypatch seam, so the new kwarg must
    be passed only when a model is set.

    An unconditional `claude_model=` raises `TypeError` on a runner written
    against the old signature — and the step-3 call site is wrapped in a broad
    `except Exception`, so that lands as a retryable "deep research failed" on
    every module instead of a visible signature error. Same guarantee steps 4, 5
    and apply already have.
    """
    seen, result = _run_with_old_signature_research(tmp_path, monkeypatch)

    assert len(seen) == 1, "step 3 never reached the runner"
    mr = result.module_runs["v1/kv_offload"]
    assert mr.status == "SUCCEEDED"


def test_step3_forwards_the_model_when_one_is_configured(
    tmp_path: Path, monkeypatch
) -> None:
    """The kwarg is still passed when a model is pinned — the old-signature
    tolerance above must not become "never forward the model"."""
    from spotlights_engine.model_config import ModelConfig

    seen: dict = {}

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

    def _research(inp, options=None, **kw):
        seen.update(kw)
        return make_research_output()

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
        models=ModelConfig(claude="aws/claude-opus-4-8"),
    )
    run_with_telemetry(make_input(repo), config=cfg)

    assert seen.get("claude_model") == "aws/claude-opus-4-8"
