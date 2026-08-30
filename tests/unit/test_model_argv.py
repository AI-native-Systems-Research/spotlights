"""Every agent call site emits `--model` when configured, and nothing when not.

The engine has six Claude call sites and they were all built without a model
flag; the risk this file guards is a step being missed, or an empty value
reaching argv as `--model ""`. Each test drives the real argv builder and
inspects the command, without launching a subprocess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spotlights_engine.module_deep_research.claude_exec import (
    ClaudeExecClient,
    ClaudeExecOptions,
)
from spotlights_engine.module_deep_research.codex_exec import (
    CodexExecClient,
    CodexExecOptions,
)

MODEL = "aws/claude-opus-4-8"


class _StopAfterArgv(RuntimeError):
    """Raised by a stub once argv has been captured, to skip the real work."""


def _captured_argv(monkeypatch) -> list[list[str]]:
    """Intercept `subprocess.run`, record argv, and return a benign result.

    Returning an empty successful process (rather than raising) keeps each
    caller on its normal path: it parses nothing useful and reports a failure
    result, which is fine — the assertion is about the command, not the outcome.
    """
    seen: list[list[str]] = []

    def fake_run(argv, *args, **kwargs):
        seen.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return seen


# --- step 1: modules extractor -------------------------------------------------


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_extractor_stage_argv(tmp_path, monkeypatch, model, expected):
    from spotlights_engine.modules_extractor import claude_stage

    seen: list[list[str]] = []

    monkeypatch.setattr(claude_stage, "resolve_and_check", lambda **kw: ["/bin/claude"])
    monkeypatch.setattr(claude_stage, "build_schema_text", lambda t: "{}")

    def fake_stream(*, argv, **kwargs):
        seen.append(list(argv))
        raise _StopAfterArgv

    monkeypatch.setattr(claude_stage, "run_streaming_claude", fake_stream)

    with pytest.raises(_StopAfterArgv):
        claude_stage.run_structured_claude_stage(
            output_type=CodexExecOptions,  # any pydantic model; schema is stubbed
            repo_path=tmp_path,
            prompt="p",
            stage_name="s",
            attempt_dir=None,
            claude_model=model,
            max_turns=3,
            timeout_s=5,
        )

    assert seen, "argv was never built"
    assert (("--model" in seen[0]) and (MODEL in seen[0])) is expected


# --- step 3: deep research (both CLIs) ----------------------------------------


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_deep_research_claude_argv(model, expected):
    cmd = ClaudeExecClient(ClaudeExecOptions(model=model)).build_command()
    assert (("--model" in cmd) and (MODEL in cmd)) is expected


@pytest.mark.parametrize("model,expected", [("gpt-5.5", True), (None, False)])
def test_deep_research_codex_argv(model, expected, tmp_path):
    client = CodexExecClient(CodexExecOptions(cwd=tmp_path, model=model))
    cmd, _ = client.build_command()
    assert (("--model" in cmd) and ("gpt-5.5" in cmd)) is expected


# --- steps 4 and 5, and apply ------------------------------------------------


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_proposal_from_finding_argv(tmp_path, monkeypatch, model, expected):
    from spotlights_engine.proposal_from_finding_creator import claude_exec

    seen = _captured_argv(monkeypatch)
    monkeypatch.setattr(claude_exec.shutil, "which", lambda _: "/bin/claude")
    claude_exec.run_pair(
        pair_key="k",
        prompt="p",
        schema_text="{}",
        repo_path=tmp_path,
        max_turns=3,
        wallclock_s=5,
        claude_model=model,
    )
    assert seen, "argv was never built"
    assert (("--model" in seen[0]) and (MODEL in seen[0])) is expected


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_agent_proposals_claude_argv(tmp_path, monkeypatch, model, expected):
    from spotlights_engine.agent_proposals import claude_exec

    seen = _captured_argv(monkeypatch)
    monkeypatch.setattr(claude_exec.shutil, "which", lambda _: "/bin/claude")
    claude_exec.run_candidate_claude(
        candidate_id="c",
        prompt="p",
        schema_text="{}",
        repo_path=tmp_path,
        max_turns=3,
        wallclock_s=5,
        claude_model=model,
    )
    assert seen, "argv was never built"
    assert (("--model" in seen[0]) and (MODEL in seen[0])) is expected


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_apply_argv(tmp_path, monkeypatch, model, expected):
    from spotlights_engine.one_shot_apply import claude_exec

    seen = _captured_argv(monkeypatch)
    monkeypatch.setattr(claude_exec.shutil, "which", lambda _: "/bin/claude")
    claude_exec.run_apply_claude(
        candidate_id="c",
        prompt="p",
        worktree=tmp_path,
        max_turns=3,
        wallclock_s=5,
        claude_model=model,
    )
    assert seen, "argv was never built"
    assert (("--model" in seen[0]) and (MODEL in seen[0])) is expected


def test_deep_research_claude_runner_gets_the_model(tmp_path):
    """`select_runners` builds the default Claude runner — the model must land."""
    from spotlights_engine.module_deep_research.orchestration import select_runners

    runners = select_runners(
        repo_path=Path(tmp_path),
        codex_options=None,
        runner=None,
        runners=None,
        enable_claude_search=True,
        claude_model=MODEL,
    )
    claude = [r for r in runners if isinstance(r, ClaudeExecClient)]
    assert len(claude) == 1
    assert claude[0].options.model == MODEL
    assert "--model" in claude[0].build_command()


# --- blank means inherit, all the way down --------------------------------------


@pytest.mark.parametrize("model,expected", [("gpt-5.5", True), (None, False)])
def test_discovery_codex_argv_model_is_optional(tmp_path, monkeypatch, model, expected):
    """Step 2's Codex model must be able to inherit too.

    It used to be a non-optional `str` pinned to `gpt-5.5`, so a blank value in
    `models.yaml` could never actually reach the CLI's own default — the one
    place where "blank means inherit" quietly was not true.
    """
    from spotlights_engine.candidate_discovery import agents as discovery_agents
    from spotlights_engine.candidate_discovery.api import DiscoveryConfig

    monkeypatch.setattr(discovery_agents.shutil, "which", lambda _n: "/bin/codex")
    cfg = DiscoveryConfig(
        repo_path=tmp_path, artifacts_dir=tmp_path / "a", codex_model=model
    )
    runner = discovery_agents.CodexRunner(cfg)
    schema_path = tmp_path / "s.json"
    schema_path.write_text("{}")
    iter_dir = tmp_path / "iter"
    iter_dir.mkdir()
    argv = runner._build_argv(schema_path=schema_path, iter_dir=iter_dir)

    c_values = [argv[i + 1] for i, x in enumerate(argv) if x == "-c"]
    assert any('model="gpt-5.5"' == v for v in c_values) is expected
    # The reasoning-effort `-c` is unconditional and must survive either way.
    assert any("model_reasoning_effort=" in v for v in c_values)


@pytest.mark.parametrize("model,expected", [(MODEL, True), (None, False)])
def test_legacy_single_shot_extractor_argv(tmp_path, monkeypatch, model, expected):
    """The `two_phase=False` path is a separate argv builder and was missed once."""
    from spotlights_engine.modules_extractor import agent as extractor_agent

    seen: list[list[str]] = []

    monkeypatch.setattr(
        extractor_agent, "resolve_claude_argv0", lambda _b: ["/bin/claude"]
    )
    monkeypatch.setattr(
        extractor_agent, "ensure_claude_available", lambda *a, **k: None, raising=False
    )

    def fake_stream(*, argv, **kwargs):
        seen.append(list(argv))
        raise _StopAfterArgv

    monkeypatch.setattr(
        extractor_agent, "run_streaming_claude", fake_stream, raising=False
    )

    with pytest.raises(_StopAfterArgv):
        extractor_agent.run_extraction(
            repo_path=tmp_path,
            prompt="p",
            claude_model=model,
            max_turns=3,
            timeout_s=5,
            on_event=None,
        )

    assert seen, "argv was never built"
    assert (("--model" in seen[0]) and (MODEL in seen[0])) is expected


def test_public_research_module_forwards_the_model(tmp_path, monkeypatch):
    """The non-telemetry wrapper is the documented entrypoint for library users."""
    from spotlights_engine.module_deep_research import api as dr_api

    seen: dict[str, object] = {}

    def fake_with_telemetry(request, **kwargs):
        seen.update(kwargs)
        raise _StopAfterArgv

    monkeypatch.setattr(dr_api, "research_module_with_telemetry", fake_with_telemetry)
    with pytest.raises(_StopAfterArgv):
        dr_api.research_module(object(), claude_model=MODEL)

    assert seen.get("claude_model") == MODEL
