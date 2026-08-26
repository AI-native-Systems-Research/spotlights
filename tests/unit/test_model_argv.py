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
        raise RuntimeError("stop after argv capture")

    monkeypatch.setattr(claude_stage, "run_streaming_claude", fake_stream)

    with pytest.raises(Exception):
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
