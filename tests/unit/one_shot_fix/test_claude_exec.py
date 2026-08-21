"""`run_fix_claude` against a fake `claude` binary."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.one_shot_fix.claude_exec import (
    ensure_claude_available,
    run_fix_claude,
)
from tests.unit.one_shot_fix._fixtures import (
    FAKE_CLAUDE_ARGV_RECORDER,
    FAKE_CLAUDE_FAILURE,
    FAKE_CLAUDE_NO_RESULT_EVENT,
    FAKE_CLAUDE_SUCCESS,
    prepend_to_path,
    write_fake_claude,
)
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="fake claude shim is a POSIX shell script"
)


def test_successful_run_edits_the_worktree_and_captures_usage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    fake = write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_SUCCESS)
    prepend_to_path(monkeypatch, fake.parent)

    result = run_fix_claude(
        candidate_id="cand-v1_attention-0002",
        prompt="do the thing",
        worktree=repo,
        max_turns=5,
        wallclock_s=30,
    )

    assert result.error is None
    assert "# touched by fake claude" in (repo / CAND_FILE).read_text(encoding="utf-8")
    assert result.usage is not None
    assert result.usage.input == 100
    assert result.usage.output == 50
    assert result.duration_s >= 0


def test_nonzero_exit_is_reported_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    fake = write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_FAILURE)
    prepend_to_path(monkeypatch, fake.parent)

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=30
    )

    assert result.error is not None
    assert "exit=3" in result.error
    assert b"boom" in result.stderr


def test_missing_result_event_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unlike agent_proposals, the deliverable is edited files, not a payload.

    A stream with no terminal result event costs us the usage numbers, not the
    fix — so it degrades to `usage is None` rather than failing the run.
    """
    repo = make_repo(tmp_path)
    prepend_to_path(
        monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_NO_RESULT_EVENT).parent
    )

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=30
    )

    assert result.error is None
    assert result.usage is None


def test_argv_carries_the_prompt_and_edit_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("TEST_ARGV_FILE", str(argv_file))
    prepend_to_path(
        monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_ARGV_RECORDER).parent
    )

    run_fix_claude(
        candidate_id="c1",
        prompt="THE-PROMPT",
        worktree=repo,
        max_turns=7,
        wallclock_s=30,
    )

    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert "-p" in argv
    assert "THE-PROMPT" in argv
    assert "acceptEdits" in argv  # must be able to edit; not --permission-mode plan
    assert "7" in argv  # --max-turns


def test_argv_denies_git_write_commands_that_escape_the_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--disallowedTools` must deny the git write commands whose effects reach
    past the throwaway worktree into the main repository (or a remote) —
    surviving `git worktree remove --force` + `git worktree prune`, or moving
    the agent's work out of the working tree where `collect_patch` can see it.

    File edits are already contained by `cwd`; this is the only guard on git.
    """
    repo = make_repo(tmp_path)
    argv_file = tmp_path / "argv.txt"
    monkeypatch.setenv("TEST_ARGV_FILE", str(argv_file))
    prepend_to_path(
        monkeypatch, write_fake_claude(tmp_path / "bin", script=FAKE_CLAUDE_ARGV_RECORDER).parent
    )

    run_fix_claude(
        candidate_id="c1",
        prompt="THE-PROMPT",
        worktree=repo,
        max_turns=7,
        wallclock_s=30,
    )

    argv = argv_file.read_text(encoding="utf-8").splitlines()
    assert "--disallowedTools" in argv
    denied = argv[argv.index("--disallowedTools") + 1]
    for pattern in (
        "Bash(git commit:*)",
        "Bash(git stash:*)",
        "Bash(git branch:*)",
        "Bash(git checkout:*)",
        "Bash(git push:*)",
        "Bash(git tag:*)",
        "Bash(git worktree:*)",
    ):
        assert pattern in denied


def test_timeout_is_reported_as_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = make_repo(tmp_path)
    prepend_to_path(
        monkeypatch,
        write_fake_claude(tmp_path / "bin", script="#!/bin/sh\nsleep 5\n").parent,
    )

    result = run_fix_claude(
        candidate_id="c1", prompt="p", worktree=repo, max_turns=5, wallclock_s=1
    )

    assert result.error is not None
    assert "timed out" in result.error


def test_ensure_claude_available_raises_when_not_on_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(AgentProposalsSetupError):
        ensure_claude_available()
