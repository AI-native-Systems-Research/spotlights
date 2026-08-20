"""Worktree lifecycle and patch collection."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME
from spotlights_engine.one_shot_fix.worktree import (
    collect_patch,
    create_worktree,
    remove_worktree,
    require_git_repo,
)
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout


def test_require_git_repo_returns_head_sha(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = require_git_repo(repo)
    assert len(sha) == 40
    assert sha == _git(repo, "rev-parse", "HEAD").strip()


def test_require_git_repo_rejects_a_non_git_directory(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, git=False)
    with pytest.raises(NotAGitRepoError):
        require_git_repo(repo)


def test_worktree_is_detached_and_creates_no_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    sha = require_git_repo(repo)
    branches_before = _git(repo, "branch", "--list")
    wt = create_worktree(repo, sha)
    try:
        assert (wt.path / CAND_FILE).is_file()
        assert _git(wt.path, "rev-parse", "HEAD").strip() == sha
        # --detach: no branch named after the worktree dir appears in the repo.
        assert _git(repo, "branch", "--list") == branches_before
    finally:
        remove_worktree(wt)


def test_remove_worktree_prunes_and_is_idempotent(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    remove_worktree(wt)
    assert not wt.path.exists()
    assert _git(repo, "worktree", "list").count("\n") == 1  # only the main tree
    remove_worktree(wt)  # second call must not raise


def test_collect_patch_includes_a_modified_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text() + "# appended\n", encoding="utf-8")
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "# appended" in patch
    assert CAND_FILE in patch
    assert summary is None


def test_collect_patch_includes_an_added_file(tmp_path: Path) -> None:
    """The `git add -N` regression: a new file must not be silently dropped."""
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / "pkg" / "attn" / "table.py").write_text(
            "TILE_TABLE = {128: 64}\n", encoding="utf-8"
        )
        patch, _ = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "pkg/attn/table.py" in patch
    assert "TILE_TABLE" in patch


def test_collect_patch_extracts_and_excludes_the_change_summary(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / CHANGE_SUMMARY_NAME).write_text(
            "Swapped the heuristic for a table.\n", encoding="utf-8"
        )
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert summary == "Swapped the heuristic for a table.\n"
    assert CHANGE_SUMMARY_NAME not in patch


def test_collect_patch_is_empty_when_nothing_changed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        patch, summary = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert patch == ""
    assert summary is None


def test_remove_worktree_never_raises_even_when_git_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`remove_worktree`'s docstring promises it never raises.

    `_run_git`'s `try/except` re-raises `OSError`/`SubprocessError` (including
    a timeout) as `WorktreeError` regardless of `check=False` — `check` only
    guards the return-code check, not the exception boundary. So if the
    underlying git call blows up (e.g. it times out), `remove_worktree` must
    still swallow it: it runs in a `finally`, so letting the exception through
    would mask whatever error triggered the cleanup in the first place.

    We monkeypatch `_run_git` itself to always raise `WorktreeError`, rather
    than forcing a real timeout or corrupting the worktree path on disk: it
    exercises the exact boundary the finding describes (a `WorktreeError`
    escaping the git call inside `remove_worktree`) deterministically and
    without a 120s wait or brittle filesystem tricks.
    """
    import spotlights_engine.one_shot_fix.worktree as worktree_mod

    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))

    def _always_raise(*args: object, **kwargs: object) -> None:
        raise worktree_mod.WorktreeError("simulated git failure")

    monkeypatch.setattr(worktree_mod, "_run_git", _always_raise)

    remove_worktree(wt)  # must not raise, despite every _run_git call failing

    # Cleanup didn't happen via monkeypatched git calls, but rmtree still runs.
    assert not wt.parent.exists()


def test_collect_patch_survives_non_utf8_bytes_in_a_diff(tmp_path: Path) -> None:
    """Non-UTF8 bytes in an edited file must not crash `collect_patch`.

    `_run_git` decodes `git diff`'s output with `text=True` and strict
    decoding by default. If the agent's edit introduces bytes invalid under
    the locale's encoding, `git diff` still emits them verbatim on stdout, and
    strict decoding raises `UnicodeDecodeError` (a `ValueError`, not caught by
    `_run_git`'s `except (OSError, subprocess.SubprocessError)`), so it
    propagates out of `collect_patch` unwrapped. The patch should still be
    collected — degraded but present — rather than lost entirely.
    """
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        # Append invalid UTF-8 bytes (no NUL byte, so git still treats the
        # file as text and includes the raw bytes in the diff rather than
        # reporting it as binary).
        target.write_bytes(target.read_bytes() + b"\n# bad bytes: \xff\xfe end\n")
        patch, _ = collect_patch(wt)  # must not raise UnicodeDecodeError
    finally:
        remove_worktree(wt)
    assert CAND_FILE in patch
    assert "bad bytes" in patch
