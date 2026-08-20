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
