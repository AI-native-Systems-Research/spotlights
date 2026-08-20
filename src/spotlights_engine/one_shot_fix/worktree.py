"""Throwaway git worktrees and patch collection.

The design puts this plumbing in Python rather than in the skill's markdown
for one concrete reason: `collect_patch` must run `git add -N` before
`git diff`, or a file the agent *added* is invisible to the diff and silently
dropped from the patch. A checklist in prose forgets that; a unit test does
not.

Worktrees are always created with `--detach`. Without it git creates a branch
named after the worktree path's basename *in the target repo*, which violates
the repo-untouched requirement.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError, WorktreeError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME

_GIT_TIMEOUT_S = 120


@dataclass
class Worktree:
    """A detached, throwaway checkout of `repo` at `base_sha`.

    `parent` is the temp directory holding `path`; it is removed alongside the
    worktree so no empty scaffolding is left behind.
    """

    path: Path
    base_sha: str
    repo: Path
    parent: Path


def _run_git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorktreeError(f"git {' '.join(args)} failed in {cwd}: {exc}") from exc
    if check and completed.returncode != 0:
        raise WorktreeError(
            f"git {' '.join(args)} failed in {cwd} "
            f"(exit {completed.returncode}): {completed.stderr.strip()}"
        )
    return completed


def require_git_repo(repo: Path) -> str:
    """Return `repo`'s HEAD sha, or raise `NotAGitRepoError`.

    `prep-evolve` tolerates a non-git target and records a `None` commit. `fix`
    cannot: a worktree needs a commit, and the emitted patch is meaningless
    without a recorded base to apply it to.
    """
    completed = _run_git(repo, "rev-parse", "HEAD", check=False)
    sha = completed.stdout.strip()
    if completed.returncode != 0 or not sha:
        raise NotAGitRepoError(
            f"{repo} is not a git checkout (git rev-parse HEAD failed). "
            f"`fix` needs a real git repo: it creates a detached worktree at "
            f"the base commit and records that commit in the patch notes."
        )
    return sha


def create_worktree(repo: Path, base_sha: str) -> Worktree:
    """Create a detached worktree of `repo` at `base_sha` under a temp dir.

    `git worktree add` requires a non-existent path, so the temp directory is
    created first and the worktree goes in a child of it.
    """
    parent = Path(tempfile.mkdtemp(prefix="spotlights-fix-"))
    dest = parent / "worktree"
    try:
        _run_git(repo, "worktree", "add", "--detach", str(dest), base_sha)
    except WorktreeError:
        shutil.rmtree(parent, ignore_errors=True)
        raise
    return Worktree(path=dest, base_sha=base_sha, repo=repo, parent=parent)


def remove_worktree(wt: Worktree) -> None:
    """Remove the worktree and prune its metadata. Idempotent; never raises.

    This runs in a `finally`, including after a timeout or a crash. Without the
    prune, a failed run leaves a stale entry in `.git/worktrees` — and a sweep
    over N candidates leaves N of them.
    """
    _run_git(wt.repo, "worktree", "remove", "--force", str(wt.path), check=False)
    _run_git(wt.repo, "worktree", "prune", check=False)
    shutil.rmtree(wt.parent, ignore_errors=True)


def collect_patch(wt: Worktree) -> tuple[str, str | None]:
    """Return `(patch_text, change_summary)` from the worktree's dirty state.

    Order matters:
    1. read + delete `CHANGE-SUMMARY.md`, so the agent's rationale reaches
       FIX-NOTES.md but never appears in the patch;
    2. `git add -N .` so *added* files show up in the diff;
    3. `git diff` against the base commit already checked out.
    """
    summary_path = wt.path / CHANGE_SUMMARY_NAME
    summary: str | None = None
    if summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8", errors="replace")
        summary_path.unlink()

    _run_git(wt.path, "add", "-N", ".")
    patch = _run_git(wt.path, "diff").stdout
    return patch, summary


__all__ = [
    "Worktree",
    "collect_patch",
    "create_worktree",
    "remove_worktree",
    "require_git_repo",
]
