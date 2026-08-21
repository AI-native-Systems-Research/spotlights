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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from spotlights_engine.one_shot_fix.errors import NotAGitRepoError, WorktreeError
from spotlights_engine.one_shot_fix.prompts import CHANGE_SUMMARY_NAME

_GIT_TIMEOUT_S = 120

ChangeKind = Literal["added", "modified", "deleted", "renamed"]


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
            errors="replace",
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

    `check=False` only suppresses `_run_git`'s return-code check; it does not
    stop `_run_git` from raising `WorktreeError` when the underlying
    `subprocess.run` itself blows up (`OSError`, or a `SubprocessError` such as
    a timeout). Each git call is therefore wrapped individually so no such
    exception escapes this function.
    """
    try:
        _run_git(wt.repo, "worktree", "remove", "--force", str(wt.path), check=False)
    except WorktreeError:
        pass
    try:
        _run_git(wt.repo, "worktree", "prune", check=False)
    except WorktreeError:
        pass
    shutil.rmtree(wt.parent, ignore_errors=True)


@dataclass(frozen=True)
class FileChange:
    """One file's change, derived from the actual diff — never from a declared scope.

    `insertions`/`deletions` are `None` for binary files: `git diff --numstat`
    reports `-` for both rather than a count. `old_path` is set only when
    `change_kind == "renamed"`.
    """

    path: str
    change_kind: ChangeKind
    insertions: int | None
    deletions: int | None
    binary: bool
    old_path: str | None = None


@dataclass(frozen=True)
class PatchCollection:
    """Result of `collect_patch`: the raw diff, the agent's rationale, and a manifest.

    `manifest` is the per-file breakdown of exactly what `patch` contains —
    derived from `git diff --name-status`/`--numstat`, not from any declared
    scope, so it can catch a patch that disagrees with what was intended.
    """

    patch: str
    change_summary: str | None
    manifest: list[FileChange] = field(default_factory=list)


def _split_z(raw: str) -> list[str]:
    """Split `-z`-terminated git output into tokens, dropping the trailing empty one."""
    tokens = raw.split("\0")
    if tokens and tokens[-1] == "":
        tokens = tokens[:-1]
    return tokens


def _parse_name_status_z(raw: str) -> list[tuple[str, str, str | None]]:
    """Parse `git diff --name-status -z` into `(status, path, old_path)` triples.

    With `-z`, a rename/copy record is `status<NUL>old_path<NUL>new_path<NUL>`;
    every other record is `status<NUL>path<NUL>`. `-z` also disables git's
    quoting of paths with spaces or non-ASCII bytes, so no unquoting is needed.
    """
    tokens = _split_z(raw)
    records: list[tuple[str, str, str | None]] = []
    i = 0
    while i < len(tokens):
        status = tokens[i]
        if status[:1] in ("R", "C"):
            old_path, new_path = tokens[i + 1], tokens[i + 2]
            records.append((status, new_path, old_path))
            i += 3
        else:
            records.append((status, tokens[i + 1], None))
            i += 2
    return records


def _parse_numstat_z(raw: str) -> list[tuple[int | None, int | None]]:
    """Parse `git diff --numstat -z` into `(added, deleted)` pairs, in record order.

    Binary files report `-` for both counts, mapped to `None` here. A
    rename/copy record has an empty path field followed by two more
    NUL-separated path tokens (old, then new) — this only needs the counts,
    so those extra tokens are skipped; `_parse_name_status_z` is the source of
    truth for paths.
    """
    tokens = _split_z(raw)
    records: list[tuple[int | None, int | None]] = []
    i = 0
    while i < len(tokens):
        added_s, deleted_s, path = tokens[i].split("\t")
        added = None if added_s == "-" else int(added_s)
        deleted = None if deleted_s == "-" else int(deleted_s)
        records.append((added, deleted))
        i += 1 if path != "" else 3
    return records


def _change_kind(status: str) -> ChangeKind:
    code = status[:1]
    if code == "A":
        return "added"
    if code == "D":
        return "deleted"
    if code == "R":
        return "renamed"
    if code == "C":
        # A copy creates a new path; the source is untouched. There is no
        # "copied" kind in the manifest's vocabulary, so this is the closest
        # honest label.
        return "added"
    return "modified"


def _build_manifest(name_status_raw: str, numstat_raw: str) -> list[FileChange]:
    statuses = _parse_name_status_z(name_status_raw)
    counts = _parse_numstat_z(numstat_raw)
    if len(counts) != len(statuses):
        # Shouldn't happen — both come from the same worktree state, queried
        # back to back with no intervening change — but a degraded manifest
        # (paths and kinds right, counts unknown) beats crashing the whole
        # collection over a git output surprise.
        counts = [(None, None)] * len(statuses)

    manifest: list[FileChange] = []
    for (status, path, old_path), (added, deleted) in zip(statuses, counts, strict=True):
        manifest.append(
            FileChange(
                path=path,
                change_kind=_change_kind(status),
                insertions=added,
                deletions=deleted,
                binary=added is None and deleted is None,
                old_path=old_path,
            )
        )
    return manifest


def collect_patch(wt: Worktree) -> PatchCollection:
    """Return the patch, the agent's rationale, and a per-file manifest.

    Order matters:
    1. read + delete `CHANGE-SUMMARY.md`, so the agent's rationale reaches
       FIX-NOTES.md but never appears in the patch (or the manifest);
    2. `git add -N .` so *added* files show up in the diff;
    3. `git diff HEAD` (patch text) plus `git diff HEAD --name-status -z` and
       `git diff HEAD --numstat -z` (the manifest) against the base commit
       already checked out.

    `HEAD` (not a bare `git diff`, which is index-vs-worktree only) is load-
    bearing: `git add -N .`'s pathspec (`.`) does not just intent-to-add new
    paths — for a path that matches and no longer exists on disk, plain `git
    add` (this is standard `add` semantics since git 2.0, unrelated to `-N`)
    stages its deletion *fully*, not as intent-to-add. A bare `git diff`
    (index vs worktree) then finds nothing to show for that path, so a
    deleted file — and the delete-half of a rename, which is a deletion plus
    an addition of matching content — silently vanishes. `git diff HEAD`
    compares the worktree directly against the base commit regardless of
    what got staged along the way, so it sees the deletion (and the rename)
    correctly while still matching a bare `git diff` byte-for-byte for the
    modified/added cases that were already covered.
    """
    summary_path = wt.path / CHANGE_SUMMARY_NAME
    summary: str | None = None
    if summary_path.is_file():
        summary = summary_path.read_text(encoding="utf-8", errors="replace")
        summary_path.unlink()

    _run_git(wt.path, "add", "-N", ".")
    patch = _run_git(wt.path, "diff", "HEAD").stdout
    name_status_raw = _run_git(wt.path, "diff", "HEAD", "--name-status", "-z").stdout
    numstat_raw = _run_git(wt.path, "diff", "HEAD", "--numstat", "-z").stdout
    manifest = _build_manifest(name_status_raw, numstat_raw)
    return PatchCollection(patch=patch, change_summary=summary, manifest=manifest)


__all__ = [
    "FileChange",
    "PatchCollection",
    "Worktree",
    "collect_patch",
    "create_worktree",
    "remove_worktree",
    "require_git_repo",
]
