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
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "# appended" in collection.patch
    assert CAND_FILE in collection.patch
    assert collection.change_summary is None


def test_collect_patch_includes_an_added_file(tmp_path: Path) -> None:
    """The `git add -N` regression: a new file must not be silently dropped."""
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / "pkg" / "attn" / "table.py").write_text(
            "TILE_TABLE = {128: 64}\n", encoding="utf-8"
        )
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "pkg/attn/table.py" in collection.patch
    assert "TILE_TABLE" in collection.patch


def test_collect_patch_extracts_and_excludes_the_change_summary(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / CHANGE_SUMMARY_NAME).write_text(
            "Swapped the heuristic for a table.\n", encoding="utf-8"
        )
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert collection.change_summary == "Swapped the heuristic for a table.\n"
    assert CHANGE_SUMMARY_NAME not in collection.patch
    assert all(c.path != CHANGE_SUMMARY_NAME for c in collection.manifest)


def test_collect_patch_is_empty_when_nothing_changed(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert collection.patch == ""
    assert collection.change_summary is None
    assert collection.manifest == []


def test_manifest_records_a_modified_file_with_line_counts(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text() + "# appended\n", encoding="utf-8")
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == CAND_FILE
    assert change.change_kind == "modified"
    assert change.insertions == 1
    assert change.deletions == 0
    assert change.binary is False
    assert change.old_path is None


def test_manifest_records_an_added_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / "pkg" / "attn" / "table.py").write_text(
            "TILE_TABLE = {128: 64}\n", encoding="utf-8"
        )
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == "pkg/attn/table.py"
    assert change.change_kind == "added"
    assert change.insertions == 1
    assert change.deletions == 0
    assert change.binary is False


def test_manifest_records_a_deleted_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / CAND_FILE).unlink()
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == CAND_FILE
    assert change.change_kind == "deleted"
    assert change.binary is False
    assert change.deletions is not None and change.deletions > 0
    # Important 5: the manifest can say "deleted" while the patch itself
    # regresses (e.g. back to a bare `git diff`, which silently drops staged
    # deletions) and every test still stays green. Assert the patch text too.
    assert "deleted file mode" in collection.patch
    assert CAND_FILE in collection.patch


def test_manifest_records_a_binary_file_with_no_line_counts(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        (wt.path / "pkg" / "attn" / "blob.bin").write_bytes(b"\x00\x01\x02binary\x00data")
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == "pkg/attn/blob.bin"
    assert change.change_kind == "added"
    assert change.binary is True
    assert change.insertions is None
    assert change.deletions is None


def test_manifest_records_a_renamed_file_with_the_new_path(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        new_path = wt.path / "pkg" / "attn" / "tile_renamed.py"
        target.rename(new_path)
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.change_kind == "renamed"
    assert change.path == "pkg/attn/tile_renamed.py"
    assert change.old_path == CAND_FILE
    assert change.binary is False
    # Important 5: the delete half of a rename must also be visible in the
    # patch text itself, not only in the manifest's structured fields.
    assert f"rename from {CAND_FILE}" in collection.patch
    assert "rename to pkg/attn/tile_renamed.py" in collection.patch


def test_manifest_handles_a_path_with_spaces_without_truncation(tmp_path: Path) -> None:
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        odd_dir = wt.path / "pkg" / "dir with space"
        odd_dir.mkdir(parents=True)
        (odd_dir / "file name.py").write_text("X = 1\n", encoding="utf-8")
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == "pkg/dir with space/file name.py"
    assert change.change_kind == "added"


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


def test_collect_patch_survives_a_tab_in_a_filename(tmp_path: Path) -> None:
    """Critical 1: `-z` disables C-quoting, so a literal tab in a filename

    makes `git diff --numstat -z` emit a *4*-field record
    (`added\\tdeleted\\ttab\\tname.py\\0`). The naive `split("\\t")` used to
    raise `ValueError: too many values to unpack`, which — uncaught by
    `api.py`'s `(PrepEvolveError, OneShotFixError)` handler — aborted the
    entire sweep and (because `collect_patch` runs inside the `try` whose
    `finally` removes the worktree) discarded the whole session's work before
    `_write_artifacts` ever ran. Tabs in filenames are legal on POSIX.

    This must not raise, the patch must still be collected, and the odd file
    must be represented (correctly) in the manifest.
    """
    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        odd_name = "tab\tname.py"
        (wt.path / odd_name).write_text("X = 1\n", encoding="utf-8")
        collection = collect_patch(wt)  # must not raise
    finally:
        remove_worktree(wt)
    # The plain (non `-z`) patch text C-quotes special characters, so the
    # literal tab shows up escaped (`\t`) inside a quoted path — that's git's
    # own standard patch format, not a bug. The manifest (built from the `-z`
    # forms, which disable quoting) is where the real, unescaped path lives.
    assert "tab\\tname.py" in collection.patch
    assert "X = 1" in collection.patch
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == odd_name
    assert change.change_kind == "added"
    assert change.binary is False
    assert change.counts_known is True
    assert change.insertions == 1
    assert change.deletions == 0


def test_collect_patch_diffs_against_base_sha_not_head(tmp_path: Path) -> None:
    """Important 2: the agent may run `git reset` (deliberately allowed by

    `claude_exec.py` on a detached worktree — see its module docstring). A
    `git reset --soft` moves HEAD without touching the working tree, so
    diffing against `HEAD` after such a reset silently changes the diff's
    base out from under the caller. Diffing against `wt.base_sha` (the
    commit the worktree was actually created at, and the same value recorded
    in `fix.patch`'s header and `FIX-NOTES.md`) is immune to this.

    Reproduced directly against real git: base_sha = c2 ("v2"); the agent
    edits the file to "v3-agent" (uncommitted); the agent runs
    `git reset --soft HEAD~1`, which moves HEAD to c1 without touching the
    working tree or index. `git diff HEAD` then wrongly shows a diff from c1
    ("v1"); `git diff base_sha` correctly shows a diff from c2 ("v2").
    """
    repo = make_repo(tmp_path)  # c1
    # Advance the repo by a second commit (c2) so base_sha has a parent for
    # `HEAD~1` to land on, mirroring the reviewer's c1/c2/c3 scenario.
    (repo / CAND_FILE).write_text(
        (repo / CAND_FILE).read_text(encoding="utf-8") + "# c2 marker\n", encoding="utf-8"
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "c2"],
        cwd=repo,
        check=True,
    )
    base_sha = require_git_repo(repo)  # c2
    wt = create_worktree(repo, base_sha)
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text(encoding="utf-8") + "# v3-agent\n", encoding="utf-8")
        # Moves HEAD back one commit without touching the working tree/index —
        # exactly what `git reset --soft` does, and what `claude_exec.py`
        # deliberately permits the agent to run.
        _git(wt.path, "reset", "--soft", "HEAD~1")
        collection = collect_patch(wt)
    finally:
        remove_worktree(wt)
    assert "+# v3-agent" in collection.patch
    # If the diff were (still, wrongly) HEAD-relative, HEAD now points at c1,
    # so the "# c2 marker" line added in c2 would show up as *newly added*
    # too (a "+" line). Diffing against base_sha (c2) must show it only as
    # unchanged context (or not at all), never as an addition, since it was
    # already present at c2.
    assert "+# c2 marker" not in collection.patch


def test_manifest_survives_a_malformed_numstat_record_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hardening: an unexpected numstat token (neither a digit string nor

    `-`) must degrade that record to counts-unknown, never raise — the same
    crash class as Critical 1's tab bug, guarded against directly so a future
    git output surprise can't reintroduce it.
    """
    import spotlights_engine.one_shot_fix.worktree as worktree_mod

    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text() + "# appended\n", encoding="utf-8")

        real_run_git = worktree_mod._run_git

        def _corrupt_numstat(cwd, *args, **kwargs):
            completed = real_run_git(cwd, *args, **kwargs)
            if "--numstat" in args:
                completed.stdout = completed.stdout.replace("1\t0\t", "??\t0\t", 1)
            return completed

        monkeypatch.setattr(worktree_mod, "_run_git", _corrupt_numstat)
        collection = collect_patch(wt)  # must not raise
    finally:
        remove_worktree(wt)
    assert len(collection.manifest) == 1
    change = collection.manifest[0]
    assert change.path == CAND_FILE
    assert change.counts_known is False
    assert change.binary is False
    assert change.insertions is None
    assert change.deletions is None
    # The patch text itself is unaffected by the manifest-side corruption.
    assert "# appended" in collection.patch


def test_collect_patch_preserves_the_patch_when_manifest_building_blows_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Critical 1's core promise, exercised directly: if manifest parsing

    raises for a reason nobody anticipated, the patch text (already
    collected) must still come back rather than the whole collection failing
    and losing the agent's work.
    """
    import spotlights_engine.one_shot_fix.worktree as worktree_mod

    repo = make_repo(tmp_path)
    wt = create_worktree(repo, require_git_repo(repo))
    try:
        target = wt.path / CAND_FILE
        target.write_text(target.read_text() + "# appended\n", encoding="utf-8")

        def _boom(*args, **kwargs):
            raise ValueError("simulated unforeseen parse failure")

        monkeypatch.setattr(worktree_mod, "_build_manifest", _boom)
        collection = collect_patch(wt)  # must not raise
    finally:
        remove_worktree(wt)
    assert "# appended" in collection.patch
    assert CAND_FILE in collection.patch


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
        collection = collect_patch(wt)  # must not raise UnicodeDecodeError
    finally:
        remove_worktree(wt)
    assert CAND_FILE in collection.patch
    assert "bad bytes" in collection.patch
