"""Orchestration: worktree-scoped validation, batch semantics, cleanup."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from spotlights_engine.one_shot_fix import worktree as worktree_mod
from spotlights_engine.one_shot_fix.api import (
    OneShotFixInput,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.claude_exec import FixRunResult
from spotlights_engine.one_shot_fix.errors import (
    ArtifactWriteError,
    ClaudeUnavailableError,
    NotAGitRepoError,
    OneShotFixError,
    WorktreeError,
)
from spotlights_engine.prep_evolve.errors import SelectionError, StalenessError
from tests.unit.prep_evolve._fixtures import (
    CAND_FILE,
    make_repo,
    make_result_dict,
    write_index,
    write_result,
    write_sorted,
)

CAND_ID = "cand-v1_attention-0002"
SECOND_CAND_ID = "cand-v1_attention-0003"


def _git_only_path(tmp_path: Path) -> str:
    """A PATH containing only a `git` shim — no `claude`, however the real PATH is set up."""
    bin_dir = tmp_path / "git-only-bin"
    bin_dir.mkdir()
    git_real = shutil.which("git")
    assert git_real is not None
    (bin_dir / "git").symlink_to(git_real)
    return str(bin_dir)


def _worktrees(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ln for ln in out.splitlines() if ln.startswith("worktree ")]


def _runner(*, edit: str | None = "# agent edit\n", summary: str | None = "did the thing",
            error: str | None = None, new_file: str | None = None):
    """A fake claude_runner that edits the worktree it is handed."""

    def _run(*, candidate_id: str, prompt: str, worktree: Path, max_turns: int,
             wallclock_s: int) -> FixRunResult:
        if edit is not None:
            target = worktree / CAND_FILE
            target.write_text(target.read_text(encoding="utf-8") + edit, encoding="utf-8")
        if new_file is not None:
            (worktree / new_file).write_text("NEW = 1\n", encoding="utf-8")
        if summary is not None:
            (worktree / "CHANGE-SUMMARY.md").write_text(summary, encoding="utf-8")
        return FixRunResult(candidate_id=candidate_id, duration_s=0.01, error=error)

    return _run


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    """(run_dir, repo) with result.json + index.md + a committed repo."""
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path)
    write_index(run_dir, repo)
    return run_dir, repo


def test_single_candidate_writes_patch_and_notes(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    fix = result.fixes[0]
    assert fix.candidate_id == CAND_ID
    out_dir = Path(fix.path)
    assert out_dir == run_dir / "fix" / "v1_attention" / CAND_ID
    assert sorted(fix.files) == ["FIX-NOTES.md", "fix.patch"]
    patch_text = (out_dir / "fix.patch").read_text(encoding="utf-8")
    assert "# agent edit" in patch_text
    assert "did the thing" in (out_dir / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert fix.patch_produced is True

    # The header's apply recipe must use the $PWD-anchored patch path, not one
    # of the two broken forms that regressed twice (bare `git apply fix.patch`,
    # or `git -C {repo} apply fix.patch` with a relative, cwd-dependent path).
    # `repo_path` in the header is resolve()d by resolve_repo_path, so compare
    # against the resolved form here too.
    repo_resolved = repo.resolve()
    assert f'git -C {repo_resolved} apply "$PWD/fix.patch"' in patch_text
    assert "git apply fix.patch" not in patch_text
    assert f"git -C {repo_resolved} apply fix.patch" not in patch_text


def test_added_file_appears_in_the_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/table.py"),
    )
    patch = (Path(result.fixes[0].path) / "fix.patch").read_text(encoding="utf-8")
    assert "pkg/attn/table.py" in patch


def test_a_tab_in_a_created_filename_does_not_abort_the_run(run) -> None:
    """Critical 1, end to end: a tab in a filename the agent created used to

    raise an uncaught `ValueError` out of `collect_patch` — which runs inside
    the `try` whose `finally` removes the worktree — so the whole session's
    work (patch and notes) was lost before `_write_artifacts` ever ran, and
    in a sweep, the bare `ValueError` would escape `api.py`'s
    `(PrepEvolveError, OneShotFixError)` handler entirely and abort the rest
    of the candidates too. The run must complete, the patch must be written,
    and the odd file must be represented in the notes.
    """
    run_dir, repo = run
    odd_name = "tab\tname.py"
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file=odd_name),
    )
    assert len(result.fixes) == 1
    fix = result.fixes[0]
    assert fix.patch_produced is True
    out_dir = Path(fix.path)
    patch = (out_dir / "fix.patch").read_text(encoding="utf-8")
    assert "tab\\tname.py" in patch  # C-quoted in the plain patch text
    notes = (out_dir / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert odd_name in notes  # the manifest row carries the real path


def test_out_of_scope_edit_is_flagged_on_the_artifact_and_in_the_notes(run) -> None:
    """The manifest is derived from the diff, not from spec.targets — this is

    the whole point: an edit outside the declared scope must still land in
    the patch (never silently dropped) AND be visible both on `FixArtifact`
    (for the CLI) and prominently in FIX-NOTES.md (for the reviewer).
    """
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/extra.py"),
    )
    fix = result.fixes[0]
    assert fix.out_of_scope_files == ["pkg/attn/extra.py"]

    patch = (Path(fix.path) / "fix.patch").read_text(encoding="utf-8")
    assert "pkg/attn/extra.py" in patch  # not amputated from the patch

    notes = (Path(fix.path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "pkg/attn/extra.py" in notes
    assert "**This patch touches files outside the declared scope.**" in notes


def test_in_scope_only_edit_has_no_out_of_scope_files(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    fix = result.fixes[0]
    assert fix.out_of_scope_files == []
    notes = (Path(fix.path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "outside the declared scope" not in notes


def test_no_patch_produced_has_no_out_of_scope_files(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    fix = result.fixes[0]
    assert fix.patch_produced is False
    assert fix.out_of_scope_files == []


def test_validation_runs_against_the_worktree_not_a_dirty_repo(run) -> None:
    """A dirty --repo must not affect the outcome: the gate reads worktree bytes."""
    run_dir, repo = run
    # Break the candidate's symbol in the *working tree* only; HEAD is still good.
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    assert result.skipped == []


def test_staleness_at_head_raises_for_an_explicit_candidate(run) -> None:
    run_dir, repo = run
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "gut"],
        cwd=repo,
        check=True,
    )

    with pytest.raises(StalenessError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )


def test_staleness_in_a_batch_is_skipped_with_a_reason(run) -> None:
    run_dir, repo = run
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t.t", "-c", "user.name=t", "commit", "-qm", "gut"],
        cwd=repo,
        check=True,
    )

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo)),  # no --candidate → batch
        claude_runner=_runner(),
    )

    assert result.fixes == []
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "stale" in result.skipped[0].reason.lower()


def test_worktree_is_removed_on_the_success_path(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    assert _worktrees(repo) == before


def test_worktree_is_removed_when_the_agent_fails(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary=None, error="claude exit=3"),
    )
    assert _worktrees(repo) == before
    notes = (Path(result.fixes[0].path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "claude exit=3" in notes
    assert result.fixes[0].patch_produced is False


def test_worktree_is_removed_when_the_runner_raises(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)

    def _boom(**kwargs):
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_boom,
        )
    assert _worktrees(repo) == before


def test_no_edit_produces_notes_but_no_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    fix = result.fixes[0]
    assert fix.patch_produced is False
    assert fix.files == ["FIX-NOTES.md"]
    assert not (Path(fix.path) / "fix.patch").exists()


def test_a_rerun_that_produces_no_patch_removes_the_stale_patch(run) -> None:
    """A previous session's fix.patch must not survive a rerun that concludes no fix."""
    run_dir, repo = run
    first = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    out_dir = Path(first.fixes[0].path)
    assert (out_dir / "fix.patch").exists()

    second = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    fix = second.fixes[0]
    assert fix.patch_produced is False
    assert fix.files == ["FIX-NOTES.md"]
    assert not (out_dir / "fix.patch").exists()


def test_non_git_repo_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)

    with pytest.raises(NotAGitRepoError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )


def test_print_prompt_leaves_the_worktree_and_runs_no_agent(run) -> None:
    run_dir, repo = run
    called: list[str] = []

    def _spy(**kwargs):
        called.append("ran")
        raise AssertionError("claude must not run under --print-prompt")

    result = one_shot_fix(
        OneShotFixInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_spy,
    )

    assert called == []
    assert result.fixes == []
    assert len(result.prompts) == 1
    preview = result.prompts[0]
    worktree = Path(preview.worktree)
    assert worktree.is_dir()
    assert (worktree / CAND_FILE).is_file()
    assert CAND_FILE in preview.prompt
    assert len(_worktrees(repo)) == 2  # main tree + the one left for the caller
    assert Path(preview.worktree_parent) == worktree.parent

    # Not our job to clean up under --print-prompt, but don't leak in the test.
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=True
    )
    # `worktree_parent` exists precisely so a --print-prompt consumer can clean
    # up the temp scaffolding; use it here too, or the test leaks an empty
    # spotlights-fix-XXXXXXXX/ under the system temp dir on every run.
    shutil.rmtree(preview.worktree_parent, ignore_errors=True)


def test_print_prompt_without_a_candidate_raises_and_leaves_no_worktree(run) -> None:
    """A sweep under --print-prompt would leave one worktree per candidate; refuse it."""
    run_dir, repo = run
    before = _worktrees(repo)

    with pytest.raises(SelectionError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), print_prompt=True),
            claude_runner=_runner(),
        )

    assert _worktrees(repo) == before


def test_missing_claude_binary_raises_before_the_loop_with_the_real_runner(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))
    with pytest.raises(ClaudeUnavailableError):
        one_shot_fix(OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID))

    # This is the property the (not-yet-written) CLI depends on: it catches
    # `(PrepEvolveError, OneShotFixError)` once and maps both to a clean
    # stderr message. A bare RuntimeError subclass (AgentProposalsSetupError)
    # would slip past that catch and surface as a traceback.
    try:
        one_shot_fix(OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID))
    except OneShotFixError:
        pass
    else:
        pytest.fail("expected a OneShotFixError to be raised")


def test_missing_claude_is_masked_by_a_non_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The git check is cheaper and more fundamental; it must win.

    Task 7's planned CLI test asserts a non-git repo yields exit 2 with "git"
    in stderr, and passes no fake runner — so on any machine or CI runner
    lacking `claude`, the claude preflight must not fire before the git
    check, or that test would fail on the wrong error.
    """
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))

    with pytest.raises(NotAGitRepoError):
        one_shot_fix(OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID))


def test_print_prompt_does_not_require_claude_on_path(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True)
    )

    assert len(result.prompts) == 1
    subprocess.run(
        ["git", "worktree", "remove", "--force", result.prompts[0].worktree],
        cwd=repo,
        check=True,
    )
    shutil.rmtree(result.prompts[0].worktree_parent, ignore_errors=True)


def test_injected_runner_does_not_require_claude_on_path(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1


def test_top_n_over_a_ranking_selects_the_ranked_prefix(run) -> None:
    run_dir, repo = run
    sorted_dir = write_sorted(run_dir, [CAND_ID])

    result = one_shot_fix(
        OneShotFixInput(result=sorted_dir, repo=str(repo), top_n=1),
        claude_runner=_runner(),
    )

    assert len(result.fixes) == 1
    # Artifacts land under the run dir, not inside sorted/.
    assert Path(result.fixes[0].path) == run_dir / "fix" / "v1_attention" / CAND_ID


def test_out_overrides_the_artifact_base(run, tmp_path: Path) -> None:
    run_dir, repo = run
    out = tmp_path / "elsewhere"
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID, out=out),
        claude_runner=_runner(),
    )
    assert Path(result.fixes[0].path) == out / "fix" / "v1_attention" / CAND_ID


def test_an_unwritable_out_dir_skips_one_candidate_instead_of_aborting_the_sweep(
    run, tmp_path: Path
) -> None:
    """Artifact IO failures must be `OneShotFixError`s, or a sweep dies mid-way.

    `_write_artifacts` runs `mkdir` and two writes. A bare `OSError` from any
    of them propagates straight past the batch loop's
    `except (PrepEvolveError, OneShotFixError)` handler and out of
    `one_shot_fix` — so one unwritable path (full disk, read-only mount,
    a name collision) silently discards every candidate still queued, after
    their agent sessions were already paid for. Wrapping it in
    `ArtifactWriteError` turns that into one recorded skip.

    A regular file where the artifact tree needs a directory reproduces the
    failure portably: `mkdir(parents=True)` raises `NotADirectoryError`
    (an `OSError` subclass) without needing root or a special filesystem.
    """
    run_dir, repo = run
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n", encoding="utf-8")

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), out=blocked),  # batch: no --candidate
        claude_runner=_runner(),
    )

    assert result.fixes == []
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "could not write fix artifacts" in result.skipped[0].reason
    # The worktree is still gone: cleanup happens before artifacts are written.
    assert len(_worktrees(repo)) == 1


def test_an_unwritable_out_dir_raises_for_an_explicit_candidate(run, tmp_path: Path) -> None:
    """Same failure, single-candidate mode: raise, and as a `OneShotFixError`.

    The CLI catches exactly `(PrepEvolveError, OneShotFixError)` and turns it
    into `fix: <message>` with exit 2. A bare `OSError` would instead reach the
    user as a traceback.
    """
    run_dir, repo = run
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ArtifactWriteError) as excinfo:
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID, out=blocked),
            claude_runner=_runner(),
        )
    assert isinstance(excinfo.value, OneShotFixError)
    assert len(_worktrees(repo)) == 1


def test_a_patch_write_that_fails_partway_leaves_no_truncated_patch_behind(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_bytes` is not atomic, and a half-written `fix.patch` is worse than none.

    An `ENOSPC` (or a write error on a network mount) partway through a
    multi-megabyte patch leaves a *prefix* of the diff on disk under the exact
    name the docs tell a reviewer to apply. Raising over it and leaving the file
    there is the worst outcome available: `git apply` either rejects it, or — at
    an unlucky hunk boundary — applies part of it. Removing it makes the failure
    read as "no patch here", which is true, and matches the empty-patch path
    that already unlinks a stale `fix.patch`.
    """
    run_dir, repo = run
    real_write_bytes = Path.write_bytes

    def _short_write(self: Path, data: bytes) -> int:
        if self.name != "fix.patch":
            return real_write_bytes(self, data)
        real_write_bytes(self, data[: len(data) // 2])  # the truncated prefix
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", _short_write)

    with pytest.raises(ArtifactWriteError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )

    out_dir = run_dir / "fix" / "v1_attention" / CAND_ID
    assert not (out_dir / "fix.patch").exists()


def test_a_notes_write_that_fails_partway_leaves_neither_artifact_behind(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror case: the patch landed, then the notes write died.

    `FIX-NOTES.md` is human-facing, so a truncated one cannot be misapplied the
    way a truncated patch can — but it can lose the "Applying and verifying"
    section that records *which commit* the patch belongs to, and a patch whose
    base is unknown is not applicable. The directory must not be left holding
    a `fix.patch` documented by half a page either, so the handler removes
    both: the same "nothing here" invariant the empty-patch branch maintains.
    """
    run_dir, repo = run
    real_write_text = Path.write_text

    def _short_write(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if self.name != "FIX-NOTES.md":
            return real_write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]
        real_write_text(self, data[: len(data) // 2], encoding="utf-8")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", _short_write)

    with pytest.raises(ArtifactWriteError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )

    out_dir = run_dir / "fix" / "v1_attention" / CAND_ID
    assert not (out_dir / "FIX-NOTES.md").exists()
    # And the patch that *did* write successfully goes with it — a patch with
    # no notes has no recorded base commit.
    assert not (out_dir / "fix.patch").exists()


def test_a_keyboard_interrupt_during_the_agent_session_leaks_no_worktree(run) -> None:
    """`KeyboardInterrupt` is a `BaseException`, and Ctrl-C is how a real sweep ends.

    A long `fix` sweep is interrupted by hand far more often than it fails, so
    the cleanup has to sit in a `finally` — not in an `except Exception`, which
    `KeyboardInterrupt` walks straight past — or each interrupted candidate
    leaves both a stale `.git/worktrees` entry in the user's own repo and a
    temp directory holding a full checkout.

    This covers the interrupt-during-the-session half. The other half —
    `create_worktree` being called *inside* the `try`, so no window exists
    where the worktree is created but its `finally` is not yet registered — is
    not reachable from a test (it is a handful of bytecodes wide); it is
    enforced by the structure of `_process_candidate` and its comment.
    """
    run_dir, repo = run
    before = _worktrees(repo)

    def _interrupt(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_interrupt,
        )

    assert _worktrees(repo) == before
    # `git worktree list` can lag a manual rmtree; the prune in `remove_worktree`
    # is what keeps `.git/worktrees` itself clean, so assert on that directly.
    admin = repo / ".git" / "worktrees"
    assert not admin.exists() or list(admin.iterdir()) == []


def _break_the_diff(monkeypatch: pytest.MonkeyPatch, message: str) -> None:
    """Make `collect_patch`'s `git diff` fail the way a real timeout would."""
    real = worktree_mod._run_git_bytes

    def _fake(cwd: Path, *args: str, check: bool = True) -> bytes:
        if "diff" in args:
            raise WorktreeError(message)
        return real(cwd, *args, check=check)

    monkeypatch.setattr(worktree_mod, "_run_git_bytes", _fake)


def test_a_failed_diff_still_writes_notes_and_does_not_claim_no_patch(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `git diff` that fails is a lost session, not "the agent made no edit".

    `collect_patch` runs as the last statement inside the `try` whose `finally`
    destroys the worktree, and `_write_artifacts` is *after* that block. If the
    diff raised, the whole candidate record went with the worktree: no
    `FIX-NOTES.md`, no `run.error`, no usage — and in a sweep, a skip whose
    reason is a git message with no hint that a paid session was thrown away.

    Reporting it as `patch=b""` alone would be worse than nothing: the notes
    would then state "No patch was produced. The agent made no in-scope edit."
    about a session that may well have edited every file in scope. Hence the
    separate `collect_error` channel.
    """
    run_dir, repo = run
    _break_the_diff(monkeypatch, "git diff timed out after 60s")

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(summary="rewrote the tile heuristic"),
    )

    assert result.skipped == []
    fix = result.fixes[0]
    assert fix.collection_error is not None
    assert "timed out" in fix.collection_error
    assert fix.patch_produced is False
    assert fix.files == ["FIX-NOTES.md"]

    notes = (Path(fix.path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "No patch could be collected" in notes
    assert "git diff timed out after 60s" in notes
    # The false claim this whole channel exists to prevent.
    assert "The agent made no in-scope edit" not in notes
    # The agent's own summary is still recorded — flagged as a claim about work
    # that was never captured, but not discarded.
    assert "rewrote the tile heuristic" in notes
    # And the worktree is still cleaned up.
    assert len(_worktrees(repo)) == 1


def test_a_failed_diff_keeps_the_agent_session_error(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both failures are recorded; the diff failure does not swallow the first."""
    run_dir, repo = run
    _break_the_diff(monkeypatch, "git diff timed out after 60s")

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(error="claude exit=1: stderr=b'overloaded'"),
    )

    notes = (Path(result.fixes[0].path) / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert "overloaded" in notes
    assert "git diff timed out after 60s" in notes


def test_a_failed_diff_leaves_no_stale_patch_from_a_previous_run(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notes say the patch is unknown, so a previous session's must not sit there."""
    run_dir, repo = run
    first = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    out_dir = Path(first.fixes[0].path)
    assert (out_dir / "fix.patch").exists()

    _break_the_diff(monkeypatch, "git diff timed out after 60s")
    one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    assert not (out_dir / "fix.patch").exists()


def _write_two_candidates(run_dir: Path) -> None:
    """Rewrite result.json with a second candidate, cloned from the first."""
    data = make_result_dict()
    cands = data["module_runs"]["v1/attention"]["candidates"]["candidates"]
    second = json.loads(json.dumps(cands[0]))
    second["id"] = SECOND_CAND_ID
    cands.append(second)
    (run_dir / "result.json").write_text(json.dumps(data), encoding="utf-8")


def test_an_unexpected_exception_in_a_sweep_skips_one_candidate_and_continues(
    run,
) -> None:
    """Each candidate in a sweep costs a paid agent session; one bug must not burn the rest.

    The batch loop caught only `(PrepEvolveError, OneShotFixError)`. Any other
    exception type — a pydantic `ValidationError`, an `OSError` from somewhere
    not already wrapped, a bug in this module — propagated out of
    `one_shot_fix`, discarding every candidate still queued. Unlike
    `prep_evolve`, whose equally narrow loop is free to re-run, that throws
    away sessions that were never started.
    """
    run_dir, repo = run
    _write_two_candidates(run_dir)

    def _boom_on_the_first(*, candidate_id: str, prompt: str, worktree: Path,
                           max_turns: int, wallclock_s: int) -> FixRunResult:
        if candidate_id == CAND_ID:
            raise RuntimeError("a bug nobody anticipated")
        return _runner()(
            candidate_id=candidate_id,
            prompt=prompt,
            worktree=worktree,
            max_turns=max_turns,
            wallclock_s=wallclock_s,
        )

    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo)),  # batch: no --candidate
        claude_runner=_boom_on_the_first,
    )

    # The second candidate still ran.
    assert [f.candidate_id for f in result.fixes] == [SECOND_CAND_ID]
    assert result.fixes[0].patch_produced is True

    assert len(result.skipped) == 1
    skip = result.skipped[0]
    assert skip.candidate_id == CAND_ID
    # Worded so it cannot be mistaken for a candidate that legitimately could
    # not be fixed.
    assert "unexpected RuntimeError" in skip.reason
    assert "bug" in skip.reason
    assert "a bug nobody anticipated" in skip.reason

    # Neither candidate leaked a worktree.
    assert len(_worktrees(repo)) == 1


def test_a_keyboard_interrupt_in_a_sweep_still_aborts_the_sweep(run) -> None:
    """The widened handler is `except Exception`, never `BaseException`.

    Ctrl-C through a ten-candidate sweep must stop it, not be recorded as ten
    skips while the sweep keeps paying for sessions.
    """
    run_dir, repo = run
    _write_two_candidates(run_dir)

    def _interrupt(**kwargs):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo)),  # batch
            claude_runner=_interrupt,
        )


def test_an_unexpected_exception_still_raises_for_an_explicit_candidate(run) -> None:
    """Single-candidate mode keeps the traceback: there is nothing to protect."""
    run_dir, repo = run

    def _boom(**kwargs):
        raise RuntimeError("a bug nobody anticipated")

    with pytest.raises(RuntimeError):
        one_shot_fix(
            OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_boom,
        )
