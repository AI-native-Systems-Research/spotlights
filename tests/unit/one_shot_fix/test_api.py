"""Orchestration: worktree-scoped validation, batch semantics, cleanup."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.one_shot_fix.api import (
    OneShotFixInput,
    one_shot_fix,
)
from spotlights_engine.one_shot_fix.claude_exec import FixRunResult
from spotlights_engine.one_shot_fix.errors import NotAGitRepoError
from spotlights_engine.prep_evolve.errors import SelectionError, StalenessError
from tests.unit.prep_evolve._fixtures import (
    CAND_FILE,
    make_repo,
    write_index,
    write_result,
    write_sorted,
)

CAND_ID = "cand-v1_attention-0002"


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
    assert "# agent edit" in (out_dir / "fix.patch").read_text(encoding="utf-8")
    assert "did the thing" in (out_dir / "FIX-NOTES.md").read_text(encoding="utf-8")
    assert fix.patch_produced is True


def test_added_file_appears_in_the_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_fix(
        OneShotFixInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/table.py"),
    )
    patch = (Path(result.fixes[0].path) / "fix.patch").read_text(encoding="utf-8")
    assert "pkg/attn/table.py" in patch


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
    with pytest.raises(AgentProposalsSetupError):
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
