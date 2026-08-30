"""Orchestration: worktree-scoped validation, batch semantics, cleanup."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.one_shot_apply import worktree as worktree_mod
from spotlights_engine.one_shot_apply.api import (
    OneShotApplyInput,
    PromptPreview,
    one_shot_apply,
    render_prompt_block,
)
from spotlights_engine.one_shot_apply.claude_exec import ApplyRunResult
from spotlights_engine.one_shot_apply.errors import (
    ArtifactWriteError,
    ClaudeUnavailableError,
    NotAGitRepoError,
    OneShotApplyError,
    WorktreeError,
)
from spotlights_engine.one_shot_apply.usage_manifest import ApplyUsageManifest
from spotlights_engine.prep_evolve.errors import StalenessError
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
THIRD_CAND_ID = "cand-v1_attention-0004"


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
            error: str | None = None, new_file: str | None = None,
            usage: AgentUsage | None = None, duration_s: float = 0.01):
    """A fake claude_runner that edits the worktree it is handed."""

    def _run(*, candidate_id: str, prompt: str, worktree: Path, max_turns: int,
             wallclock_s: int, claude_model: str | None = None) -> ApplyRunResult:
        if edit is not None:
            target = worktree / CAND_FILE
            target.write_text(target.read_text(encoding="utf-8") + edit, encoding="utf-8")
        if new_file is not None:
            (worktree / new_file).write_text("NEW = 1\n", encoding="utf-8")
        if summary is not None:
            (worktree / "CHANGE-SUMMARY.md").write_text(summary, encoding="utf-8")
        return ApplyRunResult(
            candidate_id=candidate_id, duration_s=duration_s, error=error, usage=usage
        )

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
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.patches) == 1
    artifact = result.patches[0]
    assert artifact.candidate_id == CAND_ID
    out_dir = Path(artifact.path)
    assert out_dir == run_dir / "apply" / "v1_attention" / CAND_ID
    assert sorted(artifact.files) == [
        "APPLY-NOTES.md",
        "apply.patch",
        "apply.prompt.txt",
        "manifest.json",
    ]
    patch_text = (out_dir / "apply.patch").read_text(encoding="utf-8")
    assert "# agent edit" in patch_text
    assert "did the thing" in (out_dir / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert artifact.patch_produced is True

    # The header's apply recipe must use the $PWD-anchored patch path, not one
    # of the two broken forms that regressed twice (bare `git apply apply.patch`,
    # or `git -C {repo} apply apply.patch` with a relative, cwd-dependent path).
    # `repo_path` in the header is resolve()d by resolve_repo_path, so compare
    # against the resolved form here too.
    repo_resolved = repo.resolve()
    assert f'git -C {repo_resolved} apply "$PWD/apply.patch"' in patch_text
    assert "git apply apply.patch" not in patch_text
    assert f"git -C {repo_resolved} apply apply.patch" not in patch_text


def test_the_prompt_the_agent_ran_is_saved_beside_the_patch(run) -> None:
    """`apply.prompt.txt` records the prompt verbatim, in the `--print-prompt` shape.

    Asserted as full equality against `render_prompt_block`, fed the prompt and
    worktree the runner actually received — not a substring check. The file is
    the reviewer's only record of what the agent was *told*, so a drift between
    what the CLI prints and what the run path writes is exactly the bug: the
    skill tees that stdout into this filename, and both must produce the same
    bytes for the same candidate.
    """
    run_dir, repo = run
    seen: dict[str, object] = {}
    inner = _runner()

    def _capture(**kwargs):
        seen.update(kwargs)
        return inner(**kwargs)

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_capture,
    )
    artifact = result.patches[0]
    worktree = seen["worktree"]
    assert isinstance(worktree, Path)

    expected = render_prompt_block(
        PromptPreview(
            candidate_id=CAND_ID,
            module_qualified_name=artifact.module_qualified_name,
            repo_path=str(repo.resolve()),
            worktree=str(worktree),
            worktree_parent=str(worktree.parent),
            base_sha=artifact.base_sha,
            prompt=seen["prompt"],
        )
    )
    written = (Path(artifact.path) / "apply.prompt.txt").read_text(encoding="utf-8")
    assert written == expected


def test_the_saved_prompt_admits_its_worktree_paths_are_dead(run) -> None:
    """The worktree is already removed when `apply.prompt.txt` is written.

    `_write_artifacts` runs *after* the `finally` that calls `remove_worktree`,
    so `WORKTREE:`/`WORKTREE_PARENT:` in the saved file always name a deleted
    directory — and the prompt body names that same directory as its working
    directory. Both are kept (they correlate the artifact with the run's logs),
    so the file has to say what they are and how to get a live checkout instead.
    Asserted here rather than left implicit: the previous version of this test
    passed while the file silently advertised a path nobody could enter.
    """
    run_dir, repo = run
    seen: dict[str, object] = {}
    inner = _runner()

    def _capture(**kwargs):
        seen.update(kwargs)
        return inner(**kwargs)

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_capture,
    )
    artifact = result.patches[0]
    worktree = seen["worktree"]
    assert isinstance(worktree, Path)
    assert not worktree.exists(), "the worktree outlived the run it belongs to"

    written = (Path(artifact.path) / "apply.prompt.txt").read_text(encoding="utf-8")
    assert str(worktree) in written  # kept, not dropped
    assert "deleted when the" in written  # and labelled as gone
    # The recreate command is the whole point: without it the file records a
    # prompt whose working directory cannot be reconstructed from the file.
    assert (
        f"git -C {repo.resolve()} worktree add --detach <dir> {artifact.base_sha}" in written
    )


def test_the_prompt_is_saved_even_when_the_agent_makes_no_edit(run) -> None:
    """No patch is the case where the prompt matters most.

    "The agent declined" and "the agent was told the wrong thing" look identical
    in `APPLY-NOTES.md` alone. The prompt is what separates them, so it is
    written on the no-patch path too.
    """
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    artifact = result.patches[0]
    assert artifact.patch_produced is False
    written = (Path(artifact.path) / "apply.prompt.txt").read_text(encoding="utf-8")
    assert written.startswith(f"CANDIDATE: {CAND_ID}\n")
    assert f"BASE:      {artifact.base_sha}\n" in written


def test_added_file_appears_in_the_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/table.py"),
    )
    patch = (Path(result.patches[0].path) / "apply.patch").read_text(encoding="utf-8")
    assert "pkg/attn/table.py" in patch


def test_a_tab_in_a_created_filename_does_not_abort_the_run(run) -> None:
    """Critical 1, end to end: a tab in a filename the agent created used to

    raise an uncaught `ValueError` out of `collect_patch` — which runs inside
    the `try` whose `finally` removes the worktree — so the whole session's
    work (patch and notes) was lost before `_write_artifacts` ever ran, and
    in a sweep, the bare `ValueError` would escape `api.py`'s
    `(PrepEvolveError, OneShotApplyError)` handler entirely and abort the rest
    of the candidates too. The run must complete, the patch must be written,
    and the odd file must be represented in the notes.
    """
    run_dir, repo = run
    odd_name = "tab\tname.py"
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file=odd_name),
    )
    assert len(result.patches) == 1
    artifact = result.patches[0]
    assert artifact.patch_produced is True
    out_dir = Path(artifact.path)
    patch = (out_dir / "apply.patch").read_text(encoding="utf-8")
    assert "tab\\tname.py" in patch  # C-quoted in the plain patch text
    notes = (out_dir / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert odd_name in notes  # the manifest row carries the real path


def test_out_of_scope_edit_is_flagged_on_the_artifact_and_in_the_notes(run) -> None:
    """The manifest is derived from the diff, not from spec.targets — this is

    the whole point: an edit outside the declared scope must still land in
    the patch (never silently dropped) AND be visible both on `ApplyArtifact`
    (for the CLI) and prominently in APPLY-NOTES.md (for the reviewer).
    """
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(new_file="pkg/attn/extra.py"),
    )
    artifact = result.patches[0]
    assert artifact.out_of_scope_files == ["pkg/attn/extra.py"]

    patch = (Path(artifact.path) / "apply.patch").read_text(encoding="utf-8")
    assert "pkg/attn/extra.py" in patch  # not amputated from the patch

    notes = (Path(artifact.path) / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert "pkg/attn/extra.py" in notes
    assert "**This patch touches files outside the declared scope.**" in notes


def test_in_scope_only_edit_has_no_out_of_scope_files(run) -> None:
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    artifact = result.patches[0]
    assert artifact.out_of_scope_files == []
    notes = (Path(artifact.path) / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert "outside the declared scope" not in notes


def test_no_patch_produced_has_no_out_of_scope_files(run) -> None:
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    artifact = result.patches[0]
    assert artifact.patch_produced is False
    assert artifact.out_of_scope_files == []


def test_validation_runs_against_the_worktree_not_a_dirty_repo(run) -> None:
    """A dirty --repo must not affect the outcome: the gate reads worktree bytes."""
    run_dir, repo = run
    # Break the candidate's symbol in the *working tree* only; HEAD is still good.
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.patches) == 1
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
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
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

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo)),  # no --candidate → batch
        claude_runner=_runner(),
    )

    assert result.patches == []
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "stale" in result.skipped[0].reason.lower()


def test_worktree_is_removed_on_the_success_path(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    assert _worktrees(repo) == before


def test_worktree_is_removed_when_the_agent_fails(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary=None, error="claude exit=3"),
    )
    assert _worktrees(repo) == before
    notes = (Path(result.patches[0].path) / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert "claude exit=3" in notes
    assert result.patches[0].patch_produced is False


def test_worktree_is_removed_when_the_runner_raises(run) -> None:
    run_dir, repo = run
    before = _worktrees(repo)

    def _boom(**kwargs):
        raise RuntimeError("unexpected")

    with pytest.raises(RuntimeError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_boom,
        )
    assert _worktrees(repo) == before


def test_no_edit_produces_notes_but_no_patch(run) -> None:
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    artifact = result.patches[0]
    assert artifact.patch_produced is False
    assert artifact.files == ["APPLY-NOTES.md", "apply.prompt.txt", "manifest.json"]
    assert not (Path(artifact.path) / "apply.patch").exists()


def test_a_rerun_that_produces_no_patch_removes_the_stale_patch(run) -> None:
    """A previous session's apply.patch must not survive a rerun that concludes no change."""
    run_dir, repo = run
    first = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    out_dir = Path(first.patches[0].path)
    assert (out_dir / "apply.patch").exists()

    second = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary="cannot be done within scope"),
    )
    artifact = second.patches[0]
    assert artifact.patch_produced is False
    assert artifact.files == ["APPLY-NOTES.md", "apply.prompt.txt", "manifest.json"]
    assert not (out_dir / "apply.patch").exists()


def test_non_git_repo_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)

    with pytest.raises(NotAGitRepoError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )


def test_print_prompt_creates_no_worktree_and_runs_no_agent(run) -> None:
    """The whole point: nothing is created, so nothing has to be cleaned up.

    A worktree existed only so the gate could read bytes from a clean checkout.
    Validation now reads `--repo` — prep-evolve's own call — which leaves this
    path with no filesystem side effect in the target repo at all, and makes it
    safe to sweep.
    """
    run_dir, repo = run
    called: list[str] = []

    def _spy(**kwargs):
        called.append("ran")
        raise AssertionError("claude must not run under --print-prompt")

    before = _worktrees(repo)
    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_spy,
    )

    assert called == []
    assert result.patches == []
    assert len(result.prompts) == 1
    preview = result.prompts[0]
    assert preview.worktree is None
    assert preview.worktree_parent is None
    assert CAND_FILE in preview.prompt
    assert _worktrees(repo) == before  # no new worktree registered
    # And no `spotlights-apply-XXXX/` scaffolding left in the system temp dir.
    assert "spotlights-apply-" not in render_prompt_block(preview)


def test_print_prompt_writes_only_the_prompt_file(run) -> None:
    """`--print-prompt` writes `apply.prompt.txt` and nothing besides it.

    The file *is* the handoff — the caller reads `WORKTREE:` out of it — so it
    has to exist on disk rather than only on stdout. Exactly one file, though:
    no patch and no `CHANGE-SUMMARY.md` because no agent ran, no
    `APPLY-NOTES.md` because there is no outcome to document yet, and no
    `manifest.json` because nothing was spent. Asserted as an exact directory
    listing, not a set of `.exists()` checks: the point is what is *absent*.
    """
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_runner(),
    )

    preview = result.prompts[0]
    out_dir = Path(preview.path)
    assert out_dir == run_dir / "apply" / "v1_attention" / CAND_ID
    assert sorted(p.name for p in out_dir.iterdir()) == ["apply.prompt.txt"]
    # Same renderer, same bytes as the run path writes — the two must not drift.
    assert (out_dir / "apply.prompt.txt").read_text(encoding="utf-8") == render_prompt_block(
        preview
    )


def test_the_handoff_prompt_says_how_to_create_a_worktree(run) -> None:
    """The consumer needs a checkout and now has to make it, so hand them the command.

    The mirror of `test_the_saved_prompt_admits_its_worktree_paths_are_dead`:
    the run path's copy names two dead directories, this one names none and
    gives the recipe instead.
    """
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_runner(),
    )

    preview = result.prompts[0]
    written = (Path(preview.path) / "apply.prompt.txt").read_text(encoding="utf-8")
    assert "WORKTREE:  (none" in written
    assert (
        f"git -C {repo.resolve()} worktree add --detach <dir> {preview.base_sha}" in written
    )
    assert "still live" not in written


def _preview(**overrides) -> PromptPreview:
    """A minimal PromptPreview for exercising `render_prompt_block` directly."""
    fields = {
        "candidate_id": CAND_ID,
        "module_qualified_name": "v1/attention",
        "repo_path": "/repos/vllm",
        "worktree": "/tmp/spotlights-apply-x/worktree",
        "worktree_parent": "/tmp/spotlights-apply-x",
        "base_sha": "0" * 40,
        "prompt": "BODY",
    }
    fields.update(overrides)
    return PromptPreview(**fields)


def test_the_header_without_a_worktree_says_how_to_make_one() -> None:
    """`--print-prompt` renders no worktree path, because there is none.

    The reader has to create their own, so the header hands them the exact
    command instead of a directory. `WORKTREE:` keeps its spelling rather than
    being dropped: an existing `grep '^WORKTREE:'` gets prose back and fails
    loudly the moment it is used as a path, where a missing line would hand it
    an empty string — the input most likely to do something quiet and wrong.
    """
    block = render_prompt_block(_preview(worktree=None, worktree_parent=None, dirty=False))

    assert "WORKTREE:  (none" in block
    assert "WORKTREE_PARENT:" not in block
    assert "/tmp/spotlights-apply-x" not in block
    assert "git -C /repos/vllm worktree add --detach <dir> " + "0" * 40 in block
    assert "DIRTY:     false" in block
    assert block.endswith("PROMPT:\nBODY\n")


def test_the_header_reports_a_dirty_repo_as_dirty() -> None:
    """The gate read working-tree bytes; this is the only place that is visible.

    Not a gate — information. A consumer works at BASE, so a dirty candidate
    file means the validated bytes and the bytes they will edit can differ.
    """
    block = render_prompt_block(_preview(worktree=None, worktree_parent=None, dirty=True))
    assert "DIRTY:     true" in block


def test_the_header_with_a_worktree_keeps_both_paths_and_drops_the_hedge() -> None:
    """The run path's format is unchanged except that the NOTE no longer hedges.

    It used to have to describe both callers in one block ("if this file came
    from --print-prompt..."), because both rendered identical headers. They no
    longer do, so this block says one thing: these paths are already dead.
    """
    block = render_prompt_block(_preview())

    assert "WORKTREE:  /tmp/spotlights-apply-x/worktree\n" in block
    assert "WORKTREE_PARENT:  /tmp/spotlights-apply-x\n" in block
    assert "DIRTY:" not in block  # a permanently-false line invites a wrong reading
    assert "--print-prompt" not in block  # the hedge is gone
    assert "still live" not in block


def test_a_failed_prompt_write_skips_one_candidate_and_writes_the_rest(run) -> None:
    """Replaces the old "gives the worktree back" test: there is no worktree now.

    What still matters is that one unwritable directory costs one candidate.
    Blocking the *first* one proves the sweep continues past a failure rather
    than merely surviving one at the end.
    """
    run_dir, repo = run
    _write_candidates(run_dir, [SECOND_CAND_ID])
    # A *file* where the first candidate's apply dir must be: `mkdir` raises
    # FileExistsError there and nowhere else.
    blocked = run_dir / "apply" / "v1_attention" / CAND_ID
    blocked.parent.mkdir(parents=True)
    blocked.write_text("not a directory\n", encoding="utf-8")

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), print_prompt=True),
        claude_runner=_runner(),
    )

    assert [p.candidate_id for p in result.prompts] == [SECOND_CAND_ID]
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "could not write the apply prompt" in result.skipped[0].reason
    assert _worktrees(repo) == [f"worktree {repo.resolve()}"]


def test_print_prompt_validates_the_repo_and_records_a_dirty_tree(run) -> None:
    """prep-evolve parity: uncommitted work does not refuse the run.

    The dirty file here is the **candidate's own**, which is the case that
    matters — it is the hotspot being worked on, so it is the file most likely to
    be dirty. Appended *after* the recorded range, so lines 3-5 still hold
    `_get_tile_size` and the gate is genuinely green: this is a dirty candidate
    file that passes, not one that fails. `DIRTY: true` is the only trace, and it
    is information rather than a gate. See "Known limitations" in the design doc.
    """
    run_dir, repo = run
    cand = repo / CAND_FILE
    cand.write_text(
        cand.read_text(encoding="utf-8") + "\n# uncommitted, below the recorded range\n",
        encoding="utf-8",
    )

    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_runner(),
    )

    assert len(result.prompts) == 1
    assert result.prompts[0].dirty is True
    written = (Path(result.prompts[0].path) / "apply.prompt.txt").read_text(encoding="utf-8")
    assert "DIRTY:     true" in written


def test_print_prompt_gate_reads_the_working_tree_not_head(run) -> None:
    """The documented cost of dropping the worktree, pinned rather than left implicit.

    A gutted *working tree* now fails the gate even though HEAD is fine — the
    exact inverse of `test_validation_runs_against_the_worktree_not_a_dirty_repo`,
    which still holds for the run path. Loud, so it gets noticed; the silent
    inverse (working tree matches, HEAD does not) is the known limitation the
    design records `git show` as the upgrade path for.
    """
    run_dir, repo = run
    (repo / CAND_FILE).write_text("# gutted\n" * 20, encoding="utf-8")

    with pytest.raises(StalenessError):
        one_shot_apply(
            OneShotApplyInput(
                result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
            ),
            claude_runner=_runner(),
        )


def test_print_prompt_clears_an_earlier_applys_artifacts(run) -> None:
    """Reruns share the directory, and `--print-prompt` leaves only the prompt in it.

    A patch, notes and manifest from an earlier apply of this candidate describe a
    session that has been superseded by the handoff starting now. Keeping them
    would leave the directory documenting an outcome the caller is about to
    replace — and a patch beside a prompt that produced it is exactly the
    confusion `apply.patch`'s base-commit header exists to prevent.

    Destructive on purpose: `--print-prompt` is how you *restart* a candidate, so
    the previous patch is discarded before the new attempt begins.
    """
    run_dir, repo = run
    applied = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    out_dir = Path(applied.patches[0].path)
    assert sorted(p.name for p in out_dir.iterdir()) == [
        "APPLY-NOTES.md",
        "apply.patch",
        "apply.prompt.txt",
        "manifest.json",
    ]

    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_runner(),
    )

    preview = result.prompts[0]
    assert Path(preview.path) == out_dir
    assert sorted(p.name for p in out_dir.iterdir()) == ["apply.prompt.txt"]


def test_a_full_apply_writes_all_four_artifacts(run) -> None:
    """The run path's complete output: patch, notes, prompt, manifest — no more, no less.

    Pinned as an exact listing against both the directory and `ApplyArtifact.files`,
    because the two are read by different consumers (a human opening the folder, and
    `/spotlights-share-candidates` folding it into a bundle) and a file present in
    one but unreported by the other is invisible to whichever reads the other.
    """
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    expected = ["APPLY-NOTES.md", "apply.patch", "apply.prompt.txt", "manifest.json"]
    artifact = result.patches[0]
    assert sorted(p.name for p in Path(artifact.path).iterdir()) == expected
    assert sorted(artifact.files) == expected


def test_print_prompt_without_a_candidate_writes_one_prompt_per_candidate(run) -> None:
    """The guard existed only to bound the worktree leak; there is no leak now.

    `--print-prompt` composes with bare selection exactly like the run path,
    which is the second half of what made the flag skill-only.
    """
    run_dir, repo = run
    _write_candidates(run_dir, [SECOND_CAND_ID])
    before = _worktrees(repo)

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), print_prompt=True),
        claude_runner=_runner(),
    )

    assert [p.candidate_id for p in result.prompts] == [CAND_ID, SECOND_CAND_ID]
    assert result.skipped == []
    assert _worktrees(repo) == before
    for preview in result.prompts:
        assert sorted(p.name for p in Path(preview.path).iterdir()) == ["apply.prompt.txt"]


def test_print_prompt_top_n_clears_only_the_selected_candidates(run) -> None:
    """Clearing is scoped to what the invocation selected, not to the whole run.

    `--print-prompt` deletes an earlier apply's patch/notes/manifest, and a sweep
    does that per candidate. An unselected candidate's finished apply must
    survive untouched — asserted through a third candidate's `apply.patch`,
    which is the file whose loss is unrecoverable.
    """
    run_dir, repo = run
    _write_candidates(run_dir, [SECOND_CAND_ID, THIRD_CAND_ID])
    sorted_dir = write_sorted(run_dir, [CAND_ID, SECOND_CAND_ID, THIRD_CAND_ID])

    # A finished apply for the candidate that --top-n 2 will not reach.
    untouched = run_dir / "apply" / "v1_attention" / THIRD_CAND_ID
    untouched.mkdir(parents=True)
    (untouched / "apply.patch").write_text("# an earlier apply\n", encoding="utf-8")

    # A stale file for the first candidate, seeded before the one_shot_apply call.
    stale = run_dir / "apply" / "v1_attention" / CAND_ID
    stale.mkdir(parents=True)
    (stale / "apply.patch").write_text("# superseded\n", encoding="utf-8")

    result = one_shot_apply(
        OneShotApplyInput(result=sorted_dir, repo=str(repo), top_n=2, print_prompt=True),
        claude_runner=_runner(),
    )

    assert [p.candidate_id for p in result.prompts] == [CAND_ID, SECOND_CAND_ID]
    assert (untouched / "apply.patch").read_text(encoding="utf-8") == "# an earlier apply\n"
    assert not (untouched / "apply.prompt.txt").exists()
    # And clearing did happen for the two that were selected.
    selected = run_dir / "apply" / "v1_attention" / CAND_ID
    assert sorted(p.name for p in selected.iterdir()) == ["apply.prompt.txt"]


def test_missing_claude_binary_raises_before_the_loop_with_the_real_runner(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))
    with pytest.raises(ClaudeUnavailableError):
        one_shot_apply(OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID))

    # This is the property the (not-yet-written) CLI depends on: it catches
    # `(PrepEvolveError, OneShotApplyError)` once and maps both to a clean
    # stderr message. A bare RuntimeError subclass (AgentProposalsSetupError)
    # would slip past that catch and surface as a traceback.
    try:
        one_shot_apply(OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID))
    except OneShotApplyError:
        pass
    else:
        pytest.fail("expected a OneShotApplyError to be raised")


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
        one_shot_apply(OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID))


def test_print_prompt_does_not_require_claude_on_path(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True)
    )

    assert len(result.prompts) == 1


def test_injected_runner_does_not_require_claude_on_path(
    run, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_dir, repo = run
    monkeypatch.setenv("PATH", _git_only_path(tmp_path))

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    assert len(result.patches) == 1


def test_top_n_over_a_ranking_selects_the_ranked_prefix(run) -> None:
    run_dir, repo = run
    sorted_dir = write_sorted(run_dir, [CAND_ID])

    result = one_shot_apply(
        OneShotApplyInput(result=sorted_dir, repo=str(repo), top_n=1),
        claude_runner=_runner(),
    )

    assert len(result.patches) == 1
    # Artifacts land under the run dir, not inside sorted/.
    assert Path(result.patches[0].path) == run_dir / "apply" / "v1_attention" / CAND_ID


def test_out_overrides_the_artifact_base(run, tmp_path: Path) -> None:
    run_dir, repo = run
    out = tmp_path / "elsewhere"
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID, out=out),
        claude_runner=_runner(),
    )
    assert Path(result.patches[0].path) == out / "apply" / "v1_attention" / CAND_ID


def test_an_unwritable_out_dir_skips_one_candidate_instead_of_aborting_the_sweep(
    run, tmp_path: Path
) -> None:
    """Artifact IO failures must be `OneShotApplyError`s, or a sweep dies mid-way.

    `_write_artifacts` runs `mkdir` and two writes. A bare `OSError` from any
    of them propagates straight past the batch loop's
    `except (PrepEvolveError, OneShotApplyError)` handler and out of
    `one_shot_apply` — so one unwritable path (full disk, read-only mount,
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

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), out=blocked),  # batch: no --candidate
        claude_runner=_runner(),
    )

    assert result.patches == []
    assert len(result.skipped) == 1
    assert result.skipped[0].candidate_id == CAND_ID
    assert "could not write apply artifacts" in result.skipped[0].reason
    # The worktree is still gone: cleanup happens before artifacts are written.
    assert len(_worktrees(repo)) == 1


def test_an_unwritable_out_dir_raises_for_an_explicit_candidate(run, tmp_path: Path) -> None:
    """Same failure, single-candidate mode: raise, and as a `OneShotApplyError`.

    The CLI catches exactly `(PrepEvolveError, OneShotApplyError)` and turns it
    into `apply: <message>` with exit 2. A bare `OSError` would instead reach the
    user as a traceback.
    """
    run_dir, repo = run
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(ArtifactWriteError) as excinfo:
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID, out=blocked),
            claude_runner=_runner(),
        )
    assert isinstance(excinfo.value, OneShotApplyError)
    assert len(_worktrees(repo)) == 1


def test_a_patch_write_that_fails_partway_leaves_no_truncated_patch_behind(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`write_bytes` is not atomic, and a half-written `apply.patch` is worse than none.

    An `ENOSPC` (or a write error on a network mount) partway through a
    multi-megabyte patch leaves a *prefix* of the diff on disk under the exact
    name the docs tell a reviewer to apply. Raising over it and leaving the file
    there is the worst outcome available: `git apply` either rejects it, or — at
    an unlucky hunk boundary — applies part of it. Removing it makes the failure
    read as "no patch here", which is true, and matches the empty-patch path
    that already unlinks a stale `apply.patch`.
    """
    run_dir, repo = run
    real_write_bytes = Path.write_bytes

    def _short_write(self: Path, data: bytes) -> int:
        if self.name != "apply.patch":
            return real_write_bytes(self, data)
        real_write_bytes(self, data[: len(data) // 2])  # the truncated prefix
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_bytes", _short_write)

    with pytest.raises(ArtifactWriteError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )

    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    assert not (out_dir / "apply.patch").exists()


def test_a_notes_write_that_fails_partway_leaves_neither_artifact_behind(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror case: the patch landed, then the notes write died.

    `APPLY-NOTES.md` is human-facing, so a truncated one cannot be misapplied the
    way a truncated patch can — but it can lose the "Applying and verifying"
    section that records *which commit* the patch belongs to, and a patch whose
    base is unknown is not applicable. The directory must not be left holding
    a `apply.patch` documented by half a page either, so the handler removes
    both: the same "nothing here" invariant the empty-patch branch maintains.
    """
    run_dir, repo = run
    real_write_text = Path.write_text

    def _short_write(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if self.name != "APPLY-NOTES.md":
            return real_write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]
        real_write_text(self, data[: len(data) // 2], encoding="utf-8")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", _short_write)

    with pytest.raises(ArtifactWriteError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )

    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    assert not (out_dir / "APPLY-NOTES.md").exists()
    # And the patch that *did* write successfully goes with it — a patch with
    # no notes has no recorded base commit. `apply.prompt.txt` goes too: on its
    # own it documents an apply that left nothing else behind.
    assert not (out_dir / "apply.patch").exists()
    assert not (out_dir / "apply.prompt.txt").exists()
    # The notes write fails *before* the manifest write, which sits under its
    # own guard after this handler — so this run's manifest was never reached,
    # let alone created. Absent for a different reason than the three above, and
    # still absent. The stale case is the next test.
    assert not (out_dir / "manifest.json").exists()


def test_a_failed_write_also_clears_a_previous_runs_manifest(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reruns share the directory, so "nothing here" has to mean all four names.

    This run's `manifest.json` is written after the cleanup handler and is
    unreachable from it — but a *previous* apply of this candidate left one in
    the same directory. If the handler skipped that name, a failed rerun would
    leave last run's accounting sitting alone beside no patch and no notes:
    accounting for a patch that is not there, which is the one state the
    handler exists to prevent.
    """
    run_dir, repo = run
    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text('{"stale": true}\n', encoding="utf-8")

    real_write_text = Path.write_text

    def _fail_the_notes(self: Path, data: str, *args: object, **kwargs: object) -> int:
        if self.name != "APPLY-NOTES.md":
            return real_write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", _fail_the_notes)

    with pytest.raises(ArtifactWriteError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_runner(),
        )

    assert not (out_dir / "manifest.json").exists()
    assert not (out_dir / "apply.patch").exists()
    assert not (out_dir / "APPLY-NOTES.md").exists()
    assert not (out_dir / "apply.prompt.txt").exists()


def test_a_manifest_write_that_fails_leaves_the_patch_and_its_notes_behind(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one write whose failure must cost nothing but itself.

    `manifest.json` is written after the `finally` that destroys the worktree,
    so the patch beside it cannot be produced again — while a directory holding
    patch + notes + prompt and no manifest is exactly what every apply
    directory looked like before the manifest existed, i.e. a state a reader
    reads correctly. So an `ENOSPC` on the last and smallest write degrades the
    accounting instead of unlinking three good artifacts and raising
    `ArtifactWriteError`, which the batch loop would record as a skipped
    candidate for a patch that had already succeeded.
    """
    run_dir, repo = run
    real_write_text = Path.write_text

    def _fail_only_the_manifest(
        self: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        if self.name != "manifest.json":
            return real_write_text(self, data, *args, **kwargs)  # type: ignore[arg-type]
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Path, "write_text", _fail_only_the_manifest)

    # No `pytest.raises`: the call returning at all is half of what this test
    # asserts.
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )

    artifact = result.patches[0]
    out_dir = Path(artifact.path)
    assert artifact.patch_produced is True
    assert (out_dir / "apply.patch").stat().st_size > 0
    assert (out_dir / "APPLY-NOTES.md").exists()
    assert (out_dir / "apply.prompt.txt").exists()
    assert not (out_dir / "manifest.json").exists()
    # The file list is the artifact's own record of what a reviewer will find,
    # so it must not promise a manifest that is not there.
    assert artifact.files == ["APPLY-NOTES.md", "apply.patch", "apply.prompt.txt"]


def test_a_keyboard_interrupt_during_the_agent_session_leaks_no_worktree(run) -> None:
    """`KeyboardInterrupt` is a `BaseException`, and Ctrl-C is how a real sweep ends.

    A long `apply` sweep is interrupted by hand far more often than it fails, so
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
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
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
    `APPLY-NOTES.md`, no `run.error`, no usage — and in a sweep, a skip whose
    reason is a git message with no hint that a paid session was thrown away.

    Reporting it as `patch=b""` alone would be worse than nothing: the notes
    would then state "No patch was produced. The agent made no in-scope edit."
    about a session that may well have edited every file in scope. Hence the
    separate `collect_error` channel.
    """
    run_dir, repo = run
    _break_the_diff(monkeypatch, "git diff timed out after 60s")

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(summary="rewrote the tile heuristic"),
    )

    assert result.skipped == []
    artifact = result.patches[0]
    assert artifact.collection_error is not None
    assert "timed out" in artifact.collection_error
    assert artifact.patch_produced is False
    assert artifact.files == ["APPLY-NOTES.md", "apply.prompt.txt", "manifest.json"]

    notes = (Path(artifact.path) / "APPLY-NOTES.md").read_text(encoding="utf-8")
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

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(error="claude exit=1: stderr=b'overloaded'"),
    )

    notes = (Path(result.patches[0].path) / "APPLY-NOTES.md").read_text(encoding="utf-8")
    assert "overloaded" in notes
    assert "git diff timed out after 60s" in notes


def test_a_failed_diff_leaves_no_stale_patch_from_a_previous_run(
    run, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The notes say the patch is unknown, so a previous session's must not sit there."""
    run_dir, repo = run
    first = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    out_dir = Path(first.patches[0].path)
    assert (out_dir / "apply.patch").exists()

    _break_the_diff(monkeypatch, "git diff timed out after 60s")
    one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(),
    )
    assert not (out_dir / "apply.patch").exists()


def _write_candidates(run_dir: Path, extra_ids: list[str]) -> None:
    """Rewrite result.json with `extra_ids` cloned from the first candidate."""
    data = make_result_dict()
    cands = data["module_runs"]["v1/attention"]["candidates"]["candidates"]
    for cid in extra_ids:
        clone = json.loads(json.dumps(cands[0]))
        clone["id"] = cid
        cands.append(clone)
    (run_dir / "result.json").write_text(json.dumps(data), encoding="utf-8")


def _write_two_candidates(run_dir: Path) -> None:
    _write_candidates(run_dir, [SECOND_CAND_ID])


def test_an_unexpected_exception_in_a_sweep_skips_one_candidate_and_continues(
    run,
) -> None:
    """Each candidate in a sweep costs a paid agent session; one bug must not burn the rest.

    The batch loop caught only `(PrepEvolveError, OneShotApplyError)`. Any other
    exception type — a pydantic `ValidationError`, an `OSError` from somewhere
    not already wrapped, a bug in this module — propagated out of
    `one_shot_apply`, discarding every candidate still queued. Unlike
    `prep_evolve`, whose equally narrow loop is free to re-run, that throws
    away sessions that were never started.
    """
    run_dir, repo = run
    _write_two_candidates(run_dir)

    def _boom_on_the_first(*, candidate_id: str, prompt: str, worktree: Path,
                           max_turns: int, wallclock_s: int,
                           claude_model: str | None = None) -> ApplyRunResult:
        if candidate_id == CAND_ID:
            raise RuntimeError("a bug nobody anticipated")
        return _runner()(
            candidate_id=candidate_id,
            prompt=prompt,
            worktree=worktree,
            max_turns=max_turns,
            wallclock_s=wallclock_s,
        )

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo)),  # batch: no --candidate
        claude_runner=_boom_on_the_first,
    )

    # The second candidate still ran.
    assert [f.candidate_id for f in result.patches] == [SECOND_CAND_ID]
    assert result.patches[0].patch_produced is True

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
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo)),  # batch
            claude_runner=_interrupt,
        )


def test_an_unexpected_exception_still_raises_for_an_explicit_candidate(run) -> None:
    """Single-candidate mode keeps the traceback: there is nothing to protect."""
    run_dir, repo = run

    def _boom(**kwargs):
        raise RuntimeError("a bug nobody anticipated")

    with pytest.raises(RuntimeError):
        one_shot_apply(
            OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
            claude_runner=_boom,
        )


def test_the_manifest_lands_beside_the_patch_and_agrees_with_the_artifact(run) -> None:
    """The session's tokens were previously parsed, copied onto ApplyArtifact, and
    dropped. This is the surface that ends that: a clean apply and a timed-out
    apply used to report the same thing about cost, which was nothing."""
    run_dir, repo = run
    usage = AgentUsage(
        input=100, output=200, cache_read=300, cache_create=400,
        model="aws/claude-opus-5", api_time_s=12.5,
    )
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(usage=usage, duration_s=33.0),
    )

    artifact = result.patches[0]
    assert "manifest.json" in artifact.files
    out_dir = Path(artifact.path)

    m = ApplyUsageManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert m.candidate_id == artifact.candidate_id
    assert m.module_qualified_name == artifact.module_qualified_name
    assert m.target.commit_sha == artifact.base_sha
    assert m.total_tokens == 1000
    assert m.models_used[0].role == "one_shot_apply"
    assert m.timing.wall_clock_s == 33.0
    assert m.timing.api_time_s == 12.5
    # Priced by the bundled contracted table, so a real dollar figure — the
    # amount is the bundled table's business, not this test's.
    assert m.cost.amount_usd > 0.0
    assert m.external_cost is not None
    assert m.external_cost.amount_usd > m.cost.amount_usd


def test_the_manifest_keys_are_sorted_on_disk_like_the_run_manifest(run) -> None:
    """`json.dumps(..., indent=2, sort_keys=True)`, matching persistence.py:89.
    Stable key order is what keeps a re-run's diff to the values that changed."""
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(usage=AgentUsage(input=1, output=2, model="aws/claude-opus-5")),
    )
    text = (Path(result.patches[0].path) / "manifest.json").read_text(encoding="utf-8")
    keys = list(json.loads(text).keys())
    assert keys == sorted(keys)
    assert text.endswith("\n")


def test_a_session_that_produced_no_patch_still_gets_a_manifest(run) -> None:
    """Written unconditionally, like the notes and the prompt and unlike the
    patch. A session that errored or chose not to edit still spent tokens, and
    that is precisely when someone asks what it cost."""
    run_dir, repo = run
    # The fixture repo has no `origin`, and the builder appends a second note for
    # that gap. Give it one, so `notes` below is exactly the degraded-usage
    # sentence and nothing else — and so the resolved `repo_url` is asserted to
    # have actually reached the file, which is the only check on that wiring.
    origin = "https://example.invalid/attn.git"
    subprocess.run(["git", "remote", "add", "origin", origin], cwd=repo, check=True)

    result = one_shot_apply(
        OneShotApplyInput(result=run_dir, repo=str(repo), candidate=CAND_ID),
        claude_runner=_runner(edit=None, summary=None, error="model unavailable"),
    )

    artifact = result.patches[0]
    assert artifact.patch_produced is False
    out_dir = Path(artifact.path)
    assert not (out_dir / "apply.patch").exists()

    m = ApplyUsageManifest.model_validate_json(
        (out_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert m.models_used == []
    assert m.target.repo_url == origin
    # `startswith`, not equality: `spotlights_commit_sha()` shells out to `git
    # rev-parse HEAD` and returns "" from a wheel/sdist export with no engine
    # checkout, which appends a second, unrelated note. The degraded-usage
    # sentence is what this test is about, and it is pinned in full.
    assert m.notes.startswith("no usage records found: model unavailable")


def test_print_prompt_writes_no_manifest(run) -> None:
    """`--print-prompt` returns before the runner, so there is no session and
    nothing to account for."""
    run_dir, repo = run
    result = one_shot_apply(
        OneShotApplyInput(
            result=run_dir, repo=str(repo), candidate=CAND_ID, print_prompt=True
        ),
        claude_runner=_runner(),
    )
    assert result.patches == []
    assert not (run_dir / "apply" / "v1_attention" / CAND_ID / "manifest.json").exists()
