"""`spotlights-engine apply` argument handling and output shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.cli import main as engine_main
from spotlights_engine.one_shot_apply.cli import main as apply_main
from tests.unit.prep_evolve._fixtures import CAND_FILE, make_repo, write_index, write_result

CAND_ID = "cand-v1_attention-0002"


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path)
    write_index(run_dir, repo)
    return run_dir, repo


def test_bad_top_n_exits_2(run, capsys: pytest.CaptureFixture) -> None:
    run_dir, repo = run
    code = apply_main(
        ["--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err


def test_zero_top_n_exits_2(run) -> None:
    run_dir, repo = run
    assert apply_main(["--result", str(run_dir), "--repo", str(repo), "--top-n", "0"]) == 2


def test_unknown_candidate_exits_2(run, capsys: pytest.CaptureFixture) -> None:
    run_dir, repo = run
    code = apply_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", "cand-nope-0001"]
    )
    assert code == 2
    assert "apply:" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--max-turns", "0"),
        ("--wallclock", "0"),
        ("--max-turns", "-1"),
    ],
)
def test_non_positive_max_turns_or_wallclock_exits_2(
    run, capsys: pytest.CaptureFixture, flag: str, value: str
) -> None:
    run_dir, repo = run
    code = apply_main(
        [
            "--result",
            str(run_dir),
            "--repo",
            str(repo),
            "--candidate",
            CAND_ID,
            flag,
            value,
        ]
    )
    err = capsys.readouterr().err
    assert code == 2
    assert "apply:" in err
    # The message must name the CLI flag the user actually typed, not the
    # pydantic model field (e.g. "wallclock_s") or its multi-line dump.
    assert flag in err
    assert "https://errors.pydantic.dev" not in err
    assert err.strip().count("\n") == 0


def test_non_git_repo_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    run_dir = tmp_path / "spotlights-out"
    run_dir.mkdir()
    write_result(run_dir)
    repo = make_repo(tmp_path, git=False)
    code = apply_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID]
    )
    assert code == 2
    assert "git" in capsys.readouterr().err


def test_print_prompt_writes_the_prompt_and_worktree_to_stdout(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = apply_main(
        [
            "--result",
            str(run_dir),
            "--repo",
            str(repo),
            "--candidate",
            CAND_ID,
            "--print-prompt",
        ]
    )
    captured = capsys.readouterr()
    out = captured.out
    assert code == 0
    assert "WORKTREE:" in out
    assert CAND_FILE in out
    # `--print-prompt` writes nothing and skips nothing, so there is no tally to
    # print: "0 candidate(s) written" under a prompt dump would read as failure.
    assert "candidate(s) written" not in captured.err

    # The worktree is left in place for the caller; clean it up, including the
    # temp-dir parent that WORKTREE_PARENT points at (create_worktree allocates
    # it via tempfile.mkdtemp; only this cleanup removes it).
    import shutil
    import subprocess

    worktree = next(
        ln.split("WORKTREE:", 1)[1].strip()
        for ln in out.splitlines()
        if ln.startswith("WORKTREE:")
    )
    worktree_parent = next(
        ln.split("WORKTREE_PARENT:", 1)[1].strip()
        for ln in out.splitlines()
        if ln.startswith("WORKTREE_PARENT:")
    )
    assert Path(worktree).is_dir()
    assert Path(worktree_parent).is_dir()
    subprocess.run(["git", "worktree", "remove", "--force", worktree], cwd=repo, check=True)
    shutil.rmtree(worktree_parent, ignore_errors=True)
    assert not Path(worktree_parent).exists()


def test_out_of_scope_files_are_flagged_on_stderr(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The operator running a sweep should see this without opening APPLY-NOTES.md.

    Everything but the `claude -p` call itself runs for real here (real repo,
    real worktree, real notes rendering) — only the agent session is faked,
    by monkeypatching the two module-level hooks `one_shot_apply` uses for it
    (`ensure_claude_available`, `run_apply_claude`), the same seam the CLI has
    no flag to bypass.
    """
    run_dir, repo = run
    import spotlights_engine.one_shot_apply.api as api_mod
    from spotlights_engine.one_shot_apply.claude_exec import ApplyRunResult

    def _fake_run_apply_claude(
        *, candidate_id: str, prompt: str, worktree: Path, max_turns: int,
        wallclock_s: int, claude_model: str | None = None,
    ) -> ApplyRunResult:
        target = worktree / CAND_FILE
        target.write_text(target.read_text(encoding="utf-8") + "# in-scope\n", encoding="utf-8")
        extra = worktree / "pkg" / "attn" / "extra.py"
        extra.write_text("EXTRA = 1\n", encoding="utf-8")
        (worktree / "CHANGE-SUMMARY.md").write_text(
            "did the in-scope thing, plus an extra file", encoding="utf-8"
        )
        return ApplyRunResult(candidate_id=candidate_id, duration_s=0.01)

    monkeypatch.setattr(api_mod, "ensure_claude_available", lambda: None)
    monkeypatch.setattr(api_mod, "run_apply_claude", _fake_run_apply_claude)

    code = apply_main(["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID])
    captured = capsys.readouterr()

    assert code == 0
    assert f"{CAND_ID}:" in captured.out  # the machine-readable line stays on stdout
    assert "pkg/attn/extra.py" in captured.err
    assert "outside the declared scope" in captured.err


def test_a_failed_patch_collection_is_not_printed_as_notes_only(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """"notes only (no patch)" reads as a decision; a failed diff is a lost session.

    The operator watching a sweep scroll past has only this one line per
    candidate. Printing a `git diff` that timed out the same way as an agent
    that deliberately made no edit hides the one case that needs a re-run.
    """
    import spotlights_engine.one_shot_apply.cli as cli_mod
    from spotlights_engine.one_shot_apply.api import ApplyArtifact, OneShotApplyResult

    run_dir, repo = run
    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    monkeypatch.setattr(
        cli_mod,
        "one_shot_apply",
        lambda inp, cfg: OneShotApplyResult(
            patches=[
                ApplyArtifact(
                    candidate_id=CAND_ID,
                    module_qualified_name="v1/attention",
                    path=str(out_dir),
                    files=["APPLY-NOTES.md"],
                    patch_produced=False,
                    base_sha="0" * 40,
                    collection_error="git diff timed out after 60s",
                )
            ]
        ),
    )

    code = apply_main(["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID])
    captured = capsys.readouterr()

    assert code == 0
    assert "notes only (no patch)" not in captured.out
    assert "patch collection FAILED" in captured.out
    assert "git diff timed out after 60s" in captured.err
    assert "re-run this candidate" in captured.err


@pytest.mark.parametrize(
    ("patch_produced", "collection_error", "expected"),
    [
        (True, None, "(3 files)"),
        (False, None, "(2 files, no patch)"),
        (False, "git diff timed out after 60s", "(2 files, patch collection FAILED)"),
    ],
)
def test_the_per_candidate_line_leads_with_the_file_count(
    run,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    patch_produced: bool,
    collection_error: str | None,
    expected: str,
) -> None:
    """`prep-evolve`'s shape: `<id>: <path> (N files)`.

    The count is what the two arms share, so a sweep over both scans as one
    column. It cannot carry the outcome, though — 2 files means "no patch" and
    "the diff was lost" alike — so those two keep a suffix and only the normal
    case reads bare.
    """
    import spotlights_engine.one_shot_apply.cli as cli_mod
    from spotlights_engine.one_shot_apply.api import ApplyArtifact, OneShotApplyResult

    run_dir, repo = run
    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    files = ["APPLY-NOTES.md", "apply.prompt.txt"]
    if patch_produced:
        files.append("apply.patch")
    monkeypatch.setattr(
        cli_mod,
        "one_shot_apply",
        lambda inp, cfg: OneShotApplyResult(
            patches=[
                ApplyArtifact(
                    candidate_id=CAND_ID,
                    module_qualified_name="v1/attention",
                    path=str(out_dir),
                    files=sorted(files),
                    patch_produced=patch_produced,
                    base_sha="0" * 40,
                    collection_error=collection_error,
                )
            ]
        ),
    )

    code = apply_main(["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID])
    captured = capsys.readouterr()

    assert code == 0
    assert f"{CAND_ID}: {out_dir} {expected}" in captured.out
    # The old shape is gone, not merely supplemented.
    assert "patch + notes" not in captured.out
    assert "notes only" not in captured.out


def test_the_summary_line_tallies_what_was_written_and_skipped(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The closing tally, matching `prep-evolve: N bundle(s) written, M skipped`.

    On stderr, so stdout stays the machine-readable one-line-per-candidate list
    a caller can pipe.
    """
    import spotlights_engine.one_shot_apply.cli as cli_mod
    from spotlights_engine.one_shot_apply.api import (
        ApplyArtifact,
        OneShotApplyResult,
        SkippedApply,
    )

    run_dir, repo = run
    out_dir = run_dir / "apply" / "v1_attention" / CAND_ID
    monkeypatch.setattr(
        cli_mod,
        "one_shot_apply",
        lambda inp, cfg: OneShotApplyResult(
            patches=[
                ApplyArtifact(
                    candidate_id=CAND_ID,
                    module_qualified_name="v1/attention",
                    path=str(out_dir),
                    files=["APPLY-NOTES.md", "apply.patch", "apply.prompt.txt"],
                    patch_produced=True,
                    base_sha="0" * 40,
                )
            ],
            skipped=[
                SkippedApply(reason="no longer in result.json", candidate_id="cand-other-0001"),
                SkippedApply(reason="no scope", candidate_id="cand-other-0002"),
            ],
        ),
    )

    code = apply_main(["--result", str(run_dir), "--repo", str(repo)])
    captured = capsys.readouterr()

    assert code == 0
    assert "apply: 1 candidate(s) written, 2 skipped" in captured.err
    assert "candidate(s) written" not in captured.out


def test_a_skip_only_run_still_reports_the_tally(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Nothing written is exactly when the operator most needs the count."""
    import spotlights_engine.one_shot_apply.cli as cli_mod
    from spotlights_engine.one_shot_apply.api import OneShotApplyResult, SkippedApply

    run_dir, repo = run
    monkeypatch.setattr(
        cli_mod,
        "one_shot_apply",
        lambda inp, cfg: OneShotApplyResult(
            skipped=[SkippedApply(reason="no longer in result.json", candidate_id=CAND_ID)]
        ),
    )

    apply_main(["--result", str(run_dir), "--repo", str(repo)])
    assert "apply: 0 candidate(s) written, 1 skipped" in capsys.readouterr().err


def test_engine_dispatch_routes_fix_to_the_subcommand(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = engine_main(
        ["apply", "--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err


def test_a_selection_skip_without_a_module_does_not_print_a_literal_none(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A skip raised during *selection* has no module, and must not read `None/cand-…`.

    `_select` records a `SkippedApply` for a ranked id that is no longer in
    result.json — it knows the id but not the module, so
    `module_qualified_name` is None. The old
    `f"{s.module_qualified_name}/{s.candidate_id}"` interpolated that straight
    into the operator's stderr as a literal `None/cand-xxxx`, which reads like
    a real module named "None".
    """
    import spotlights_engine.one_shot_apply.cli as cli_mod
    from spotlights_engine.one_shot_apply.api import OneShotApplyResult, SkippedApply

    run_dir, repo = run
    monkeypatch.setattr(
        cli_mod,
        "one_shot_apply",
        lambda inp, cfg: OneShotApplyResult(
            skipped=[SkippedApply(reason="no longer in result.json", candidate_id=CAND_ID)]
        ),
    )

    code = apply_main(["--result", str(run_dir), "--repo", str(repo)])
    err = capsys.readouterr().err

    assert code == 0  # a recorded skip is still a produced outcome
    assert f"skipped {CAND_ID}: no longer in result.json" in err
    assert "None" not in err
