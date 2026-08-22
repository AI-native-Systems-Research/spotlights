"""`spotlights-engine fix` argument handling and output shape."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.cli import main as engine_main
from spotlights_engine.one_shot_fix.cli import main as fix_main
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
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err


def test_zero_top_n_exits_2(run) -> None:
    run_dir, repo = run
    assert fix_main(["--result", str(run_dir), "--repo", str(repo), "--top-n", "0"]) == 2


def test_unknown_candidate_exits_2(run, capsys: pytest.CaptureFixture) -> None:
    run_dir, repo = run
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", "cand-nope-0001"]
    )
    assert code == 2
    assert "fix:" in capsys.readouterr().err


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
    code = fix_main(
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
    assert "fix:" in err
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
    code = fix_main(
        ["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID]
    )
    assert code == 2
    assert "git" in capsys.readouterr().err


def test_print_prompt_writes_the_prompt_and_worktree_to_stdout(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = fix_main(
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
    out = capsys.readouterr().out
    assert code == 0
    assert "WORKTREE:" in out
    assert CAND_FILE in out

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
    """The operator running a sweep should see this without opening FIX-NOTES.md.

    Everything but the `claude -p` call itself runs for real here (real repo,
    real worktree, real notes rendering) — only the agent session is faked,
    by monkeypatching the two module-level hooks `one_shot_fix` uses for it
    (`ensure_claude_available`, `run_fix_claude`), the same seam the CLI has
    no flag to bypass.
    """
    run_dir, repo = run
    import spotlights_engine.one_shot_fix.api as api_mod
    from spotlights_engine.one_shot_fix.claude_exec import FixRunResult

    def _fake_run_fix_claude(
        *, candidate_id: str, prompt: str, worktree: Path, max_turns: int, wallclock_s: int
    ) -> FixRunResult:
        target = worktree / CAND_FILE
        target.write_text(target.read_text(encoding="utf-8") + "# in-scope\n", encoding="utf-8")
        extra = worktree / "pkg" / "attn" / "extra.py"
        extra.write_text("EXTRA = 1\n", encoding="utf-8")
        (worktree / "CHANGE-SUMMARY.md").write_text(
            "did the in-scope thing, plus an extra file", encoding="utf-8"
        )
        return FixRunResult(candidate_id=candidate_id, duration_s=0.01)

    monkeypatch.setattr(api_mod, "ensure_claude_available", lambda: None)
    monkeypatch.setattr(api_mod, "run_fix_claude", _fake_run_fix_claude)

    code = fix_main(["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID])
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
    import spotlights_engine.one_shot_fix.cli as cli_mod
    from spotlights_engine.one_shot_fix.api import FixArtifact, OneShotFixResult

    run_dir, repo = run
    out_dir = run_dir / "fix" / "v1_attention" / CAND_ID
    monkeypatch.setattr(
        cli_mod,
        "one_shot_fix",
        lambda inp, cfg: OneShotFixResult(
            fixes=[
                FixArtifact(
                    candidate_id=CAND_ID,
                    module_qualified_name="v1/attention",
                    path=str(out_dir),
                    files=["FIX-NOTES.md"],
                    patch_produced=False,
                    base_sha="0" * 40,
                    collection_error="git diff timed out after 60s",
                )
            ]
        ),
    )

    code = fix_main(["--result", str(run_dir), "--repo", str(repo), "--candidate", CAND_ID])
    captured = capsys.readouterr()

    assert code == 0
    assert "notes only (no patch)" not in captured.out
    assert "patch collection FAILED" in captured.out
    assert "git diff timed out after 60s" in captured.err
    assert "re-run this candidate" in captured.err


def test_engine_dispatch_routes_fix_to_the_subcommand(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = engine_main(
        ["fix", "--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err


def test_a_selection_skip_without_a_module_does_not_print_a_literal_none(
    run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A skip raised during *selection* has no module, and must not read `None/cand-…`.

    `_select` records a `SkippedFix` for a ranked id that is no longer in
    result.json — it knows the id but not the module, so
    `module_qualified_name` is None. The old
    `f"{s.module_qualified_name}/{s.candidate_id}"` interpolated that straight
    into the operator's stderr as a literal `None/cand-xxxx`, which reads like
    a real module named "None".
    """
    import spotlights_engine.one_shot_fix.cli as cli_mod
    from spotlights_engine.one_shot_fix.api import OneShotFixResult, SkippedFix

    run_dir, repo = run
    monkeypatch.setattr(
        cli_mod,
        "one_shot_fix",
        lambda inp, cfg: OneShotFixResult(
            skipped=[SkippedFix(reason="no longer in result.json", candidate_id=CAND_ID)]
        ),
    )

    code = fix_main(["--result", str(run_dir), "--repo", str(repo)])
    err = capsys.readouterr().err

    assert code == 0  # a recorded skip is still a produced outcome
    assert f"skipped {CAND_ID}: no longer in result.json" in err
    assert "None" not in err
