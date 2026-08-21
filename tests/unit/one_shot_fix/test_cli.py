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


def test_engine_dispatch_routes_fix_to_the_subcommand(
    run, capsys: pytest.CaptureFixture
) -> None:
    run_dir, repo = run
    code = engine_main(
        ["fix", "--result", str(run_dir), "--repo", str(repo), "--top-n", "zero"]
    )
    assert code == 2
    assert "--top-n" in capsys.readouterr().err
