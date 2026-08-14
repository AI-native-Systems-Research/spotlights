"""Live-target validation tests (plan §10.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.prep_evolve.errors import StalenessError
from spotlights_engine.prep_evolve.resolve import (
    load_result,
    resolve_candidate,
    resolve_candidates,
    resolve_module_run,
)
from spotlights_engine.prep_evolve.validate_target import (
    capture_revision,
    validate_candidate_target,
    validate_scope_file,
)

from . import _fixtures as fx


def _candidate(tmp_path: Path):
    loaded = load_result(fx.write_result(tmp_path))
    run = resolve_module_run(loaded, "v1/attention")
    candidates = resolve_candidates(run, "v1/attention")
    return resolve_candidate(candidates, "cand-v1_attention-0002")


def test_valid_candidate(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == fx.CAND_START
    assert validated.line_end == fx.CAND_END
    assert len(validated.source_excerpt_sha256) == 64


def test_excerpt_hash_stable(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    a = validate_candidate_target(repo, cand)
    b = validate_candidate_target(repo, cand)
    assert a.source_excerpt_sha256 == b.source_excerpt_sha256


def test_path_escape_rejected(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].file = "../outside.py"
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_absolute_candidate_file_rejected(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].file = str((repo / fx.CAND_FILE).resolve())
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_relative_repo_path_validates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    monkeypatch.chdir(tmp_path)
    validated = validate_candidate_target(Path("repo"), cand)
    assert validated.line_start == fx.CAND_START


def test_line_range_out_of_bounds(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].line_end = 100000
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_staleness_symbol_missing(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    # Overwrite the candidate file so the recorded symbol is gone.
    target = repo / fx.CAND_FILE
    target.write_text("\n".join(f"# line {i}" for i in range(1, 12)) + "\n")
    cand = _candidate(tmp_path)
    with pytest.raises(StalenessError) as exc:
        validate_candidate_target(repo, cand)
    assert "stale" in str(exc.value)


def test_qualified_symbol_class_far_from_method(tmp_path: Path) -> None:
    """A `Class.method` symbol whose recorded range covers only the method body
    must validate even when the class declaration is far above the range. The
    container (class) token lives at the `class` line, not near the method, so
    requiring it inside the ±window is a false-positive staleness error."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = ["class LRUCachePolicy:"]  # line 1: class decl, far from the method
    body += [f"    # filler {i}" for i in range(2, 40)]  # lines 2..39
    body += [
        "    def evict(self, n):  # line 40",  # recorded range starts here
        "        candidates = []",
        "        return candidates",  # line 42
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "LRUCachePolicy.evict"
    cand.locations[0].spans[0].line_start = 40
    cand.locations[0].spans[0].line_end = 42

    validated = validate_candidate_target(repo, cand)
    assert validated.line_start == 40
    assert validated.line_end == 42


def test_qualified_symbol_renamed_container_still_stale(tmp_path: Path) -> None:
    """If the container class is genuinely gone from the file, staleness must
    still fire (the relaxation is file-wide, not unconditional)."""
    repo = fx.make_repo(tmp_path)
    target = repo / fx.CAND_FILE
    body = [f"    # filler {i}" for i in range(1, 40)]
    body += [
        "    def evict(self, n):  # line 40",
        "        return []",
    ]
    target.write_text("\n".join(body) + "\n", encoding="utf-8")

    cand = _candidate(tmp_path)
    cand.locations[0].spans[0].symbol = "LRUCachePolicy.evict"
    cand.locations[0].spans[0].line_start = 40
    cand.locations[0].spans[0].line_end = 41

    with pytest.raises(StalenessError) as exc:
        validate_candidate_target(repo, cand)
    assert "LRUCachePolicy" in str(exc.value)


def test_missing_file(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    (repo / fx.CAND_FILE).unlink()
    cand = _candidate(tmp_path)
    with pytest.raises(StalenessError):
        validate_candidate_target(repo, cand)


def test_scope_file_validation(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path)
    validate_scope_file(repo, fx.MAIN_FILE_EXTRA)
    with pytest.raises(StalenessError):
        validate_scope_file(repo, "does/not/exist.py")
    with pytest.raises(StalenessError):
        validate_scope_file(repo, "../escape.py")


def test_git_revision_capture(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=True)
    rev = capture_revision(repo, "2026-06-16T00:00:00+00:00")
    assert rev.git_commit is not None
    assert rev.dirty is False
    assert rev.captured_at == "2026-06-16T00:00:00+00:00"


def test_git_revision_non_repo(tmp_path: Path) -> None:
    repo = fx.make_repo(tmp_path, git=False)
    rev = capture_revision(repo, "t")
    assert rev.git_commit is None
    assert rev.dirty is None
