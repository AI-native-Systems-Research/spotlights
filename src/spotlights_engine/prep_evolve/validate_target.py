"""Live-repo validation for prep-evolve (the staleness gate).

This is the only module that reads the target repo. It validates the selected
candidate (and any scope-only main-file targets) against the live tree before
any bundle is written, and captures the repo git revision. Keeping this
separate from `extract.py` lets the resolver/spec tests stay unit-level while
making the staleness gate explicit (plan §4).
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.prep_evolve.errors import (
    RepoResolutionError,
    StalenessError,
)
from spotlights_engine.prep_evolve.spec import SourceRevision
from spotlights_engine.schemas.candidate import Candidate

# How far above/below the recorded range to look for the recorded symbol when
# checking staleness. result.json does not carry the original excerpt, so the
# check is intentionally heuristic.
_SYMBOL_WINDOW = 5


@dataclass
class ValidatedCandidate:
    """The validated, live-repo-confirmed view of the candidate target."""

    line_start: int
    line_end: int
    source_excerpt_sha256: str


def _resolve_inside(repo_path: Path, rel_file: str) -> Path:
    """Resolve `rel_file` inside `repo_path`, rejecting path escapes."""
    rel_path = Path(rel_file)
    if rel_path.is_absolute():
        raise StalenessError(f"target file {rel_file!r} must be repo-relative, not absolute")

    repo_root = repo_path.resolve()
    resolved = (repo_root / rel_path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise StalenessError(
            f"target file {rel_file!r} resolves outside the repo ({resolved} not under {repo_root})"
        )
    return resolved


def validate_scope_file(repo_path: Path, rel_file: str) -> None:
    """Validate a scope-only (whole-file) target: containment + existence."""
    resolved = _resolve_inside(repo_path, rel_file)
    if not resolved.is_file():
        raise StalenessError(f"scope target file does not exist in repo: {rel_file}")


def validate_candidate_target(
    repo_path: Path,
    candidate: Candidate,
) -> ValidatedCandidate:
    """Validate the selected candidate against the live repo.

    Checks path containment, file existence, line bounds, and a heuristic
    staleness gate (the recorded symbol must appear in/around the recorded
    range). Returns the validated range and the excerpt hash.
    """
    resolved = _resolve_inside(repo_path, candidate.file)
    if not resolved.is_file():
        raise StalenessError(f"candidate file does not exist in repo: {candidate.file}")

    lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
    n = len(lines)
    start, end = candidate.line_start, candidate.line_end
    if start < 1 or end < start or end > n:
        raise StalenessError(
            f"candidate line range [{start}, {end}] is out of bounds for "
            f"{candidate.file} ({n} lines). result.json is stale relative to "
            f"the repo; re-run spotlights or correct the selected result."
        )

    # 1-indexed inclusive slice.
    excerpt = "\n".join(lines[start - 1 : end])
    digest = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()

    # Staleness heuristic: the recorded symbol should appear within the slice
    # (plus a small window) so we don't wrap an EVOLVE-BLOCK around the wrong
    # code.
    win_start = max(0, start - 1 - _SYMBOL_WINDOW)
    win_end = min(n, end + _SYMBOL_WINDOW)
    window_text = "\n".join(lines[win_start:win_end])
    if candidate.symbol and candidate.symbol not in window_text:
        raise StalenessError(
            f"recorded symbol {candidate.symbol!r} not found near lines "
            f"[{start}, {end}] of {candidate.file}. result.json is stale "
            f"relative to the repo; re-run spotlights or correct the selected "
            f"result."
        )

    return ValidatedCandidate(
        line_start=start,
        line_end=end,
        source_excerpt_sha256=digest,
    )


def capture_revision(repo_path: Path, captured_at: str) -> SourceRevision:
    """Best-effort git revision capture. `git_commit`/`dirty` are `None` when
    `repo_path` is not a git checkout or git is unavailable."""
    commit: str | None = None
    dirty: bool | None = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if head.returncode == 0:
            commit = head.stdout.strip() or None
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if status.returncode == 0:
                dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None

    return SourceRevision(git_commit=commit, dirty=dirty, captured_at=captured_at)


def ensure_repo_dir(repo_path: Path) -> None:
    """Re-assert that `repo_path` is an existing directory (defensive)."""
    if not repo_path.is_dir():
        raise RepoResolutionError(f"repo path is not a directory: {repo_path}")


__all__ = [
    "ValidatedCandidate",
    "capture_revision",
    "ensure_repo_dir",
    "validate_candidate_target",
    "validate_scope_file",
]
