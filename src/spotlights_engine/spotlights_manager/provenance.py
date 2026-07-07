"""Run provenance capture: what's needed to reproduce a Spotlights run.

Captured once at first run start and pinned into the internal manifest's
`provenance` object; resume reads it back rather than recomputing, so a
resume from a different working tree can't silently rewrite provenance.

All values degrade to empty strings when git is unavailable (non-git target,
wheel-installed engine) — the run manifest notes the gap instead of blocking
local analysis, unless the caller opts into strict provenance.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def _git_output(args: list[str], cwd: Path) -> str:
    """Run a git query; empty string on any failure (missing git, non-repo)."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def resolve_commit_sha(path: Path) -> str:
    return _git_output(["rev-parse", "HEAD"], cwd=path)


def resolve_repo_url(repo_path: Path, explicit: str | None = None) -> str:
    """Prefer the caller-supplied URL; fall back to `git remote get-url origin`."""
    if explicit:
        return explicit
    return _git_output(["remote", "get-url", "origin"], cwd=repo_path)


def spotlights_commit_sha() -> str:
    """HEAD of *this* engine checkout. The tool changes between runs, so the
    target commit alone does not reproduce a run. Empty when the engine is not
    running from a git checkout (e.g. wheel install)."""
    return resolve_commit_sha(Path(__file__).resolve().parent)


def collect_provenance(
    *,
    repo_path: Path,
    repo_url: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Assemble the provenance object pinned into the internal manifest."""
    provenance: dict[str, Any] = {
        "repo_url": resolve_repo_url(repo_path, repo_url),
        "target_commit_sha": resolve_commit_sha(repo_path),
        "spotlights_commit_sha": spotlights_commit_sha(),
    }
    if run_id:
        provenance["run_id"] = run_id
    return provenance


__all__ = [
    "collect_provenance",
    "resolve_commit_sha",
    "resolve_repo_url",
    "spotlights_commit_sha",
]
