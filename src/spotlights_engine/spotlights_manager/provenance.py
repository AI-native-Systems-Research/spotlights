"""Run provenance capture: what's needed to reproduce a Spotlights run.

Captured once at first run start and pinned into the internal manifest's
`provenance` object; resume reads it back rather than recomputing, so a
resume from a different working tree can't silently rewrite provenance.

All values degrade to empty strings when git is unavailable (non-git target,
wheel-installed engine) — the run manifest notes the gap instead of blocking
local analysis, unless the caller opts into strict provenance.
"""

from __future__ import annotations

import json
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

#: Distribution that ships this module, as named in `pyproject.toml`. Used to
#: recover the engine's commit from install metadata when there is no checkout.
_DIST_NAME = "spotlights-engine"


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


def _installed_vcs_commit() -> str:
    """The commit the installer resolved this distribution from (PEP 610).

    `uv tool install git+https://.../spotlights.git` — what `install.sh` runs —
    pins the exact commit in the dist-info's `direct_url.json`, but installs a
    plain directory of files rather than a checkout, so `git rev-parse` has
    nothing to answer from. A local-path install (`uv tool install .`) records
    `dir_info` with no commit, and a plain wheel records nothing at all; both
    stay empty.
    """
    try:
        raw = metadata.distribution(_DIST_NAME).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return ""
    try:
        info = json.loads(raw or "")
    except ValueError:
        return ""
    if not isinstance(info, dict):
        return ""
    vcs_info = info.get("vcs_info")
    if not isinstance(vcs_info, dict) or vcs_info.get("vcs") != "git":
        return ""
    return str(vcs_info.get("commit_id") or "")


def spotlights_commit_sha() -> str:
    """HEAD of *this* engine checkout, or the commit it was installed from. The
    tool changes between runs, so the target commit alone does not reproduce a
    run.

    Git first: a clone or editable install is the live tree, and its HEAD beats
    whatever an older install recorded. Empty only when neither source knows —
    a wheel or local-path install with no checkout behind it.
    """
    return resolve_commit_sha(Path(__file__).resolve().parent) or _installed_vcs_commit()


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
