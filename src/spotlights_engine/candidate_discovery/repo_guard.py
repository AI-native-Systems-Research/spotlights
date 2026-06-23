"""§6.1 target-repo mutation detector.

Snapshots `repo_path` before each subprocess call and re-snapshots afterwards;
any difference raises `DiscoveryMutationError`. Coverage is limited to
`repo_path` — the orchestrator does not detect writes outside this subtree.
`--permission-mode plan` (Claude) and `--sandbox read-only` (Codex) are the
primary defenses there.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.candidate_discovery.errors import DiscoveryMutationError

_EXCLUDE_DIRS = frozenset({".git", "__pycache__", ".venv"})


@dataclass(frozen=True)
class _GitSnapshot:
    status: bytes
    manifest: dict[str, tuple[int, int, str]]


class RepoGuard:
    def __init__(self, repo_path: Path) -> None:
        self._repo_path = repo_path
        self._use_git = (repo_path / ".git").is_dir()

    @contextmanager
    def observe(self) -> Iterator[None]:
        before = self._snapshot()
        try:
            yield
        finally:
            after = self._snapshot()
            if before != after:
                raise DiscoveryMutationError(
                    "target repo mutated during agent invocation",
                    repo_path=str(self._repo_path),
                    diff=self._diff(before, after),
                )

    def _snapshot(self):
        if self._use_git:
            return self._git_snapshot()
        return self._manifest_snapshot()

    def _git_snapshot(self) -> _GitSnapshot:
        return _GitSnapshot(
            status=self._git_status(),
            manifest=self._manifest_snapshot(),
        )

    def _git_status(self) -> bytes:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(self._repo_path),
                "-c",
                "core.quotepath=off",
                "status",
                "--porcelain=v1",
                "-z",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def _manifest_snapshot(self) -> dict[str, tuple[int, int, str]]:
        manifest: dict[str, tuple[int, int, str]] = {}
        for root, dirs, files in os.walk(self._repo_path):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(self._repo_path).as_posix()
                try:
                    st = full.lstat()
                except OSError:
                    manifest[rel] = (0, 0, "missing")
                    continue
                manifest[rel] = (st.st_mtime_ns, st.st_size, _xattr_digest(full))
        return manifest

    def _diff(self, before, after):
        if isinstance(before, _GitSnapshot):
            return {
                "git_status": {
                    "before_bytes": before.status.decode("utf-8", "replace"),
                    "after_bytes": after.status.decode("utf-8", "replace"),
                },
                "manifest": _manifest_diff(before.manifest, after.manifest),
            }
        if isinstance(before, bytes):
            return {"before_bytes": before.decode("utf-8", "replace"),
                    "after_bytes": after.decode("utf-8", "replace")}
        return _manifest_diff(before, after)


def _manifest_diff(
    before: dict[str, tuple[int, int, str]],
    after: dict[str, tuple[int, int, str]],
) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def _xattr_digest(path: Path) -> str:
    if not hasattr(os, "listxattr"):
        return "no-xattr-api"
    try:
        names = sorted(os.listxattr(path, follow_symlinks=False))
    except OSError:
        return "xattr-unavailable"
    if not names:
        return "no-xattrs"
    hasher = hashlib.sha256()
    for name in names:
        try:
            value = os.getxattr(path, name, follow_symlinks=False)
        except OSError:
            value = b""
        hasher.update(name.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(value)
        hasher.update(b"\x00")
    return hasher.hexdigest()


__all__ = ["RepoGuard"]
