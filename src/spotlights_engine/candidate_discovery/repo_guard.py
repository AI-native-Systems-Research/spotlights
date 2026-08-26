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

# Finder/Spotlight write these into directories they touch, including while an
# agent is mid-invocation. They are OS metadata, not target-repo content, so
# their appearance must not read as a mutation.
_EXCLUDE_FILES = frozenset({".DS_Store", ".localized"})


@dataclass(frozen=True)
class _GitSnapshot:
    status: bytes
    manifest: dict[str, tuple[int, str]]


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
        return _filter_status(self._git_status_raw())

    def _git_status_raw(self) -> bytes:
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

    def _manifest_snapshot(self) -> dict[str, tuple[int, str]]:
        manifest: dict[str, tuple[int, str]] = {}
        for root, dirs, files in os.walk(self._repo_path):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for name in files:
                if name in _EXCLUDE_FILES:
                    continue
                full = Path(root) / name
                rel = full.relative_to(self._repo_path).as_posix()
                try:
                    st = full.lstat()
                except OSError:
                    manifest[rel] = (0, "missing")
                    continue
                manifest[rel] = (st.st_size, _content_digest(full))
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


def _filter_status(raw: bytes) -> bytes:
    """Drop `_EXCLUDE_FILES` entries from `git status --porcelain=v1 -z` output.

    The manifest snapshot skips OS metadata by name, but in git mode the status
    bytes are part of the snapshot too, so an untracked `.DS_Store` appearing
    mid-invocation would still read as a mutation. Records are NUL-terminated
    `XY <path>`; rename/copy records are followed by a second NUL-terminated
    origin path, which is dropped with its record.
    """
    fields = raw.split(b"\x00")
    kept: list[bytes] = []
    i = 0
    while i < len(fields):
        record = fields[i]
        if not record:
            i += 1
            continue
        # "XY " prefix, then the path.
        code, _, path = record[:2], record[2:3], record[3:]
        has_origin = b"R" in code or b"C" in code
        excluded = Path(path.decode("utf-8", "replace")).name in _EXCLUDE_FILES
        if not excluded:
            kept.append(record)
            if has_origin and i + 1 < len(fields):
                kept.append(fields[i + 1])
        i += 2 if has_origin else 1
    if not kept:
        return b""
    return b"\x00".join(kept) + b"\x00"


def _manifest_diff(
    before: dict[str, tuple[int, str]],
    after: dict[str, tuple[int, str]],
) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def _content_digest(path: Path) -> str:
    """SHA-256 of the file's contents. Detects real mutation (write) while
    ignoring macOS metadata / mtime / xattr churn from Spotlight, Time Machine,
    Finder, etc., which repeatedly touch files during agent runs. Symlinks
    hash their target-path bytes so link redirects still register as changes.
    """
    hasher = hashlib.sha256()
    try:
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8", "replace")
            hasher.update(b"symlink:")
            hasher.update(target)
            return hasher.hexdigest()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    except OSError:
        return "unreadable"
    return hasher.hexdigest()


__all__ = ["RepoGuard"]
