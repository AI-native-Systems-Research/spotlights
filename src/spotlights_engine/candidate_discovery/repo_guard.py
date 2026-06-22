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
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.candidate_discovery.errors import DiscoveryMutationError

_EXCLUDE_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".venv",
        # Some local agent integrations persist per-project state under the
        # target cwd even in planning/read-only modes. Treat that tool-owned
        # state like cache metadata so the guard still protects source files
        # without failing on an otherwise read-only agent invocation.
        ".omc",
    }
)


@dataclass(frozen=True)
class _GitSnapshot:
    status: bytes
    manifest: dict[str, tuple[int, int, str, str]]


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
        status = self._git_status()
        return _GitSnapshot(
            status=status,
            # In git repos, `git status` already proves whether the terminal
            # source tree changed. Hash only paths that were dirty/untracked at
            # snapshot time so pre-existing scratch files are still protected
            # without re-hashing large clean repositories after every agent call.
            manifest=self._manifest_snapshot(_porcelain_paths(status)),
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
                "--",
                ".",
                *[f":(exclude){name}" for name in sorted(_EXCLUDE_DIRS)],
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout

    def _manifest_snapshot(
        self, paths: Iterable[str] | None = None
    ) -> dict[str, tuple[int, int, str, str]]:
        manifest: dict[str, tuple[int, int, str, str]] = {}
        if paths is not None:
            for rel in sorted(set(paths)):
                if not rel or _is_excluded_rel(rel):
                    continue
                full = self._repo_path / rel
                if full.is_dir():
                    self._add_tree_to_manifest(full, manifest)
                else:
                    self._add_file_to_manifest(full, rel, manifest)
            return manifest

        self._add_tree_to_manifest(self._repo_path, manifest)
        return manifest

    def _add_tree_to_manifest(
        self,
        tree_root: Path,
        manifest: dict[str, tuple[int, int, str, str]],
    ) -> None:
        for root, dirs, files in os.walk(tree_root):
            dirs[:] = [d for d in dirs if d not in _EXCLUDE_DIRS]
            for name in files:
                full = Path(root) / name
                rel = full.relative_to(self._repo_path).as_posix()
                self._add_file_to_manifest(full, rel, manifest)

    @staticmethod
    def _add_file_to_manifest(
        full: Path,
        rel: str,
        manifest: dict[str, tuple[int, int, str, str]],
    ) -> None:
        try:
            st = full.lstat()
        except OSError:
            manifest[rel] = (0, 0, "missing", "missing")
            return
        manifest[rel] = (
            st.st_mtime_ns,
            st.st_size,
            _xattr_digest(full),
            _content_digest(full),
        )

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
            return {
                "before_bytes": before.decode("utf-8", "replace"),
                "after_bytes": after.decode("utf-8", "replace"),
            }
        return _manifest_diff(before, after)


def _manifest_diff(
    before: dict[str, tuple[int, int, str, str]],
    after: dict[str, tuple[int, int, str, str]],
) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    changed = sorted(k for k in before_keys & after_keys if before[k] != after[k])
    return {"added": added, "removed": removed, "changed": changed}


def _porcelain_paths(status: bytes) -> list[str]:
    paths: list[str] = []
    fields = status.split(b"\0")
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if not field:
            continue
        if len(field) < 4 or field[2:3] != b" ":
            continue
        code = field[:2]
        path = field[3:].decode("utf-8", "replace")
        paths.append(path)
        if code[:1] in {b"R", b"C"} or code[1:2] in {b"R", b"C"}:
            if i < len(fields) and fields[i]:
                paths.append(fields[i].decode("utf-8", "replace"))
            i += 1
    return paths


def _is_excluded_rel(rel: str) -> bool:
    parts = Path(rel).parts
    return any(part in _EXCLUDE_DIRS for part in parts)


def _content_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError:
        return "content-unavailable"
    return hasher.hexdigest()


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
