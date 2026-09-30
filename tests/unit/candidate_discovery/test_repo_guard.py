"""Unit tests for `spotlights_engine.candidate_discovery.repo_guard`."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from spotlights_engine.candidate_discovery.errors import DiscoveryMutationError
from spotlights_engine.candidate_discovery.repo_guard import RepoGuard


def _init_git_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.local"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "a.py").write_text("a = 1\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "init"], check=True)


def test_git_mode_clean_does_not_raise(tmp_path):
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with guard.observe():
        pass


def test_git_mode_new_untracked_file_raises(tmp_path):
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / "new.py").write_text("x = 2\n")


def test_git_mode_staged_change_raises(tmp_path):
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / "a.py").write_text("changed\n")
            subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)


def test_git_mode_already_dirty_tracked_file_change_raises(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / "a.py").write_text("dirty before guard\n")
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError) as exc:
        with guard.observe():
            (tmp_path / "a.py").write_text("changed again while observed\n")
    assert exc.value.context["diff"]["manifest"]["changed"] == ["a.py"]


def test_git_mode_already_untracked_file_change_raises(tmp_path):
    _init_git_repo(tmp_path)
    (tmp_path / "scratch.py").write_text("untracked before guard\n")
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError) as exc:
        with guard.observe():
            (tmp_path / "scratch.py").write_text("changed while observed\n")
    assert exc.value.context["diff"]["manifest"]["changed"] == ["scratch.py"]


def test_non_git_mode_touch_raises(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / "a.py").write_text("a = 2\n")


def test_non_git_mode_rename_raises(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / "a.py").rename(tmp_path / "b.py")


def test_non_git_mode_new_file_raises(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / "new.py").write_text("hi\n")


def test_non_git_mode_clean_does_not_raise(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with guard.observe():
        # touch a read but no write
        (tmp_path / "a.py").read_text()


def test_non_git_mode_skips_pycache(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "stale.pyc").write_text("garbage")
    guard = RepoGuard(tmp_path)
    with guard.observe():
        # Writing inside __pycache__ should be invisible to the guard.
        (pycache / "new.pyc").write_text("new garbage")


def test_non_git_mode_skips_finder_metadata(tmp_path):
    """Finder metadata dropped into the tree mid-run is not a mutation."""
    (tmp_path / "a.py").write_text("a = 1\n")
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("b = 2\n")
    guard = RepoGuard(tmp_path)
    with guard.observe():
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")
        (sub / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")


def test_git_mode_skips_finder_metadata(tmp_path):
    """Same in git mode, where the porcelain status is part of the snapshot.

    A target repo that does not ignore `.DS_Store` reports it as untracked, so
    the status bytes must be filtered too — not just the manifest.
    """
    _init_git_repo(tmp_path)
    sub = tmp_path / "pkg"
    sub.mkdir()
    (sub / "b.py").write_text("b = 2\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-q", "-m", "pkg"], check=True)
    guard = RepoGuard(tmp_path)
    with guard.observe():
        (tmp_path / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")
        (sub / ".DS_Store").write_bytes(b"\x00\x00\x00\x01")


def test_git_mode_real_mutation_still_raises_alongside_metadata(tmp_path):
    """Filtering metadata must not mask a real write in the same invocation."""
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            (tmp_path / ".DS_Store").write_bytes(b"\x00")
            (tmp_path / "a.py").write_text("a = 999\n")


def test_git_mode_renamed_file_still_raises(tmp_path):
    """A staged rename survives status filtering (its origin path is paired)."""
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            subprocess.run(["git", "-C", str(tmp_path), "mv", "a.py", "b.py"], check=True)


def test_non_git_mode_mtime_only_touch_does_not_raise(tmp_path):
    """Rewriting identical bytes (or bumping mtime) is not a mutation.

    This is the regression the content-hash manifest exists for: on macOS,
    Spotlight / Time Machine / Finder touch files during an agent run, and an
    mtime-keyed manifest reported those as repo mutations.
    """
    p = tmp_path / "a.py"
    p.write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with guard.observe():
        p.write_text("a = 1\n")  # same content, new mtime
        os.utime(p, (0, 0))  # and an explicit mtime bump


@pytest.mark.skipif(
    not hasattr(os, "setxattr"), reason="xattr APIs unavailable on this platform"
)
def test_non_git_mode_xattr_change_does_not_raise(tmp_path):
    """Extended-attribute churn is deliberately invisible to the guard.

    Inverted from the previous contract: the manifest used to include an xattr
    digest, so this raised. It now keys on content, because macOS metadata
    churn made the xattr digest a source of false mutations.
    """
    p = tmp_path / "a.py"
    p.write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    try:
        os.setxattr(p, "user.test", b"v1", follow_symlinks=False)
    except OSError:
        pytest.skip("filesystem does not support user xattrs")
    with guard.observe():
        os.setxattr(p, "user.test", b"v2", follow_symlinks=False)
