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


def test_git_mode_skips_local_agent_state(tmp_path):
    _init_git_repo(tmp_path)
    guard = RepoGuard(tmp_path)
    with guard.observe():
        omc = tmp_path / ".omc"
        omc.mkdir()
        (omc / "project-memory.json").write_text("{}\n")


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


def test_non_git_mode_skips_local_agent_state(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    with guard.observe():
        omc = tmp_path / ".omc"
        omc.mkdir()
        (omc / "project-memory.json").write_text("{}\n")


@pytest.mark.skipif(
    not hasattr(os, "setxattr"), reason="xattr APIs unavailable on this platform"
)
def test_non_git_mode_xattr_change_raises(tmp_path):
    p = tmp_path / "a.py"
    p.write_text("a = 1\n")
    guard = RepoGuard(tmp_path)
    try:
        os.setxattr(p, "user.test", b"v1", follow_symlinks=False)
    except OSError:
        pytest.skip("filesystem does not support user xattrs")
    with pytest.raises(DiscoveryMutationError):
        with guard.observe():
            os.setxattr(p, "user.test", b"v2", follow_symlinks=False)
