"""`spotlights-engine init` installs the fix-candidate skill."""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.init_skills import _plan_install_items, install_skills

REL = ".claude/commands/spotlights-fix-candidate/SKILL.md"


def test_fix_candidate_is_in_the_install_plan() -> None:
    rel_paths = {item.rel_path for item in _plan_install_items()}
    assert REL in rel_paths


def test_init_writes_the_skill_into_claude_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert install_skills(scope="project") == 0
    installed = tmp_path / REL
    assert installed.is_file()
    text = installed.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "name: fix-candidate" in text
    assert "--print-prompt" in text


def test_the_skill_collects_the_patch_against_the_base_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The in-session collect step must diff the base commit, like `collect_patch`.

    A bare `git diff` is index-vs-worktree, and `git add -N .` stages a
    deletion *fully* — so a bare diff drops a deleted file and the delete-half
    of a rename, silently. The interactive path would then produce a different,
    incomplete patch from `spotlights-engine fix` for the same candidate.
    """
    monkeypatch.chdir(tmp_path)
    assert install_skills(scope="project") == 0
    text = (tmp_path / REL).read_text(encoding="utf-8")
    assert 'git diff "<BASE>" >>' in text
    assert "git diff >>" not in text
