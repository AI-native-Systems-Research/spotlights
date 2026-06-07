"""Tests for `spotlights-engine init` (skill installer)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from spotlights_engine import init_skills


@pytest.fixture
def fake_bundle(monkeypatch, tmp_path):
    """Stub the bundled-templates lookup with a temp directory we control.

    Returns the directory; callers populate it with `<slug>.md` files to
    simulate the package's `_templates/commands/` payload.
    """
    bundle = tmp_path / "_bundle"
    bundle.mkdir()

    monkeypatch.setattr(init_skills, "_bundled_templates", lambda: bundle)
    return bundle


@pytest.fixture
def project_root(monkeypatch, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    monkeypatch.chdir(root)
    return root


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestInstallSkills:
    def test_installs_with_prefix_and_writes_manifest(self, fake_bundle, project_root):
        body = "# objective skill body\n"
        (fake_bundle / "objective-setting.md").write_text(body, encoding="utf-8")

        rc = init_skills.install_skills(scope="project")
        assert rc == 0

        installed = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        assert installed.is_file()
        assert installed.read_text(encoding="utf-8") == body

        manifest_path = project_root / ".spotlights" / "manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["integration"] == "spotlights"
        rel = ".claude/commands/spotlights-objective-setting.md"
        assert manifest["files"] == {rel: _sha256_text(body)}

    def test_skips_existing_without_force(self, fake_bundle, project_root):
        (fake_bundle / "objective-setting.md").write_text("bundled\n", encoding="utf-8")
        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        target.parent.mkdir(parents=True)
        target.write_text("user wrote this\n", encoding="utf-8")

        rc = init_skills.install_skills(scope="project")
        assert rc == 0
        assert target.read_text(encoding="utf-8") == "user wrote this\n"

    def test_force_overwrites_unchanged_managed_file(self, fake_bundle, project_root):
        v1 = "v1 body\n"
        (fake_bundle / "objective-setting.md").write_text(v1, encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        v2 = "v2 body — upgraded\n"
        (fake_bundle / "objective-setting.md").write_text(v2, encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        assert target.read_text(encoding="utf-8") == v2

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        rel = ".claude/commands/spotlights-objective-setting.md"
        assert manifest["files"][rel] == _sha256_text(v2)

    def test_force_preserves_user_edited_managed_file(self, fake_bundle, project_root):
        v1 = "v1 body\n"
        (fake_bundle / "objective-setting.md").write_text(v1, encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        edited = "user edited this\n"
        target.write_text(edited, encoding="utf-8")

        v2 = "v2 body\n"
        (fake_bundle / "objective-setting.md").write_text(v2, encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        # User's edit must survive.
        assert target.read_text(encoding="utf-8") == edited
        # Manifest still records the original (pre-edit) hash so a future
        # `--force` after the user reverts will once again be safe to overwrite.
        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        rel = ".claude/commands/spotlights-objective-setting.md"
        assert manifest["files"][rel] == _sha256_text(v1)

    def test_force_does_not_touch_unmanaged_existing_file(self, fake_bundle, project_root):
        # File pre-exists with no manifest entry. Even with --force, install_skills
        # must not touch it (it's not ours to overwrite).
        (fake_bundle / "objective-setting.md").write_text("bundled\n", encoding="utf-8")
        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        target.parent.mkdir(parents=True)
        target.write_text("user's own\n", encoding="utf-8")

        assert init_skills.install_skills(scope="project", force=True) == 0
        assert target.read_text(encoding="utf-8") == "user's own\n"

    def test_user_scope_writes_to_home(self, fake_bundle, monkeypatch, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        # Path.home() reads HOME on POSIX; on darwin/linux this is enough.

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        rc = init_skills.install_skills(scope="user")
        assert rc == 0

        assert (home / ".claude" / "commands" / "spotlights-objective-setting.md").is_file()
        assert (home / ".spotlights" / "manifest.json").is_file()

    def test_empty_bundle_returns_error(self, fake_bundle, project_root):
        # Bundle directory exists but contains no .md files.
        rc = init_skills.install_skills(scope="project")
        assert rc == 1


class TestCliEntryPoint:
    def test_init_routed_via_main(self, fake_bundle, project_root):
        from spotlights_engine.cli import main as engine_main

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        rc = engine_main(["init"])
        assert rc == 0
        assert (
            project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        ).is_file()

    def test_init_force_flag_parsed(self, fake_bundle, project_root):
        from spotlights_engine.init_skills import main as init_main

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_main(["--force"]) == 0
