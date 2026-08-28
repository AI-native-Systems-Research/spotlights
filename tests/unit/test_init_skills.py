"""Tests for `spotlights-engine init` (skill installer)."""

from __future__ import annotations

import hashlib
import json

import pytest

from spotlights_engine import init_skills


@pytest.fixture(autouse=True)
def sandboxed_home(monkeypatch, tmp_path):
    """Keep user-scope installs off the real home.

    `install_skills` defaults to `scope="user"`, so a test that forgets to pass
    a scope would otherwise write into the developer's own ~/.claude/commands/.
    """
    home = tmp_path / "sandbox-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    # A developer's own CLAUDE_CONFIG_DIR would otherwise redirect user-scope
    # installs mid-test; tests that exercise it set it explicitly.
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home


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

    def test_plain_rerun_preserves_manifest_ownership(self, fake_bundle, project_root):
        """A no-op re-run must not forget what it owns.

        The manifest is the only record distinguishing our files from the user's.
        Wiping it on a plain re-run left a later --force unable to upgrade anything.
        """
        body = "v1 body\n"
        (fake_bundle / "objective-setting.md").write_text(body, encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        manifest_path = project_root / ".spotlights" / "manifest.json"
        rel = ".claude/commands/spotlights-objective-setting.md"
        first = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
        assert first == {rel: _sha256_text(body)}

        # Plain re-run: every file already exists, so nothing is written...
        assert init_skills.install_skills(scope="project") == 0
        # ...and the ownership record survives unchanged.
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["files"] == first

    def test_force_upgrades_after_a_plain_rerun(self, fake_bundle, project_root):
        """--force must still upgrade when a no-op re-run happened in between."""
        (fake_bundle / "objective-setting.md").write_text("v1 body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0
        assert init_skills.install_skills(scope="project") == 0  # the no-op re-run

        v2 = "v2 body — upgraded\n"
        (fake_bundle / "objective-setting.md").write_text(v2, encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        assert target.read_text(encoding="utf-8") == v2

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

    def test_default_scope_is_user(self, fake_bundle, monkeypatch, tmp_path):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        cwd = tmp_path / "somewhere-else"
        cwd.mkdir()
        monkeypatch.chdir(cwd)

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_skills.install_skills() == 0
        assert init_skills.build_argparser().parse_args([]).scope == "user"

        assert (home / ".claude" / "commands" / "spotlights-objective-setting.md").is_file()
        assert not (cwd / ".claude").exists()

    def test_user_scope_honours_claude_config_dir(
        self, fake_bundle, monkeypatch, tmp_path
    ):
        """CLAUDE_CONFIG_DIR moves Claude Code's whole ~/.claude, so install there.

        Writing to $HOME/.claude with that variable set put the skills where
        Claude Code never looks, while still reporting success.
        """
        home = tmp_path / "home"
        home.mkdir()
        relocated = tmp_path / "relocated-config"
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(relocated))

        body = "body\n"
        (fake_bundle / "objective-setting.md").write_text(body, encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0

        # Commands sit directly under the config dir, not under a nested .claude/.
        assert (relocated / "commands" / "spotlights-objective-setting.md").is_file()
        assert not (home / ".claude").exists()

        # The manifest lives beside what it describes, keyed relative to that root.
        manifest = json.loads(
            (relocated / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["files"] == {
            "commands/spotlights-objective-setting.md": _sha256_text(body)
        }

    def test_claude_config_dir_ignored_for_project_scope(
        self, fake_bundle, project_root, monkeypatch, tmp_path
    ):
        """A project's own .claude/commands/ is read from the project regardless."""
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "relocated"))
        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")

        assert init_skills.install_skills(scope="project") == 0
        assert (
            project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        ).is_file()

    def test_blank_claude_config_dir_falls_back_to_home(
        self, fake_bundle, monkeypatch, tmp_path
    ):
        """An empty/whitespace value is not a path; treat it as unset."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "   ")

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0
        assert (home / ".claude" / "commands" / "spotlights-objective-setting.md").is_file()

    @pytest.mark.parametrize(
        "payload",
        ['{"files": null}', '["not", "a", "dict"]', '{"files": {"a": 3}}', "null"],
    )
    def test_malformed_manifest_does_not_crash(
        self, fake_bundle, project_root, capsys, payload
    ):
        """A wrong-shaped manifest degrades to "no manifest", never a traceback.

        `_read_manifest` runs on every install now, so a payload that parses as
        JSON but isn't the expected shape used to crash a plain `init`.
        """
        manifest_dir = project_root / ".spotlights"
        manifest_dir.mkdir()
        (manifest_dir / "manifest.json").write_text(payload, encoding="utf-8")
        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")

        assert init_skills.install_skills(scope="project") == 0
        assert (
            project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        ).is_file()
        assert "unusable manifest" in capsys.readouterr().err

    def test_empty_bundle_returns_error(self, fake_bundle, project_root):
        # Bundle directory exists but contains no .md files.
        rc = init_skills.install_skills(scope="project")
        assert rc == 1

    def test_installs_directory_skill_with_structure(self, fake_bundle, project_root):
        skill = fake_bundle / "share-candidates"
        (skill / "tests").mkdir(parents=True)
        (skill / "SKILL.md").write_text("# share candidates\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("print('hi')\n", encoding="utf-8")
        nested = skill / "tests" / "test_build_bundle.py"
        nested.write_text("def test_x():\n    pass\n", encoding="utf-8")

        rc = init_skills.install_skills(scope="project")
        assert rc == 0

        base = project_root / ".claude" / "commands" / "spotlights-share-candidates"
        assert (base / "SKILL.md").read_text(encoding="utf-8") == "# share candidates\n"
        assert (base / "build_bundle.py").read_text(encoding="utf-8") == "print('hi')\n"
        assert (base / "tests" / "test_build_bundle.py").is_file()

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        files = manifest["files"]
        assert ".claude/commands/spotlights-share-candidates/SKILL.md" in files
        assert ".claude/commands/spotlights-share-candidates/build_bundle.py" in files
        assert ".claude/commands/spotlights-share-candidates/tests/test_build_bundle.py" in files
        assert files[".claude/commands/spotlights-share-candidates/SKILL.md"] == _sha256_text(
            "# share candidates\n"
        )

    def test_mixed_bundle_installs_both_kinds(self, fake_bundle, project_root):
        (fake_bundle / "objective-setting.md").write_text("# obj\n", encoding="utf-8")
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("# sc\n", encoding="utf-8")

        rc = init_skills.install_skills(scope="project")
        assert rc == 0

        commands = project_root / ".claude" / "commands"
        assert (commands / "spotlights-objective-setting.md").is_file()
        assert (commands / "spotlights-share-candidates" / "SKILL.md").is_file()

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        assert ".claude/commands/spotlights-objective-setting.md" in manifest["files"]
        assert ".claude/commands/spotlights-share-candidates/SKILL.md" in manifest["files"]

    def test_directory_skill_excludes_junk(self, fake_bundle, project_root):
        skill = fake_bundle / "share-candidates"
        (skill / "__pycache__").mkdir(parents=True)
        (skill / "tests" / "__pycache__").mkdir(parents=True)
        (skill / "SKILL.md").write_text("# sc\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("x = 1\n", encoding="utf-8")
        (skill / "build_bundle.pyc").write_text("junk\n", encoding="utf-8")
        (skill / ".DS_Store").write_text("junk\n", encoding="utf-8")
        pycache = skill / "__pycache__"
        (pycache / "build_bundle.cpython-314.pyc").write_text("junk\n", encoding="utf-8")
        (skill / "tests" / "__pycache__" / "t.pyc").write_text("junk\n", encoding="utf-8")

        rc = init_skills.install_skills(scope="project")
        assert rc == 0

        base = project_root / ".claude" / "commands" / "spotlights-share-candidates"
        assert (base / "SKILL.md").is_file()
        assert (base / "build_bundle.py").is_file()
        assert not (base / "build_bundle.pyc").exists()
        assert not (base / ".DS_Store").exists()
        assert not (base / "__pycache__").exists()
        assert not (base / "tests").exists()  # tests/ held only __pycache__, nothing copied

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        for rel in manifest["files"]:
            assert "__pycache__" not in rel
            assert not rel.endswith(".pyc")
            assert not rel.endswith(".DS_Store")

    def test_force_directory_skill_per_file_independence(
        self, fake_bundle, project_root
    ):
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("v1 skill\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("v1 script\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        base = (
            project_root
            / ".claude"
            / "commands"
            / "spotlights-share-candidates"
        )
        # User edits the script but not the manifest.
        (base / "build_bundle.py").write_text(
            "user edited script\n", encoding="utf-8"
        )

        # Bundle upgrades both files.
        (skill / "SKILL.md").write_text("v2 skill\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("v2 script\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        # Unedited file upgraded; edited file preserved.
        assert (base / "SKILL.md").read_text(encoding="utf-8") == "v2 skill\n"
        assert (base / "build_bundle.py").read_text(
            encoding="utf-8"
        ) == "user edited script\n"

        manifest = json.loads(
            (
                project_root / ".spotlights" / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        files = manifest["files"]
        assert (
            files[
                ".claude/commands/spotlights-share-candidates/SKILL.md"
            ]
            == _sha256_text("v2 skill\n")
        )
        # Preserved file keeps its ORIGINAL (v1) recorded hash so a future
        # revert is upgradeable.
        assert (
            files[
                ".claude/commands/spotlights-share-candidates/build_bundle.py"
            ]
            == _sha256_text("v1 script\n")
        )

    def test_force_ignores_unmanaged_file_in_skill_dir(
        self, fake_bundle, project_root
    ):
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("v1\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        base = (
            project_root
            / ".claude"
            / "commands"
            / "spotlights-share-candidates"
        )
        # User drops their own file into the installed skill dir
        # (no manifest entry).
        (base / "notes.md").write_text("my notes\n", encoding="utf-8")

        (skill / "SKILL.md").write_text("v2\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        assert (base / "SKILL.md").read_text(encoding="utf-8") == "v2\n"
        assert (base / "notes.md").read_text(encoding="utf-8") == "my notes\n"

        manifest = json.loads(
            (
                project_root / ".spotlights" / "manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert (
            ".claude/commands/spotlights-share-candidates/notes.md"
            not in manifest["files"]
        )

    def test_subdir_without_skill_md_is_ignored(
        self, fake_bundle, project_root
    ):
        # A subdirectory with files but no SKILL.md is NOT a directory skill.
        not_a_skill = fake_bundle / "helpers"
        not_a_skill.mkdir()
        (not_a_skill / "README.md").write_text("not a skill\n", encoding="utf-8")
        (not_a_skill / "util.py").write_text("x = 1\n", encoding="utf-8")
        # A real slash command so the run installs something.
        (fake_bundle / "objective-setting.md").write_text(
            "# obj\n", encoding="utf-8"
        )

        rc = init_skills.install_skills(scope="project")
        assert rc == 0

        commands = project_root / ".claude" / "commands"
        # The valid slash command installed.
        assert (commands / "spotlights-objective-setting.md").is_file()
        # The non-skill subdir was ignored: no prefixed dir, no raw dir.
        assert not (commands / "spotlights-helpers").exists()
        assert not (commands / "helpers").exists()

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for rel in manifest["files"]:
            assert "helpers" not in rel


class TestCliEntryPoint:
    def test_init_routed_via_main(self, fake_bundle, project_root):
        from spotlights_engine.cli import main as engine_main

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        rc = engine_main(["init", "--scope", "project"])
        assert rc == 0
        assert (
            project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        ).is_file()

    def test_init_force_flag_parsed(self, fake_bundle, project_root):
        from spotlights_engine.init_skills import main as init_main

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_main(["--scope", "project", "--force"]) == 0
