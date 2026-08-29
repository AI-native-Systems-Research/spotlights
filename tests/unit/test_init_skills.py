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

    def test_plain_rerun_keeps_files_in_the_manifest(self, fake_bundle, project_root):
        """A no-op re-run must not forget what it recorded.

        The files it skipped are still installed, so they belong in the record an
        uninstall would read. An earlier version emptied `files` on every re-run.
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
        # ...and the record survives unchanged.
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

    def test_force_overwrites_unchanged_file(self, fake_bundle, project_root):
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

    def test_force_overwrites_an_edited_file(self, fake_bundle, project_root):
        """`--force` means force: an edited copy of a file we ship is replaced.

        Preserving edits is what the plain re-run is for.
        """
        (fake_bundle / "objective-setting.md").write_text("v1 body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        target.write_text("user edited this\n", encoding="utf-8")

        v2 = "v2 body\n"
        (fake_bundle / "objective-setting.md").write_text(v2, encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        assert target.read_text(encoding="utf-8") == v2
        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        rel = ".claude/commands/spotlights-objective-setting.md"
        assert manifest["files"][rel] == _sha256_text(v2)

    def test_force_overwrites_a_file_it_never_installed(self, fake_bundle, project_root):
        """No manifest entry, no prior install — `--force` still writes our version.

        The decision is "does the bundle ship this path", never "did we install
        this copy": a first `init --force` on a machine with hand-placed skills is
        exactly the case where the user wants the bundled version to win.
        """
        bundled = "bundled\n"
        (fake_bundle / "objective-setting.md").write_text(bundled, encoding="utf-8")
        target = project_root / ".claude" / "commands" / "spotlights-objective-setting.md"
        target.parent.mkdir(parents=True)
        target.write_text("hand-placed\n", encoding="utf-8")

        assert init_skills.install_skills(scope="project", force=True) == 0
        assert target.read_text(encoding="utf-8") == bundled

    def test_force_refuses_to_write_through_a_symlink(self, fake_bundle, project_root, capsys):
        """A symlink under the commands directory must not redirect the write.

        Following one would let a link planted there send bundled content to any
        path on disk, so the install skips it and says so on stderr.
        """
        (fake_bundle / "objective-setting.md").write_text("bundled\n", encoding="utf-8")
        commands = project_root / ".claude" / "commands"
        commands.mkdir(parents=True)
        outside = project_root / "outside.md"
        outside.write_text("do not clobber me\n", encoding="utf-8")
        (commands / "spotlights-objective-setting.md").symlink_to(outside)

        # A refused file is a file the user asked for and did not get.
        assert init_skills.install_skills(scope="project", force=True) == 1
        assert outside.read_text(encoding="utf-8") == "do not clobber me\n"
        assert "symlink" in capsys.readouterr().err

    def test_a_refused_symlink_keeps_its_prior_record(self, fake_bundle, project_root):
        """Refusing to write does not uninstall the file that is already there.

        Dropping it from `files` would make an uninstall miss a file we put on
        disk, and contradicts the record's whole purpose.
        """
        (fake_bundle / "objective-setting.md").write_text("v1\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        rel = ".claude/commands/spotlights-objective-setting.md"
        target = project_root / rel
        target.unlink()
        target.symlink_to(project_root / "outside.md")

        assert init_skills.install_skills(scope="project", force=True) == 1
        files = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )["files"]
        assert files[rel] == _sha256_text("v1\n")

    def test_force_refuses_to_write_through_a_symlinked_parent(self, fake_bundle, project_root):
        """The guard walks the whole path, not just the leaf.

        A symlinked skill *directory* redirects every file inside it, so it has to
        be caught too.
        """
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("bundled skill\n", encoding="utf-8")

        commands = project_root / ".claude" / "commands"
        commands.mkdir(parents=True)
        outside = project_root / "outside-dir"
        outside.mkdir()
        (commands / "spotlights-share-candidates").symlink_to(outside)

        assert init_skills.install_skills(scope="project", force=True) == 1
        assert list(outside.iterdir()) == []

    @pytest.mark.parametrize("linked", [".claude", ".claude/commands"])
    def test_a_symlinked_claude_dir_is_written_through(
        self, fake_bundle, monkeypatch, tmp_path, linked
    ):
        """Dotfiles setups symlink ~/.claude (or commands/) into a managed repo.

        Claude Code reads through those links, so the installer writes through
        them. An earlier version drew the symlink boundary at the scope root,
        which refused every file of such an install and reported success.
        """
        home = tmp_path / "linked-home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        real = tmp_path / "dotfiles" / "claude"
        real.mkdir(parents=True)

        link = home / linked
        if linked != ".claude":
            (home / ".claude").mkdir()
        link.symlink_to(real)

        body = "body\n"
        (fake_bundle / "objective-setting.md").write_text(body, encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0

        installed = home / ".claude" / "commands" / "spotlights-objective-setting.md"
        assert installed.read_text(encoding="utf-8") == body
        files = json.loads((home / ".spotlights" / "manifest.json").read_text(encoding="utf-8"))[
            "files"
        ]
        assert files == {".claude/commands/spotlights-objective-setting.md": _sha256_text(body)}

    def test_a_symlinked_spotlights_dir_is_written_through(self, fake_bundle, project_root):
        """`.spotlights/` gets the same treatment as `.claude/`: user infrastructure.

        Deliberate, not an oversight. It sits directly under the scope root, at the
        same level as `.claude`, and symlinking it into a dotfiles repo is the same
        legitimate setup. Guarding it would refuse to record an install for exactly
        the users the symlink boundary above exists to support, and there is no
        escalation to prevent: planting that link needs write access to the scope
        root, which already allows writing the skill files directly, and the only
        content we send through it is our own manifest JSON.
        """
        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        real = project_root / "dotfiles-spotlights"
        real.mkdir()
        (project_root / ".spotlights").symlink_to(real)

        assert init_skills.install_skills(scope="project") == 0
        assert (real / "manifest.json").is_file()

    def test_noop_rerun_keeps_prior_version_and_timestamp(
        self, fake_bundle, project_root, monkeypatch
    ):
        """`version`/`installed_at` describe the files on disk, not the run.

        A re-run after a package upgrade that writes nothing must not claim the
        new version for content the old one put there.
        """
        (fake_bundle / "objective-setting.md").write_text("v1\n", encoding="utf-8")
        monkeypatch.setattr(init_skills, "_package_version", lambda: "0.1.0")
        assert init_skills.install_skills(scope="project") == 0

        manifest_path = project_root / ".spotlights" / "manifest.json"
        first = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Package upgraded, but the bundle is already fully installed.
        monkeypatch.setattr(init_skills, "_package_version", lambda: "0.2.0")
        assert init_skills.install_skills(scope="project") == 0

        after = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert after["version"] == "0.1.0"
        assert after["installed_at"] == first["installed_at"]

        # ...and --force, which does write, brings both up to date.
        assert init_skills.install_skills(scope="project", force=True) == 0
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["version"] == "0.2.0"

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

    def test_user_scope_honours_claude_config_dir(self, fake_bundle, monkeypatch, tmp_path):
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

    def test_blank_claude_config_dir_matches_claude_code(self, fake_bundle, monkeypatch, tmp_path):
        """A set-but-empty value is a cwd-relative root, not a fallback to $HOME.

        Claude Code resolves `process.env.CLAUDE_CONFIG_DIR ?? join(homedir(),
        ".claude")` — nullish, so `""` is used as the root and it reads
        `./commands`. Verified against the installed CLI: with
        `CLAUDE_CONFIG_DIR=""` it does not see `~/.claude`'s config at all.
        Treating blank as unset wrote to `~/.claude/commands/` and reported
        success while Claude Code looked in the working directory.
        """
        home = tmp_path / "home"
        home.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(cwd)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0

        assert (cwd / "commands" / "spotlights-objective-setting.md").is_file()
        assert not (home / ".claude").exists()

    def test_claude_config_dir_is_not_tilde_expanded(self, fake_bundle, monkeypatch, tmp_path):
        """Claude Code passes the value to path.join verbatim — no `~` expansion.

        So `~/relocated` is a directory literally named `~`, relative to cwd.
        Expanding it ourselves would write to $HOME/relocated while Claude Code
        read ./~/relocated.
        """
        home = tmp_path / "home"
        home.mkdir()
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.chdir(cwd)
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", "~/relocated")

        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0

        assert (cwd / "~" / "relocated" / "commands").is_dir()
        assert not (home / "relocated").exists()

    @pytest.mark.parametrize("suffix", ["", "/", "/."])
    def test_claude_config_dir_naming_the_default_keeps_default_layout(
        self, fake_bundle, monkeypatch, tmp_path, suffix
    ):
        """`CLAUDE_CONFIG_DIR=~/.claude` must not move the manifest.

        It names exactly where the default layout already installs. Honouring it
        literally put the files in the right place but moved the manifest to
        ~/.claude/.spotlights/, orphaning an existing ~/.spotlights/ — and an
        install whose manifest cannot be found is one --force can never upgrade.
        """
        home = tmp_path / "home"
        (home / ".claude").mkdir(parents=True)
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", f"{home / '.claude'}{suffix}")

        body = "body\n"
        (fake_bundle / "objective-setting.md").write_text(body, encoding="utf-8")
        assert init_skills.install_skills(scope="user") == 0

        # Files where Claude Code reads them, manifest where the default keeps it.
        assert (home / ".claude" / "commands" / "spotlights-objective-setting.md").is_file()
        assert (home / ".spotlights" / "manifest.json").is_file()
        assert not (home / ".claude" / ".spotlights").exists()

        manifest = json.loads(
            (home / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["files"] == {
            ".claude/commands/spotlights-objective-setting.md": _sha256_text(body)
        }

    def test_a_lost_manifest_does_not_block_an_upgrade(self, fake_bundle, project_root):
        """`--force` never reads the manifest, so losing it changes nothing.

        Ownership used to gate overwriting, which made one lost or unreadable
        manifest permanently downgrade `--force` to a no-op. It is a record now.
        """
        (fake_bundle / "objective-setting.md").write_text("v1 body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        manifest_path = project_root / ".spotlights" / "manifest.json"
        rel = ".claude/commands/spotlights-objective-setting.md"
        manifest_path.unlink()

        v2 = "v2 body — upgraded\n"
        (fake_bundle / "objective-setting.md").write_text(v2, encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        assert (project_root / rel).read_text(encoding="utf-8") == v2
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["files"] == {
            rel: _sha256_text(v2)
        }

    def test_manifest_keeps_records_for_files_no_longer_bundled(self, fake_bundle, project_root):
        """A dropped skill stays in the record, so an uninstall can still find it.

        The record is the only trace of what older versions wrote; rebuilding it
        from just this bundle would strand those files on disk forever.
        """
        (fake_bundle / "objective-setting.md").write_text("body\n", encoding="utf-8")
        (fake_bundle / "retired.md").write_text("retired body\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        (fake_bundle / "retired.md").unlink()
        assert init_skills.install_skills(scope="project", force=True) == 0

        manifest = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )
        assert ".claude/commands/spotlights-retired.md" in manifest["files"]

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

    def test_force_upgrades_every_file_of_a_directory_skill(self, fake_bundle, project_root):
        """A partly-edited skill directory upgrades whole, not half.

        Per-file edit detection used to leave one file at v1 next to another at
        v2 — a skill in a state no release ever shipped.
        """
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("v1 skill\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("v1 script\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        base = project_root / ".claude" / "commands" / "spotlights-share-candidates"
        (base / "build_bundle.py").write_text("user edited script\n", encoding="utf-8")

        (skill / "SKILL.md").write_text("v2 skill\n", encoding="utf-8")
        (skill / "build_bundle.py").write_text("v2 script\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project", force=True) == 0

        assert (base / "SKILL.md").read_text(encoding="utf-8") == "v2 skill\n"
        assert (base / "build_bundle.py").read_text(encoding="utf-8") == "v2 script\n"

        files = json.loads(
            (project_root / ".spotlights" / "manifest.json").read_text(encoding="utf-8")
        )["files"]
        assert files[".claude/commands/spotlights-share-candidates/SKILL.md"] == (
            _sha256_text("v2 skill\n")
        )
        assert files[".claude/commands/spotlights-share-candidates/build_bundle.py"] == (
            _sha256_text("v2 script\n")
        )

    def test_force_leaves_files_the_bundle_does_not_ship(self, fake_bundle, project_root):
        """`--force` overwrites our files, not everything in the directory.

        The user's own notes sitting beside an installed skill are not ours to
        replace at any force level.
        """
        skill = fake_bundle / "share-candidates"
        skill.mkdir()
        (skill / "SKILL.md").write_text("v1\n", encoding="utf-8")
        assert init_skills.install_skills(scope="project") == 0

        base = project_root / ".claude" / "commands" / "spotlights-share-candidates"
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
