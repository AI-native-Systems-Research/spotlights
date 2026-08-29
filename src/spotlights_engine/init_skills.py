"""Implementation of `spotlights-engine init`.

Copies bundled skill templates into the user's `.claude/commands/` directory,
applying the `spotlights-` prefix at install time (mirrors spec-kit's pattern
of keeping unprefixed names in `templates/commands/` and prefixing per-agent
on install).

Manifest at `<scope-root>/.spotlights/manifest.json` records `path -> sha256`
for each managed file, so a future `init --force` can safely overwrite only
files the user hasn't edited. Paths are relative to the scope root, which
`CLAUDE_CONFIG_DIR` can move for user scope — see `_scope_layout`.

The manifest is a cache, never the sole source of truth: if it goes missing or
unreadable, an install re-adopts any file that still matches the bundled bytes,
so a lost record cannot permanently lock `--force` out of upgrading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from importlib.resources.abc import Traversable
from pathlib import Path

_PACKAGE = "spotlights_engine"
_TEMPLATES_SUBDIR = "_templates/commands"
_SKILL_PREFIX = "spotlights-"
_MANIFEST_DIR = ".spotlights"
_MANIFEST_NAME = "manifest.json"
_CLAUDE_COMMANDS_REL = Path(".claude") / "commands"
_SKILL_MANIFEST_NAME = "SKILL.md"
_EXCLUDED_DIR_NAMES = frozenset({"__pycache__"})
_EXCLUDED_FILE_NAMES = frozenset({".DS_Store"})


@dataclass
class _InstallItem:
    """One file to install, resolved to its manifest-relative POSIX path."""

    rel_path: str  # e.g. ".claude/commands/spotlights-share-candidates/SKILL.md"
    source: Traversable
    label: str  # human-facing, e.g. "spotlights-share-candidates/SKILL.md"


def _is_excluded_file(name: str) -> bool:
    return name in _EXCLUDED_FILE_NAMES or name.endswith(".pyc")


def _walk_skill_dir(
    node: Traversable, rel_parts: list[str]
) -> Iterator[tuple[list[str], Traversable]]:
    """Yield (path-parts-under-commands, file) for each installable file in a
    directory skill, skipping __pycache__/, *.pyc, and .DS_Store."""
    for entry in node.iterdir():
        if entry.is_dir():
            if entry.name in _EXCLUDED_DIR_NAMES:
                continue
            yield from _walk_skill_dir(entry, [*rel_parts, entry.name])
        elif entry.is_file():
            if _is_excluded_file(entry.name):
                continue
            yield ([*rel_parts, entry.name], entry)


def _package_version() -> str:
    try:
        return version("spotlights-engine")
    except PackageNotFoundError:
        return "0.0.0+local"


def _bundled_templates() -> Traversable:
    """Locate the bundled `commands/` template directory.

    Two layouts are valid:
    - **Wheel install**: `force-include` in pyproject.toml copies the repo-root
      `templates/` into the package as `spotlights_engine/_templates/`.
      `importlib.resources` finds it there.
    - **Editable install** (`uv sync` / `pip install -e`): `force-include` does
      not run. Fall back to the repo-root `templates/commands/` discovered by
      walking up from this file's location.
    """
    packaged = resources.files(_PACKAGE) / _TEMPLATES_SUBDIR
    if packaged.is_dir():
        return packaged

    # Editable-install fallback: <repo>/src/spotlights_engine/init_skills.py
    # → <repo>/templates/commands/
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "templates" / "commands"
        if candidate.is_dir():
            return candidate

    # Return the (non-existent) packaged path so caller's `is_dir()` check
    # produces a useful error message pointing at the canonical location.
    return packaged


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_dir(a: Path, b: Path) -> bool:
    """True when two paths name the same directory (trailing slashes, symlinks…)."""
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return False


def _claude_config_dir() -> Path | None:
    """Resolve `CLAUDE_CONFIG_DIR` the way Claude Code itself does, or None.

    Claude Code resolves its config root as, in JS:

        (process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude")).normalize("NFC")

    Three details of that expression are load-bearing, because our only job is
    to write where Claude Code reads — any divergence installs skills that
    silently never appear:

    - `??` is nullish, not `||`: a set-but-empty value is used *as* the root, so
      Claude Code reads a cwd-relative `commands/`. It does not fall back to
      `~/.claude` (verified against the installed CLI: with `CLAUDE_CONFIG_DIR=""`
      it does not see `~/.claude`'s config at all).
    - No tilde expansion. A literal `~/foo` is a directory named `~`, not `$HOME/foo`.
    - NFC normalization, which matters on macOS, where the filesystem hands back
      decomposed (NFD) names for non-ASCII paths.
    """
    raw = os.environ.get("CLAUDE_CONFIG_DIR")
    if raw is None:
        return None
    return Path(unicodedata.normalize("NFC", raw))


def _scope_layout(scope: str) -> tuple[Path, Path]:
    """Return `(scope_root, commands_rel)` for a scope.

    Everything the installer writes lives under `scope_root`: skills at
    `scope_root / commands_rel`, the manifest at `scope_root/.spotlights/`.
    Manifest keys are POSIX paths relative to `scope_root`, so a manifest is
    only meaningful next to the root it was written for.

    `CLAUDE_CONFIG_DIR` relocates Claude Code's whole `~/.claude` directory, so
    under user scope it becomes the root and commands sit directly beneath it —
    except when it merely spells the default location, which must not move the
    manifest away from an existing one. Project scope is unaffected by that
    variable: a project's own `.claude/commands/` is read from the project either way.
    """
    if scope == "project":
        return Path.cwd(), _CLAUDE_COMMANDS_REL
    if scope != "user":
        raise ValueError(f"unknown scope: {scope!r}")

    config_dir = _claude_config_dir()
    default_root = Path.home()
    # `CLAUDE_CONFIG_DIR=~/.claude` names exactly where the default layout already
    # installs. Honouring it literally would put the files in the right place but
    # move the manifest to ~/.claude/.spotlights/, orphaning ~/.spotlights/ — and
    # an install whose manifest cannot be found is one `--force` can never upgrade.
    if config_dir is None or _same_dir(config_dir, default_root / ".claude"):
        return default_root, _CLAUDE_COMMANDS_REL
    return config_dir, Path("commands")


def _warn_unusable_manifest(manifest_path: Path, reason: str) -> None:
    print(
        f"spotlights-engine init: ignoring unusable manifest {manifest_path} "
        f"({reason}); previously installed files will be treated as unmanaged.",
        file=sys.stderr,
    )


def _read_manifest(scope_root: Path) -> dict | None:
    """Read the manifest, or None when it is missing, unreadable or malformed.

    A manifest of the wrong shape is as useless to us as corrupt JSON, and
    reaching into it would crash the install, so both degrade to "no manifest".
    Losing the ownership record is worth a warning, so it is not silent.
    """
    manifest_path = scope_root / _MANIFEST_DIR / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _warn_unusable_manifest(manifest_path, str(exc))
        return None
    if not isinstance(payload, dict):
        _warn_unusable_manifest(
            manifest_path, f"expected a JSON object, got {type(payload).__name__}"
        )
        return None
    files = payload.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        _warn_unusable_manifest(
            manifest_path, "'files' is not an object of path -> sha256 strings"
        )
        return None
    return payload


def _write_manifest(scope_root: Path, files: dict[str, str]) -> Path:
    manifest_dir = scope_root / _MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / _MANIFEST_NAME
    payload = {
        "integration": "spotlights",
        "version": _package_version(),
        "installed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": files,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _plan_install_items(
    commands_rel: Path = _CLAUDE_COMMANDS_REL,
) -> list[_InstallItem]:
    """Resolve every bundled file to an _InstallItem.

    Top-level `<slug>.md` files become prefixed slash commands. Directory
    skills (subdirs with SKILL.md) have every file installed with structure preserved.
    `commands_rel` is the commands directory relative to the scope root, which
    `CLAUDE_CONFIG_DIR` can change — see `_scope_layout`.
    """
    root = _bundled_templates()
    if not root.is_dir():
        return []
    items: list[_InstallItem] = []
    for entry in root.iterdir():
        if entry.is_file() and entry.name.endswith(".md"):
            slug = entry.name[:-3]
            target_name = f"{_SKILL_PREFIX}{slug}.md"
            rel_path = (commands_rel / target_name).as_posix()
            items.append(_InstallItem(rel_path, entry, target_name))
        elif entry.is_dir():
            if not (entry / _SKILL_MANIFEST_NAME).is_file():
                continue  # not a directory skill
            prefixed = f"{_SKILL_PREFIX}{entry.name}"
            for parts, file_node in _walk_skill_dir(entry, [prefixed]):
                rel_path = commands_rel.joinpath(*parts).as_posix()
                items.append(_InstallItem(rel_path, file_node, "/".join(parts)))
    return sorted(items, key=lambda it: it.rel_path)


def install_skills(scope: str = "user", force: bool = False) -> int:
    """Install bundled skills into the scope's Claude Code commands directory.

    Returns process exit code (0 success, non-zero error).
    """
    scope_root, commands_rel = _scope_layout(scope)
    dest_dir = scope_root / commands_rel
    dest_dir.mkdir(parents=True, exist_ok=True)

    items = _plan_install_items(commands_rel)
    if not items:
        print(
            "spotlights-engine init: no bundled skills found "
            f"(looked in {_PACKAGE}/{_TEMPLATES_SUBDIR}).",
            file=sys.stderr,
        )
        return 1

    # Always read the prior manifest: --force needs it to tell our files from the
    # user's, and a plain re-run needs it to carry ownership forward. Reading it
    # only under --force silently rewrote `files` as {} on every no-op run, which
    # then made a later --force treat everything as unmanaged and refuse to upgrade.
    existing_manifest = _read_manifest(scope_root)
    existing_files: dict[str, str] = (
        existing_manifest.get("files", {}) if existing_manifest else {}
    )

    new_files: dict[str, str] = {}
    installed: list[str] = []
    skipped_user_edited: list[str] = []
    skipped_existing: list[str] = []

    for item in items:
        target_path = scope_root / Path(item.rel_path)
        bundled_text = item.source.read_text(encoding="utf-8")
        bundled_hash = hashlib.sha256(bundled_text.encode("utf-8")).hexdigest()

        if target_path.exists():
            on_disk_hash = _sha256(target_path)
            recorded_hash = existing_files.get(item.rel_path)
            if recorded_hash is None and on_disk_hash == bundled_hash:
                # No manifest entry, but the file on disk is byte-for-byte the one
                # we ship — it is ours whatever became of the manifest (deleted,
                # malformed, emptied by the pre-fix wipe bug, or left behind when
                # CLAUDE_CONFIG_DIR moved the root). Re-adopt it: without this the
                # record can never be rebuilt, and `--force` refuses every file
                # forever, blaming the user for edits they never made.
                recorded_hash = bundled_hash

            if not force:
                skipped_existing.append(item.label)
                # Carry ownership forward so we don't lie about what we manage.
                if recorded_hash is not None:
                    new_files[item.rel_path] = recorded_hash
                continue
            # --force: only overwrite files we own and the user hasn't edited.
            if recorded_hash is None:
                # Not in manifest and not identical to ours -> not ours. Don't touch.
                skipped_user_edited.append(item.label)
                continue
            if on_disk_hash != recorded_hash:
                # User edited a file we installed previously. Don't clobber.
                skipped_user_edited.append(item.label)
                new_files[item.rel_path] = recorded_hash
                continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(bundled_text, encoding="utf-8")
        new_files[item.rel_path] = bundled_hash
        installed.append(item.label)

    # Carry over manifest entries for files we don't manage in this bundle anymore.
    bundled_rel_paths = {item.rel_path for item in items}
    for rel_path, recorded_hash in existing_files.items():
        if rel_path not in bundled_rel_paths and rel_path not in new_files:
            new_files[rel_path] = recorded_hash

    manifest_path = _write_manifest(scope_root, new_files)

    if installed:
        print(f"installed: {', '.join(installed)}")
    if skipped_existing:
        print(
            "skipped (already exists, re-run with --force to overwrite): "
            f"{', '.join(skipped_existing)}"
        )
    if skipped_user_edited:
        print(
            "skipped (user-edited or not managed by spotlights, left untouched): "
            f"{', '.join(skipped_user_edited)}"
        )
    print(f"manifest: {manifest_path}")
    print(f"destination: {dest_dir}")
    return 0


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlights-engine init",
        description=(
            "Install bundled Spotlights skills into .claude/commands/ "
            "so /spotlights-* slash commands are available in Claude Code."
        ),
    )
    p.add_argument(
        "--scope",
        choices=("project", "user"),
        default="user",
        help=(
            "Where to install. 'user' (default): ~/.claude/commands/, available "
            "in every directory (or $CLAUDE_CONFIG_DIR/commands/ when that is "
            "set). 'project': <cwd>/.claude/commands/, this checkout only."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "Overwrite skills previously installed by spotlights. "
            "User-edited files are still preserved (detected via the "
            "manifest's sha256)."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return install_skills(scope=args.scope, force=args.force)
