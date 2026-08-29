"""Implementation of `spotlights-engine init`.

Copies bundled skill templates into the user's `.claude/commands/` directory,
applying the `spotlights-` prefix at install time (mirrors spec-kit's pattern
of keeping unprefixed names in `templates/commands/` and prefixing per-agent
on install).

Overwrite policy, following spec-kit's installer:

- default: write only the files that are not there yet; leave anything that
  already exists alone.
- `--force`: write every file the bundle ships, overwriting what is on disk.
  Local edits to those files are lost, which is what "force install this
  version" means.

Manifest at `<scope-root>/.spotlights/manifest.json` records `path -> sha256`
for each file we installed. It is a *record* — of provenance, version, and what
an eventual uninstall would remove — and is never consulted to decide whether to
write. Paths are relative to the scope root, which `CLAUDE_CONFIG_DIR` can move
for user scope; see `_scope_layout`.
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


def _first_symlink(scope_root: Path, target: Path) -> Path | None:
    """First symlink at or below `scope_root` on the way to `target`, else None.

    Writing through a symlink would let a link planted inside the commands
    directory redirect the write anywhere on disk, so the installer refuses
    instead of following it (spec-kit's installer does the same).

    `scope_root` itself is deliberately not checked: a symlinked `$HOME` or
    `CLAUDE_CONFIG_DIR` is legitimate, and Claude Code reads through it too.
    """
    current = target
    while current != scope_root and current.parent != current:
        if current.is_symlink():
            return current
        current = current.parent
    return None


def _nfc(path: Path) -> str:
    return unicodedata.normalize("NFC", str(path))


def _same_dir(a: Path, b: Path) -> bool:
    """True when two paths name the same directory (trailing slashes, symlinks…).

    Both sides are NFC-normalized before comparing: we normalize
    `CLAUDE_CONFIG_DIR` to match Claude Code, but `Path.home()` comes from the
    OS, which on macOS hands back decomposed (NFD) names. Comparing one against
    the other made the "names the default location" guard miss on any home
    directory with non-ASCII characters.
    """
    try:
        return _nfc(a.resolve()) == _nfc(b.resolve())
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
    # move the manifest to ~/.claude/.spotlights/, leaving a stale ~/.spotlights/
    # behind claiming to be the record of the same install.
    if config_dir is None or _same_dir(config_dir, default_root / ".claude"):
        return default_root, _CLAUDE_COMMANDS_REL
    return config_dir, Path("commands")


def _warn_unusable_manifest(manifest_path: Path, reason: str) -> None:
    print(
        f"spotlights-engine init: ignoring unusable manifest {manifest_path} "
        f"({reason}); it will be rewritten from this install, so records of "
        "files older versions installed are lost.",
        file=sys.stderr,
    )


def _read_manifest(scope_root: Path) -> dict | None:
    """Read the manifest, or None when it is missing, unreadable or malformed.

    A manifest of the wrong shape is as useless to us as corrupt JSON, and
    reaching into it would crash the install, so both degrade to "no manifest".
    That never blocks an install — it only loses the record of files older
    versions put on disk, which is worth a warning rather than silence.
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

    # Read the prior manifest so files we installed before but skip this time stay
    # in the record. It informs the record only — never the overwrite decision.
    existing_manifest = _read_manifest(scope_root)
    existing_files: dict[str, str] = existing_manifest.get("files", {}) if existing_manifest else {}

    new_files: dict[str, str] = {}
    installed: list[str] = []
    skipped_existing: list[str] = []
    skipped_symlink: list[str] = []

    for item in items:
        target_path = scope_root / Path(item.rel_path)
        bundled_text = item.source.read_text(encoding="utf-8")
        bundled_hash = hashlib.sha256(bundled_text.encode("utf-8")).hexdigest()

        # Refuse to write through a symlink, with or without --force: a link
        # planted inside the commands directory would redirect the write to an
        # arbitrary path outside the scope root.
        link = _first_symlink(scope_root, target_path)
        if link is not None:
            skipped_symlink.append(f"{item.label} (via {link})")
            continue

        if target_path.exists() and not force:
            skipped_existing.append(item.label)
            # Keep it in the record if we installed it earlier — it is still installed.
            if item.rel_path in existing_files:
                new_files[item.rel_path] = existing_files[item.rel_path]
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(bundled_text, encoding="utf-8")
        new_files[item.rel_path] = bundled_hash
        installed.append(item.label)

    # Carry over records for files this bundle no longer ships, so an eventual
    # uninstall can still find what older versions put on disk.
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
    if skipped_symlink:
        print(
            "skipped (symlinked destination, never written through): "
            f"{', '.join(skipped_symlink)}",
            file=sys.stderr,
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
            "Overwrite every bundled skill file with this version, discarding "
            "local edits to those files. Without it, files that already exist "
            "are left untouched. Files spotlights does not ship are never "
            "written either way."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    return install_skills(scope=args.scope, force=args.force)
