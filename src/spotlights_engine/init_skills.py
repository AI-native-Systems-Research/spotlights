"""Implementation of `spotlights-engine init`.

Copies bundled skill templates into the user's `.claude/commands/` directory,
applying the `spotlights-` prefix at install time (mirrors spec-kit's pattern
of keeping unprefixed names in `templates/commands/` and prefixing per-agent
on install).

Manifest at `<scope-root>/.spotlights/manifest.json` records `path -> sha256`
for each managed file, so a future `init --force` can safely overwrite only
files the user hasn't edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


@dataclass
class _InstallItem:
    """One file to install, resolved to its manifest-relative POSIX path."""

    rel_path: str  # e.g. ".claude/commands/spotlights-share-candidates/SKILL.md"
    source: Traversable
    label: str  # human-facing, e.g. "spotlights-share-candidates/SKILL.md"


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


def _scope_root(scope: str) -> Path:
    if scope == "user":
        return Path.home()
    if scope == "project":
        return Path.cwd()
    raise ValueError(f"unknown scope: {scope!r}")


def _read_manifest(scope_root: Path) -> dict | None:
    manifest_path = scope_root / _MANIFEST_DIR / _MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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


def _plan_install_items() -> list[_InstallItem]:
    """Resolve every bundled file to an _InstallItem.

    Top-level `<slug>.md` files become prefixed slash commands. (Directory
    skills are added in a later step.)
    """
    root = _bundled_templates()
    if not root.is_dir():
        return []
    items: list[_InstallItem] = []
    for entry in root.iterdir():
        if entry.is_file() and entry.name.endswith(".md"):
            slug = entry.name[:-3]
            target_name = f"{_SKILL_PREFIX}{slug}.md"
            rel_path = (_CLAUDE_COMMANDS_REL / target_name).as_posix()
            items.append(_InstallItem(rel_path, entry, target_name))
    return sorted(items, key=lambda it: it.rel_path)


def install_skills(scope: str = "project", force: bool = False) -> int:
    """Install bundled skills into `<scope-root>/.claude/commands/`.

    Returns process exit code (0 success, non-zero error).
    """
    scope_root = _scope_root(scope)
    dest_dir = scope_root / _CLAUDE_COMMANDS_REL
    dest_dir.mkdir(parents=True, exist_ok=True)

    items = _plan_install_items()
    if not items:
        print(
            "spotlights-engine init: no bundled skills found "
            f"(looked in {_PACKAGE}/{_TEMPLATES_SUBDIR}).",
            file=sys.stderr,
        )
        return 1

    existing_manifest = _read_manifest(scope_root) if force else None
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
            if not force:
                skipped_existing.append(item.label)
                # Preserve any prior manifest entry as-is so we don't lie about ownership.
                if item.rel_path in existing_files:
                    new_files[item.rel_path] = existing_files[item.rel_path]
                continue
            # --force: only overwrite files we own and the user hasn't edited.
            on_disk_hash = _sha256(target_path)
            recorded_hash = existing_files.get(item.rel_path)
            if recorded_hash is None:
                # Not in manifest -> not ours. Don't touch.
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
        default="project",
        help=(
            "Where to install. 'project' (default): <cwd>/.claude/commands/. "
            "'user': ~/.claude/commands/."
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
