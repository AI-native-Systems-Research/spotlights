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

Symlinks anywhere on the destination path are followed, including at the file
itself, and a dangling one has the directories its target needs created — see
`_ensure_dir` and `_write_target`. Symlinking `~/.claude`, `commands/`, an
individual skill directory, an individual skill *file*, or `.spotlights/` into a
dotfiles repo (stow, chezmoi, plain `ln -s`) is routine, and Claude Code reads
through those links.

A destination we cannot write is reported and exits non-zero; it never raises.
The manifest is still written first, so the files that did land are recorded —
a half-done install nothing has a record of is the worst outcome available.

Manifest at `<scope-root>/.spotlights/manifest.json` records `path -> sha256`
for each file we installed. It is a *record* — of provenance, version, and what
an eventual uninstall would remove — and is never consulted to decide whether to
write. `version`/`installed_at` describe the files on disk, so they only advance
when a run brings every bundled file up to date. Paths are relative to the scope
root, which `CLAUDE_CONFIG_DIR` can move for user scope; see `_scope_layout`.
"""

from __future__ import annotations

import argparse
import errno
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


def _write_target(path: Path) -> Path:
    """Where a write to `path` actually lands.

    Writing through a symlink hits its target, so the target's parent is what has
    to exist — `path.parent` is the *link's* directory, which says nothing about
    whether the write can succeed. For a dangling link they differ, and creating
    the wrong one leaves `write_text` raising `FileNotFoundError`.
    """
    return path.resolve() if path.is_symlink() else path


def _ensure_dir(path: Path, _seen: frozenset[Path] = frozenset()) -> None:
    """`mkdir -p` that tolerates a symlinked directory, dangling or not.

    A freshly-cloned dotfiles repo leaves its links dangling until something
    populates the target, and `Path.mkdir(exist_ok=True)` raises `FileExistsError`
    on a dangling link — it only forgives a path that is already a *directory*.
    `parents=True` re-raises it for a link partway up too, so walk the path
    ourselves and create what each dangling link points at.

    Raises `OSError`, never `RecursionError`: non-strict `resolve()` *stops* at a
    symlink loop and hands back the still-looping path, so recursing on it alone
    would never terminate. `_seen` catches that and reports it as the `ELOOP` it is,
    which the caller already handles as an unusable destination.
    """
    if path.is_dir():  # follows links, so a live symlinked directory ends here
        return
    if path.is_symlink():
        target = path.resolve()  # dangling; non-strict, so this is the real target
        if target in _seen or target == path:
            raise OSError(errno.ELOOP, "Too many levels of symbolic links", str(path))
        _ensure_dir(target, _seen | {target})
        return
    if path.parent != path:
        _ensure_dir(path.parent, _seen)
    path.mkdir(exist_ok=True)


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


def _prior_str(prior: dict | None, key: str) -> str | None:
    """A non-empty string field from the prior manifest, or None.

    `_read_manifest` only shape-checks `files`, so `version` may be absent, null,
    or a number. Anything that is not a string is dropped rather than copied into
    every manifest we rewrite from then on.
    """
    if prior is None:
        return None
    value = prior.get(key)
    return value if isinstance(value, str) and value else None


def _write_manifest(
    scope_root: Path, files: dict[str, str], prior: dict | None, fully_installed: bool
) -> Path:
    """Write the manifest, keeping prior provenance unless the install is complete.

    `version` and `installed_at` describe the files on disk, not the run, and they
    only advance when *every* bundled file was written this run. A partial run —
    a new release adding one skill beside skills an older one installed, which is
    the ordinary upgrade path — leaves mixed content on disk, and stamping the new
    version onto it would claim an upgrade that did not happen. `--force` brings
    everything up to date and so always advances them.

    The two are carried forward as a pair or not at all: they state one fact
    together, and taking the old `version` beside a fresh `installed_at` would
    report an ancient install as seconds old.
    """
    manifest_dir = scope_root / _MANIFEST_DIR
    _ensure_dir(manifest_dir)
    manifest_path = manifest_dir / _MANIFEST_NAME
    _ensure_dir(_write_target(manifest_path).parent)

    installed_version = _package_version()
    installed_at = datetime.now(UTC).isoformat(timespec="seconds")
    if not fully_installed:
        prior_version = _prior_str(prior, "version")
        prior_at = _prior_str(prior, "installed_at")
        if prior_version and prior_at:
            installed_version, installed_at = prior_version, prior_at

    payload = {
        "integration": "spotlights",
        "version": installed_version,
        "installed_at": installed_at,
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

    # Resolve the bundle before creating anything: a mispackaged wheel should not
    # leave an empty commands/ directory behind on a run that installs nothing.
    items = _plan_install_items(commands_rel)
    if not items:
        print(
            "spotlights-engine init: no bundled skills found "
            f"(looked in {_PACKAGE}/{_TEMPLATES_SUBDIR}).",
            file=sys.stderr,
        )
        return 1

    try:
        _ensure_dir(dest_dir)
    except OSError as exc:
        print(
            f"spotlights-engine init: cannot create {dest_dir} ({exc}).",
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
    failure: str | None = None

    for item in items:
        target_path = scope_root / Path(item.rel_path)

        if target_path.exists() and not force:
            skipped_existing.append(item.label)
            # Skipping is not uninstalling: a file an earlier run put on disk is
            # still on disk, so it stays in the record an uninstall would read.
            if item.rel_path in existing_files:
                new_files[item.rel_path] = existing_files[item.rel_path]
            continue

        # Bytes, not text: the walker installs every file a skill directory holds,
        # so a future binary asset must not abort the install, and a hash of the
        # bytes we wrote stays correct where text mode would translate newlines.
        try:
            bundled_bytes = item.source.read_bytes()
            _ensure_dir(_write_target(target_path).parent)
            target_path.write_bytes(bundled_bytes)
        except OSError as exc:
            # Stop at the first failure but still record what landed, below: a
            # half-done install nothing knows about is the worst outcome here.
            failure = f"{item.label} ({exc})"
            break
        new_files[item.rel_path] = hashlib.sha256(bundled_bytes).hexdigest()
        installed.append(item.label)

    # Everything the record already held that this run did not rewrite stays in it:
    # files this bundle no longer ships, and — when the loop stopped early — files
    # it never reached. Both are still on disk, so an uninstall still needs them.
    for rel_path, recorded_hash in existing_files.items():
        new_files.setdefault(rel_path, recorded_hash)

    try:
        manifest_path = _write_manifest(
            scope_root,
            new_files,
            existing_manifest,
            fully_installed=len(installed) == len(items),
        )
    except OSError as exc:
        print(
            f"spotlights-engine init: installed {len(installed)} file(s) but could "
            f"not write the manifest in {scope_root / _MANIFEST_DIR} ({exc}).",
            file=sys.stderr,
        )
        return 1

    if installed:
        print(f"installed: {', '.join(installed)}")
    if skipped_existing:
        print(
            "skipped (already exists, re-run with --force to overwrite): "
            f"{', '.join(skipped_existing)}"
        )
    print(f"manifest: {manifest_path}")
    print(f"destination: {dest_dir}")
    if failure is not None:
        print(
            f"spotlights-engine init: could not install {failure}. "
            "The manifest records the files that were installed.",
            file=sys.stderr,
        )
        return 1
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
