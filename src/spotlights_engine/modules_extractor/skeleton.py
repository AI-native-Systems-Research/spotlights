"""Stage 2 — deterministic directory inventory and skeleton.

Pure Python. Enumerates every non-ignored, non-excluded, non-symlink directory
under the selected source root whose subtree contains a non-`__init__` source
file, classifies each as mandatory (`required`) or foldable, and computes a
content fingerprint over the accepted source files so Stage 5 can detect
concurrent repository mutation.

The completeness guarantee is structural: a `required` directory must be
emitted or validly folded by Stage 3, and Stage 5 recomputes this inventory and
its fingerprint before accepting a tree. See
`design/module_extraction_fix_impl_plan.md` (Stage 2).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from spotlights_engine.modules_extractor.stage_schemas import Skeleton, SkeletonNode

# Version the extension set + walk profile together: both affect the coverage
# guarantee, so a change must invalidate the `inventory_fingerprint`.
ALGORITHM_PROFILE_VERSION = "skeleton.v3"

# Compared case-insensitively against a file suffix. Lock files, manifests, and
# other non-source files are naturally excluded (their suffixes aren't here).
SOURCE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
        ".kt", ".kts", ".rb", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs",
        ".swift", ".scala", ".m", ".mm", ".php", ".ex", ".exs", ".cu", ".cuh",
        ".vue", ".svelte", ".proto", ".sh", ".bash", ".lua", ".dart", ".fs",
        ".fsi", ".fsx", ".clj", ".cljs", ".r", ".jl", ".sol", ".zig", ".hs",
        ".lhs", ".ml", ".mli", ".erl", ".hrl", ".nim", ".pl", ".pm",
        ".groovy", ".elm", ".ps1", ".psm1", ".d", ".rkt", ".cmake",
    }
)

# Directory basenames pruned exactly (plus hidden dirs and *.egg-info, handled
# in `_is_ignored_dirname`). Deliberately does NOT include generic "generated"
# names: projects often ship first-party `gen`/`generated` directories.
_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules", "vendor", ".git", "dist", "build", "target",
        "__pycache__", ".venv", ".tox", ".mypy_cache", ".pytest_cache",
        ".idea", ".vscode",
    }
)

_INIT_BASENAME = "__init__.py"


def is_source_file(name: str) -> bool:
    return PurePosixPath(name).suffix.lower() in SOURCE_EXTENSIONS


def _is_init(name: str) -> bool:
    return name == _INIT_BASENAME


def _is_ignored_dirname(name: str) -> bool:
    if name in _IGNORED_DIR_NAMES:
        return True
    if name.startswith(".") and name not in (".",):
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def _rel_join(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


@dataclass
class ScanResult:
    """Accepted (non-symlink) source files and skipped symlinks from a walk."""

    files: list[str] = field(default_factory=list)  # sorted repo-relative POSIX
    skipped_symlinks: list[str] = field(default_factory=list)  # sorted


def _excluded_dir(rel: str, excluded_dirs: frozenset[str]) -> bool:
    """True when `rel` equals or is nested under any excluded path."""
    if rel in excluded_dirs:
        return True
    for ex in excluded_dirs:
        if rel == ex or rel.startswith(ex + "/"):
            return True
    return False


def scan_source_files(
    repo_path: Path,
    start_rel: str = "",
    *,
    excluded: frozenset[str] = frozenset(),
) -> ScanResult:
    """Safe recursive scan for accepted source files below `start_rel`.

    Prunes ignored directories, skips symlinks (both files and directories,
    recording them), and prunes any path equal to or under an `excluded` entry.
    Returns sorted lists so the result is stable across creation order and
    platforms.
    """
    files: list[str] = []
    symlinks: list[str] = []

    start_abs = repo_path / start_rel if start_rel else repo_path

    def _recurse(abs_dir: Path, rel_dir: str) -> None:
        try:
            entries = sorted(abs_dir.iterdir(), key=lambda p: p.name)
        except (OSError, NotADirectoryError):
            return
        for entry in entries:
            name = entry.name
            rel = _rel_join(rel_dir, name)
            is_symlink = entry.is_symlink()
            if entry.is_dir():
                if is_symlink:
                    symlinks.append(rel)
                    continue
                if _is_ignored_dirname(name):
                    continue
                if _excluded_dir(rel, excluded):
                    continue
                _recurse(entry, rel)
            elif entry.is_file() or is_symlink:
                if is_symlink:
                    if is_source_file(name):
                        symlinks.append(rel)
                    continue
                if not is_source_file(name):
                    continue
                if rel in excluded:
                    continue
                files.append(rel)

    if start_abs.is_dir() and not start_abs.is_symlink():
        _recurse(start_abs, start_rel)

    return ScanResult(files=sorted(files), skipped_symlinks=sorted(symlinks))


def compute_fingerprint(
    repo_path: Path,
    files: list[str],
    *,
    extra_paths: list[str],
) -> str:
    """Content fingerprint = profile version + (path, sha256(content)) per
    accepted source file + sorted extra paths (exclusions / skipped symlinks).

    Uses content digests, never mtimes/sizes, so a same-size edit is detected.
    """
    h = hashlib.sha256()
    h.update(ALGORITHM_PROFILE_VERSION.encode("utf-8"))
    h.update(b"\x00files\x00")
    for rel in sorted(files):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        try:
            digest = hashlib.sha256((repo_path / rel).read_bytes()).hexdigest()
        except OSError:
            digest = "MISSING"
        h.update(digest.encode("utf-8"))
        h.update(b"\x00")
    h.update(b"\x00extra\x00")
    for rel in sorted(extra_paths):
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ── Raw walk producing the classified tree ───────────────────────────────


@dataclass
class _RawDir:
    rel: str
    direct_source_files: list[str]  # repo-relative, non-init, sorted
    direct_all_source_files: list[str]  # includes __init__.py, sorted
    children: list[_RawDir]  # source-bearing children only, sorted by rel

    @property
    def has_source(self) -> bool:
        if self.direct_source_files:
            return True
        return any(c.has_source for c in self.children)


def _build_raw(
    repo_path: Path,
    source_root: str,
    excluded_dirs: frozenset[str],
    excluded_files: frozenset[str],
) -> tuple[list[_RawDir], list[str]]:
    """Build the raw source-bearing directory tree under `source_root`.

    Returns (top-level source-bearing dirs, sorted skipped-symlink paths).
    Never emits the source root itself as a node.
    """
    symlinks: list[str] = []
    start_abs = repo_path / source_root if source_root else repo_path

    def _recurse(abs_dir: Path, rel_dir: str) -> _RawDir | None:
        try:
            entries = sorted(abs_dir.iterdir(), key=lambda p: p.name)
        except (OSError, NotADirectoryError):
            return None
        direct_all: list[str] = []
        direct_noninit: list[str] = []
        children: list[_RawDir] = []
        for entry in entries:
            name = entry.name
            rel = _rel_join(rel_dir, name)
            is_symlink = entry.is_symlink()
            if entry.is_dir():
                if is_symlink:
                    symlinks.append(rel)
                    continue
                if _is_ignored_dirname(name):
                    continue
                if _excluded_dir(rel, excluded_dirs):
                    continue
                child = _recurse(entry, rel)
                if child is not None and child.has_source:
                    children.append(child)
            elif entry.is_file() or is_symlink:
                if is_symlink:
                    if is_source_file(name):
                        symlinks.append(rel)
                    continue
                if not is_source_file(name):
                    continue
                if rel in excluded_files:
                    continue
                direct_all.append(rel)
                if not _is_init(name):
                    direct_noninit.append(rel)
        return _RawDir(
            rel=rel_dir,
            direct_source_files=sorted(direct_noninit),
            direct_all_source_files=sorted(direct_all),
            children=sorted(children, key=lambda c: c.rel),
        )

    if not start_abs.is_dir() or start_abs.is_symlink():
        return [], sorted(symlinks)

    root = _recurse(start_abs, source_root)
    top: list[_RawDir] = []
    if root is not None:
        top = [c for c in root.children if c.has_source]
    return sorted(top, key=lambda c: c.rel), sorted(symlinks)


def _representative_files(raw: _RawDir) -> list[str]:
    """Deterministic reading seed: entry files first, then lexical, at most 5."""

    def _is_entry(rel: str) -> bool:
        base = PurePosixPath(rel).name
        if base == _INIT_BASENAME:
            return True
        stem = PurePosixPath(base).stem
        return stem in ("index", "main")

    entry = sorted(f for f in raw.direct_all_source_files if _is_entry(f))
    rest = sorted(f for f in raw.direct_all_source_files if not _is_entry(f))
    return (entry + rest)[:5]


def _organizational_only(raw: _RawDir) -> bool:
    """A source-bearing directory whose only direct source file is `__init__.py`
    and which owns fewer than two source-bearing children (so it is not a
    structural namespace owner)."""
    return not raw.direct_source_files and len(raw.children) < 2


def _classify_node(raw: _RawDir) -> tuple[SkeletonNode, list[str]]:
    """Convert a `_RawDir` into a `SkeletonNode` (children recursed first).

    Returns (node, organizational_only_paths_in_subtree).
    """
    child_nodes: list[SkeletonNode] = []
    org_only: list[str] = []
    for c in raw.children:
        node, sub_org = _classify_node(c)
        child_nodes.append(node)
        org_only.extend(sub_org)

    direct_count = len(raw.direct_source_files)
    child_count = len(raw.children)
    subtree_count = direct_count + sum(
        c.subtree_source_file_count for c in child_nodes
    )

    reasons: list[str] = []
    if direct_count >= 2:
        reasons.append("two_or_more_direct_source_files")
    if direct_count >= 1 and child_count >= 1:
        reasons.append("direct_source_file_and_child_branch")
    if direct_count == 0 and child_count >= 2:
        reasons.append("namespace_owner_of_multiple_children")

    required = bool(reasons)
    if _organizational_only(raw):
        org_only.append(raw.rel)

    node = SkeletonNode(
        path=raw.rel,
        direct_source_file_count=direct_count,
        subtree_source_file_count=subtree_count,
        source_child_count=child_count,
        representative_files=_representative_files(raw),
        required=required,
        required_reasons=reasons,
        children=child_nodes,
    )
    return node, org_only


def _branch_has_required(node: SkeletonNode) -> bool:
    if node.required:
        return True
    return any(_branch_has_required(c) for c in node.children)


def _promote_if_no_required(node: SkeletonNode) -> None:
    """Promote the highest node of a branch with no required node to required.

    Prevents a branch made entirely of single-file passthroughs from having no
    emittable owner into which its directories could be folded.
    """
    if not _branch_has_required(node):
        node.required = True
        if "promoted_branch_owner" not in node.required_reasons:
            node.required_reasons = [*node.required_reasons, "promoted_branch_owner"]


def build_skeleton(
    repo_path: Path,
    source_root: str,
    *,
    excluded_dirs: frozenset[str] = frozenset(),
    excluded_files: frozenset[str] = frozenset(),
    inventory_fingerprint: str | None = None,
) -> Skeleton:
    """Build the validated `Skeleton` for `repo_path` rooted at `source_root`.

    `excluded_dirs` / `excluded_files` are the validated Stage-1 semantic
    exclusions (repo-relative). `inventory_fingerprint`, when omitted, is
    computed here from the accepted source files.
    """
    top_raw, skipped_symlinks = _build_raw(
        repo_path, source_root, excluded_dirs, excluded_files
    )

    nodes: list[SkeletonNode] = []
    org_only: list[str] = []
    for raw in top_raw:
        node, sub_org = _classify_node(raw)
        _promote_if_no_required(node)
        nodes.append(node)
        org_only.extend(sub_org)

    # Ignored roots are audited as the deterministic prune set; recording every
    # pruned path repo-wide would be unbounded, so we record the top-level
    # ignore/exclusion roots that the walk actually encountered.
    excluded_all = sorted(set(excluded_dirs) | set(excluded_files))

    scan = scan_source_files(
        repo_path,
        source_root,
        excluded=frozenset(excluded_dirs) | frozenset(excluded_files),
    )
    if inventory_fingerprint is None:
        inventory_fingerprint = compute_fingerprint(
            repo_path,
            scan.files,
            extra_paths=[*excluded_all, *skipped_symlinks],
        )

    return Skeleton(
        source_root=source_root,
        nodes=nodes,
        ignored=sorted(_IGNORED_DIR_NAMES),
        excluded=excluded_all,
        organizational_only=sorted(set(org_only)),
        skipped_symlinks=skipped_symlinks,
        inventory_fingerprint=inventory_fingerprint,
    )


__all__ = [
    "ALGORITHM_PROFILE_VERSION",
    "SOURCE_EXTENSIONS",
    "ScanResult",
    "build_skeleton",
    "compute_fingerprint",
    "is_source_file",
    "scan_source_files",
]
