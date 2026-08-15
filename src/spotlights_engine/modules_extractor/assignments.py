"""Stage-3 assignment validation and coverage.

Five rules replace the old Stage-3 tree/fold rules
(`design/module_extractor_simplified.md` §2.4). This module owns:

- the deterministic **collision preflight** (`compute_collision_precedence`) —
  normalized-qualified-name collision classes computed from the full skeleton
  before any model call;
- **V1** — assignment/decision keys are inventoried, real non-symlink
  directories (`validate_assignment_paths`);
- **V2** — totality over the inventory, reported as the versioned
  `AssignmentCoverageReport` (`compute_assignment_coverage`);
- **V3** — label policy, keep-reason bijection, and ownership
  (`validate_assignment_labels`); and
- **V4** — exact metadata keys and main-file territory rules
  (`validate_metadata_entries`, `compute_metadata_coverage`).

V5 (public-tree validation) lives in `derive.py`, which builds the provisional
and final `ProjectTree`s. Failures are returned as stable, path-indexed
`AssignmentIssue` lists (never raised one path at a time) so the bounded
repair budget sees every problem at once; `format_assignment_issues` caps the
rendered text for prompt size.

`coverage.py` supplies Stage-1 validation, `CrossArtifactError`, and the shared
filesystem helpers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.constants import MAX_MAIN_FILES
from spotlights_engine.modules_extractor.coverage import (
    _dir_has_direct_file,
    _is_real_dir,
    _is_real_file,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleInfo,
    Skeleton,
    SkeletonNode,
)
from spotlights_engine.schemas.project import _qualified_name

# Cap on issues rendered into one repair prompt; mirrors the v1 validator's
# `_MAX_REPORTED_PROBLEMS` rationale.
_MAX_REPORTED_ISSUES = 20


class AssignmentIssue(BaseModel):
    """One structured validation failure — stable, path-indexed, batchable."""

    model_config = ConfigDict(extra="forbid")

    code: str
    path: str | None = None
    detail: str


def format_assignment_issues(
    issues: list[AssignmentIssue], *, cap: int = _MAX_REPORTED_ISSUES
) -> str:
    """Render issues into one bounded repair-prompt message."""
    shown = issues[:cap]
    text = "; ".join(i.detail for i in shown)
    extra = len(issues) - len(shown)
    if extra > 0:
        text += f"; (and {extra} more issue(s))"
    return text


# ── Path helpers ──────────────────────────────────────────────────────────


def nearest_module_ancestor(path: str, module_paths: set[str]) -> str | None:
    """The deepest **strict** ancestor of `path` present in `module_paths`."""
    current = path
    while "/" in current:
        current = current.rsplit("/", 1)[0]
        if current in module_paths:
            return current
    return None


def nearest_module_owner_of_file(
    file_path: str, module_paths: set[str]
) -> str | None:
    """The deepest module directory containing `file_path` (a file, so its
    owner chain starts at its parent directory)."""
    return nearest_module_ancestor(file_path, module_paths)


# ── Collision preflight ───────────────────────────────────────────────────


class CollisionPrecedence(BaseModel):
    """Deterministic normalized-qualified-name collision policy.

    The unchanged public `ProjectTree` derives a module's qualified name from
    its full path (every segment normalized), so two real paths that normalize
    identically (`foo-bar` vs `foo_bar`) cannot both be modules. Within each
    collision class the path with the largest subtree count (then lexically
    smallest path) keeps module eligibility; every other member is forced
    `PART`. Computed from the full skeleton before sharding, and shown in
    every assignment prompt — this precedence outranks the size rule.
    """

    model_config = ConfigDict(extra="forbid")

    forced_part: dict[str, str] = Field(default_factory=dict)
    """Loser path -> winning path, for every nested collision class."""

    top_level_collisions: list[list[str]] = Field(default_factory=list)
    """Collision classes made of top-level skeleton paths. Fatal: the losers
    would have no legal module ancestor, which the unchanged public naming
    contract cannot represent. Each entry is the sorted class member list."""

    @property
    def fatal(self) -> bool:
        return bool(self.top_level_collisions)


def compute_collision_precedence(skeleton: Skeleton) -> CollisionPrecedence:
    """Group every skeleton path by normalized qualified name; pick winners."""
    by_qn: dict[str, list[SkeletonNode]] = {}
    for node in skeleton.iter_nodes():
        qn = _qualified_name(node.path, skeleton.source_root)
        by_qn.setdefault(qn, []).append(node)

    top_level = {n.path for n in skeleton.nodes}
    forced: dict[str, str] = {}
    fatal: list[list[str]] = []
    for _qn, nodes in sorted(by_qn.items()):
        if len(nodes) < 2:
            continue
        # Colliding QNs have the same segment count, so a class is either all
        # top-level (single-segment) or all nested.
        if any(n.path in top_level for n in nodes):
            fatal.append(sorted(n.path for n in nodes))
            continue
        winner = min(
            nodes, key=lambda n: (-n.subtree_source_file_count, n.path)
        )
        for node in nodes:
            if node.path != winner.path:
                forced[node.path] = winner.path
    return CollisionPrecedence(forced_part=forced, top_level_collisions=fatal)


# ── V1 — real inventoried paths ───────────────────────────────────────────


def validate_assignment_paths(
    tree: AssignmentTree, skeleton: Skeleton, repo_path: Path
) -> list[AssignmentIssue]:
    """V1: every (already-normalized) assignment or module-decision key is an
    inventoried, real non-symlink directory inside the repository. Extra
    inventory keys are rejected here; missing ones are V2 coverage."""
    inventory = skeleton.all_paths()
    issues: list[AssignmentIssue] = []
    for path in sorted(tree.assignments):
        if path not in inventory:
            issues.append(
                AssignmentIssue(
                    code="assignment_path_not_inventoried",
                    path=path,
                    detail=(
                        f"assignment key {path!r} is not in the skeleton "
                        "inventory; label only supplied skeleton paths"
                    ),
                )
            )
        elif not _is_real_dir(repo_path, path):
            issues.append(
                AssignmentIssue(
                    code="assignment_path_not_a_directory",
                    path=path,
                    detail=(
                        f"assignment key {path!r} is not a real non-symlink "
                        "directory inside the repository"
                    ),
                )
            )
    for path in sorted(tree.module_decisions):
        if path not in inventory:
            issues.append(
                AssignmentIssue(
                    code="decision_path_not_inventoried",
                    path=path,
                    detail=(
                        f"module_decisions key {path!r} is not in the skeleton "
                        "inventory"
                    ),
                )
            )
        elif not _is_real_dir(repo_path, path):
            issues.append(
                AssignmentIssue(
                    code="decision_path_not_a_directory",
                    path=path,
                    detail=(
                        f"module_decisions key {path!r} is not a real "
                        "non-symlink directory inside the repository"
                    ),
                )
            )
    return issues


# ── V2 — totality / coverage ──────────────────────────────────────────────


class AssignmentCoverageReport(BaseModel):
    """The versioned v2 coverage ledger (`schema_version: coverage.v2`).

    `assigned = inventory & set(assignments)`; `modules`/`parts` describe raw
    parseable labels on inventory paths; `missing = inventory - assigned` and
    `extra = set(assignments) - inventory`. Invalidity is orthogonal (see
    `invalid_assignments`), never a third label. There is no
    `optional_unaccounted`: every accepted inventory path is explicitly
    accounted for. `required_missing` is a diagnostic subset of `missing`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["coverage.v2"] = "coverage.v2"
    inventory: list[str] = Field(default_factory=list)
    required: list[str] = Field(default_factory=list)
    assigned: list[str] = Field(default_factory=list)
    modules: list[str] = Field(default_factory=list)
    parts: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)
    required_missing: list[str] = Field(default_factory=list)
    optional_inventory: list[str] = Field(default_factory=list)
    optional_modules: list[str] = Field(default_factory=list)
    optional_parts: list[str] = Field(default_factory=list)
    invalid_assignments: list[AssignmentIssue] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def compute_assignment_coverage(
    tree: AssignmentTree,
    skeleton: Skeleton,
    *,
    invalid: list[AssignmentIssue] | None = None,
) -> AssignmentCoverageReport:
    """V2 ledger over raw parseable labels. `invalid` attaches orthogonal
    structured issues for best-effort failure reports."""
    inventory = skeleton.all_paths()
    required = skeleton.required_paths()
    keys = set(tree.assignments)

    assigned = inventory & keys
    modules = {p for p in assigned if tree.assignments[p] == "MODULE"}
    parts = {p for p in assigned if tree.assignments[p] == "PART"}
    missing = inventory - assigned
    extra = keys - inventory
    optional_inventory = inventory - required

    return AssignmentCoverageReport(
        inventory=sorted(inventory),
        required=sorted(required),
        assigned=sorted(assigned),
        modules=sorted(modules),
        parts=sorted(parts),
        missing=sorted(missing),
        extra=sorted(extra),
        required_missing=sorted(required & missing),
        optional_inventory=sorted(optional_inventory),
        optional_modules=sorted(optional_inventory & modules),
        optional_parts=sorted(optional_inventory & parts),
        invalid_assignments=list(invalid or []),
    )


# ── V3 — label policy and ownership ───────────────────────────────────────


def validate_assignment_labels(
    tree: AssignmentTree,
    scope: Skeleton,
    *,
    merge_threshold: int,
    precedence: CollisionPrecedence,
    anchor_paths: set[str],
    require_owner_in_scope: bool,
) -> list[AssignmentIssue]:
    """V3 over one scope (a shard's slice, or the full skeleton).

    `anchor_paths` are the true top-level skeleton paths present in this scope
    — they must be `MODULE` and are exempt from `keep_reason`. A nested shard
    passes an empty set: its displayed root is not "top-level" merely because
    the slice contains one root node. `require_owner_in_scope` is True only
    for global validation; a `PART` with no module ancestor inside a nested
    fragment is provisionally legal (its owner resolves after union — the
    top-level anchors guarantee one exists).
    """
    nodes = {n.path: n for n in scope.iter_nodes()}
    module_paths = tree.module_paths()
    issues: list[AssignmentIssue] = []

    for path in sorted(tree.assignments):
        node = nodes.get(path)
        if node is None:
            continue  # V1 reports extras; V2 reports misses
        label = tree.assignments[path]
        winner = precedence.forced_part.get(path)
        if winner is not None:
            # Collision precedence outranks every other rule, size included.
            if label != "PART":
                issues.append(
                    AssignmentIssue(
                        code="forced_part_violated",
                        path=path,
                        detail=(
                            f"{path!r} normalizes to the same public "
                            f"qualified name as {winner!r}, which holds module "
                            "eligibility for that name; label it PART"
                        ),
                    )
                )
            continue
        if path in anchor_paths:
            if label != "MODULE":
                issues.append(
                    AssignmentIssue(
                        code="top_level_not_module",
                        path=path,
                        detail=(
                            f"top-level path {path!r} must be MODULE (every "
                            "top-level branch is a structural anchor)"
                        ),
                    )
                )
            continue
        if node.subtree_source_file_count > merge_threshold and label != "MODULE":
            issues.append(
                AssignmentIssue(
                    code="large_path_not_module",
                    path=path,
                    detail=(
                        f"{path!r} has subtree_source_file_count "
                        f"{node.subtree_source_file_count} > merge threshold "
                        f"{merge_threshold}, so it must be MODULE"
                    ),
                )
            )
        if (
            label == "MODULE"
            and node.subtree_source_file_count <= merge_threshold
        ):
            decision = tree.module_decisions.get(path)
            if decision is None or not decision.keep_reason:
                issues.append(
                    AssignmentIssue(
                        code="keep_reason_missing",
                        path=path,
                        detail=(
                            f"MODULE {path!r} is at or below the merge "
                            f"threshold ({node.subtree_source_file_count} <= "
                            f"{merge_threshold}); its module_decisions entry "
                            "needs a non-empty keep_reason explaining its "
                            "independent responsibility"
                        ),
                    )
                )

    # Exact bijection: module_decisions keys == MODULE-labeled paths.
    for path in sorted(module_paths):
        if path in nodes and path not in tree.module_decisions:
            issues.append(
                AssignmentIssue(
                    code="decision_missing",
                    path=path,
                    detail=(
                        f"MODULE {path!r} has no module_decisions entry; every "
                        "module needs one (keep_reason may be null for anchors "
                        "and above-threshold modules)"
                    ),
                )
            )
    for path in sorted(tree.module_decisions):
        if path in nodes and path not in module_paths:
            issues.append(
                AssignmentIssue(
                    code="decision_orphaned",
                    path=path,
                    detail=(
                        f"module_decisions entry for {path!r}, which is not "
                        "labeled MODULE; remove it"
                    ),
                )
            )

    if require_owner_in_scope:
        for path in sorted(p for p in tree.part_paths() if p in nodes):
            # A top-level PART is already rejected by the structural-anchor
            # rule above. Do not add the mechanically consequent no-owner
            # diagnostic as a second failure for the same decision.
            if path in anchor_paths:
                continue
            if nearest_module_ancestor(path, module_paths) is None:
                issues.append(
                    AssignmentIssue(
                        code="part_without_module_ancestor",
                        path=path,
                        detail=(
                            f"PART {path!r} has no strict ancestor labeled "
                            "MODULE to own it"
                        ),
                    )
                )
    return issues


# ── V4 — metadata keys and main-file territory ────────────────────────────


class MetadataCoverageReport(BaseModel):
    """Per-call (and final-union) Stage-3B key ledger."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["metadata_coverage.v1"] = "metadata_coverage.v1"
    requested: list[str] = Field(default_factory=list)
    provided: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    extra: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def compute_metadata_coverage(
    provided_keys: Iterable[str], requested: set[str]
) -> MetadataCoverageReport:
    provided = set(provided_keys)
    return MetadataCoverageReport(
        requested=sorted(requested),
        provided=sorted(provided),
        missing=sorted(requested - provided),
        extra=sorted(provided - requested),
    )


def validate_metadata_entries(
    modules: Mapping[str, ModuleInfo],
    requested: set[str],
    module_paths: set[str],
    repo_path: Path,
    *,
    claimed: dict[str, str] | None = None,
) -> list[AssignmentIssue]:
    """V4 over one batch (or the final union, with `requested` = all modules).

    Applies the main-file ownership and filesystem rules:

    - a main file lies under the module, and its nearest resolved module owner
      is that module;
    - real, non-symlink files — any extension;
    - zero main files only for a pure container directory (no direct
      non-symlink file of any kind); and
    - no file claimed by two modules — `claimed` threads the cross-batch map
      for the union's defense-in-depth re-check.

    Missing requested keys are the coverage class (`compute_metadata_coverage`),
    not issues here.
    """
    issues: list[AssignmentIssue] = []
    seen: dict[str, str] = dict(claimed or {})

    for path in sorted(modules):
        if path not in requested:
            issues.append(
                AssignmentIssue(
                    code="metadata_key_not_requested",
                    path=path,
                    detail=(
                        f"metadata for {path!r} was not requested in this "
                        "scope; return exactly the requested module set"
                    ),
                )
            )

    for path in sorted(modules):
        if path not in module_paths:
            continue  # already reported as not_requested
        info = modules[path]
        if not info.main_files:
            if _dir_has_direct_file(repo_path, path):
                issues.append(
                    AssignmentIssue(
                        code="main_files_required",
                        path=path,
                        detail=(
                            f"module {path!r} cites no main_files, but its "
                            f"directory holds direct files; cite "
                            f"1–{MAX_MAIN_FILES} of them"
                        ),
                    )
                )
            continue
        for f in info.main_files:
            fpath = f.path
            if fpath != path and not fpath.startswith(path + "/"):
                issues.append(
                    AssignmentIssue(
                        code="main_file_outside_module",
                        path=path,
                        detail=f"main_file {fpath!r} is not under module {path!r}",
                    )
                )
                continue
            if not _is_real_file(repo_path, fpath):
                issues.append(
                    AssignmentIssue(
                        code="main_file_not_a_file",
                        path=path,
                        detail=(
                            f"main_file {fpath!r} is not a real "
                            "non-symlink file"
                        ),
                    )
                )
                continue
            owner = nearest_module_owner_of_file(fpath, module_paths)
            if owner != path:
                issues.append(
                    AssignmentIssue(
                        code="main_file_not_owned",
                        path=path,
                        detail=(
                            f"main_file {fpath!r} belongs to module "
                            f"{owner!r}, not {path!r} (nearest-owner rule)"
                        ),
                    )
                )
                continue
            if fpath in seen and seen[fpath] != path:
                issues.append(
                    AssignmentIssue(
                        code="main_file_double_claimed",
                        path=path,
                        detail=(
                            f"main_file {fpath!r} claimed by both "
                            f"{seen[fpath]!r} and {path!r}"
                        ),
                    )
                )
                continue
            seen[fpath] = path
    if claimed is not None:
        claimed.update(seen)
    return issues


__all__ = [
    "AssignmentCoverageReport",
    "AssignmentIssue",
    "CollisionPrecedence",
    "MetadataCoverageReport",
    "compute_assignment_coverage",
    "compute_collision_precedence",
    "compute_metadata_coverage",
    "format_assignment_issues",
    "nearest_module_ancestor",
    "nearest_module_owner_of_file",
    "validate_assignment_labels",
    "validate_assignment_paths",
    "validate_metadata_entries",
]
