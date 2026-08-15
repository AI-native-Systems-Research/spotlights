"""Pure derivation for the assignment contract (Stage-3 v2).

Owner resolution, provisional (V5) and final `ProjectTree` construction,
deterministic Stage-3B metadata batching, territory metrics, and structured
warning-only lints. No LLM calls, no filesystem access, no writes — everything
here is unit-testable in isolation and deterministic for fixed inputs, so the
same labels produce the same metadata scopes and the same public tree under
`single`, top-level, and recursive assignment partitioning.

There is deliberately **no single-child rewrite** here: old Rule 4 was an
extractor aesthetic, not a public schema invariant, and cannot coexist with
"large paths must be modules" or a small module's accepted `keep_reason`. The
public `ProjectTree` schema already allows a module with one child.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.assignments import (
    compute_collision_precedence,
    format_assignment_issues,
    nearest_module_ancestor,
    validate_assignment_labels,
)
from spotlights_engine.modules_extractor.constants import (
    MAX_REPRESENTATIVE_FILES,
    MIN_LEAF_FILES,
)
from spotlights_engine.modules_extractor.coverage import CrossArtifactError
from spotlights_engine.modules_extractor.skeleton import is_entry_file
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentOrigin,
    AssignmentTree,
    ModuleInfo,
    ResolvedAssignmentTree,
    Skeleton,
)
from spotlights_engine.schemas.project import (
    ProjectTree,
    Repository,
    _normalize_module_segment,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from spotlights_engine.modules_extractor.extractor import ExtractorConfig


# ── Resolution ────────────────────────────────────────────────────────────


def resolve_assignments(
    tree: AssignmentTree, skeleton: Skeleton, merge_threshold: int
) -> ResolvedAssignmentTree:
    """Validate a total `AssignmentTree` against the full skeleton and derive
    the canonical owners/origins maps.

    Runs the collision preflight and full V2/V3 itself — the resolver never
    trusts a caller to have validated, and never trusts a persisted owner or
    origin map. Raises `CrossArtifactError` with batched issue text on any
    violation.
    """
    precedence = compute_collision_precedence(skeleton)
    if precedence.fatal:
        classes = "; ".join(
            ", ".join(repr(p) for p in group)
            for group in precedence.top_level_collisions
        )
        raise CrossArtifactError(
            "top-level skeleton paths collide on their normalized public "
            f"qualified name and cannot both be modules: {classes}. This is a "
            "limitation of the public naming contract, not a repairable model "
            "choice."
        )

    inventory = skeleton.all_paths()
    keys = set(tree.assignments)
    if keys != inventory:
        missing = sorted(inventory - keys)
        extra = sorted(keys - inventory)
        raise CrossArtifactError(
            "assignments are not total over the skeleton inventory "
            f"(missing {missing[:10]}{'…' if len(missing) > 10 else ''}, "
            f"extra {extra[:10]}{'…' if len(extra) > 10 else ''})"
        )

    issues = validate_assignment_labels(
        tree,
        skeleton,
        merge_threshold=merge_threshold,
        precedence=precedence,
        anchor_paths={n.path for n in skeleton.nodes},
        require_owner_in_scope=True,
    )
    if issues:
        raise CrossArtifactError(format_assignment_issues(issues))

    nodes = {n.path: n for n in skeleton.iter_nodes()}
    top_level = {n.path for n in skeleton.nodes}
    module_paths = tree.module_paths()

    owners: dict[str, str] = {}
    origins: dict[str, AssignmentOrigin] = {}
    for path, label in tree.assignments.items():
        if label == "MODULE":
            owners[path] = path
            if path in top_level:
                origins[path] = "top_level_anchor"
            elif nodes[path].subtree_source_file_count > merge_threshold:
                origins[path] = "size"
            else:
                origins[path] = "independent"
        else:
            owner = nearest_module_ancestor(path, module_paths)
            if owner is None:  # pragma: no cover - V3 rejected this already
                raise CrossArtifactError(
                    f"PART {path!r} has no module ancestor after validation"
                )
            owners[path] = owner
            origins[path] = (
                "qn_collision_part" if path in precedence.forced_part else "part"
            )

    return ResolvedAssignmentTree(
        assignments=dict(tree.assignments),
        module_decisions=dict(tree.module_decisions),
        owners=owners,
        origins=origins,
        collision_precedence=dict(precedence.forced_part),
        merge_threshold=merge_threshold,
    )


def verify_resolved_assignments(
    resolved: ResolvedAssignmentTree, skeleton: Skeleton
) -> None:
    """Recompute every derived map from the skeleton and compare.

    Persisted owners/origins/collision-precedence are audit data, not trusted
    input — a Stage-5 (or artifact-load) tamper check. Raises
    `CrossArtifactError` naming the first disagreement.
    """
    recomputed = resolve_assignments(
        AssignmentTree(
            assignments=dict(resolved.assignments),
            module_decisions=dict(resolved.module_decisions),
        ),
        skeleton,
        resolved.merge_threshold,
    )
    for field_name in ("owners", "origins", "collision_precedence"):
        stored = getattr(resolved, field_name)
        fresh = getattr(recomputed, field_name)
        if stored != fresh:
            diff = sorted(
                k
                for k in set(stored) | set(fresh)
                if stored.get(k) != fresh.get(k)
            )
            raise CrossArtifactError(
                f"persisted {field_name} disagree with recomputation for "
                f"{diff[:10]}{'…' if len(diff) > 10 else ''}"
            )


# ── Public-tree derivation (V5) ───────────────────────────────────────────


def _module_tree_dicts(
    module_paths: list[str], info: Mapping[str, ModuleInfo] | None
) -> list[dict[str, Any]]:
    """Nest `module_paths` by nearest-module-ancestor into `ProjectTree` dicts.

    `name` is derived from the path basename with `_normalize_module_segment`;
    description/main files come from the exact matching `ModuleInfo` when
    given (empty otherwise — the public `Module` schema allows that for the
    provisional gate).
    """
    module_set = set(module_paths)
    children: dict[str | None, list[str]] = {}
    for p in sorted(module_set):
        children.setdefault(nearest_module_ancestor(p, module_set), []).append(p)

    def _build(path: str) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": _normalize_module_segment(PurePosixPath(path).name),
            "path": path,
        }
        if info is not None:
            mi = info.get(path)
            if mi is not None:
                d["description"] = mi.description
                d["main_files"] = [
                    {"path": f.path, "role": f.role} for f in mi.main_files
                ]
        subs = [_build(c) for c in children.get(path, [])]
        if subs:
            d["submodules"] = subs
        return d

    return [_build(p) for p in children.get(None, [])]


def derive_module_forest(
    repository: Repository, tree: AssignmentTree
) -> ProjectTree:
    """Path-only provisional `ProjectTree` for one fragment (the V5 gate).

    Nests only the fragment's **local** `MODULE` paths; external `PART` owners
    are not required, so a nested shard whose root resolved `PART` still gets
    its lexical/normalized-name/source-root/qualified-name validation. Raises
    the underlying `ValueError` from the unchanged public validators.
    """
    return ProjectTree.model_validate(
        {
            "repository": repository.model_dump(),
            "modules": _module_tree_dicts(sorted(tree.module_paths()), None),
        }
    )


def derive_project_tree(
    repository: Repository,
    resolved: ResolvedAssignmentTree,
    metadata: Mapping[str, ModuleInfo],
) -> ProjectTree:
    """The real public tree: resolved labels + the exact metadata union.

    Metadata cannot change paths or names; a key mismatch is an internal
    partition error, not a repairable model response.
    """
    module_paths = resolved.module_paths()
    if set(metadata) != module_paths:
        raise CrossArtifactError(
            "metadata union keys do not equal the resolved MODULE set "
            f"(missing {sorted(module_paths - set(metadata))[:10]}, "
            f"extra {sorted(set(metadata) - module_paths)[:10]})"
        )
    return ProjectTree.model_validate(
        {
            "repository": repository.model_dump(),
            "modules": _module_tree_dicts(sorted(module_paths), metadata),
        }
    )


# ── Territories ───────────────────────────────────────────────────────────


def resolve_territories(resolved: ResolvedAssignmentTree) -> dict[str, list[str]]:
    """Module path -> sorted list of skeleton paths it owns (itself included)."""
    out: dict[str, list[str]] = {m: [] for m in resolved.module_paths()}
    for path, owner in resolved.owners.items():
        out[owner].append(path)
    return {m: sorted(paths) for m, paths in out.items()}


def territory_source_file_counts(
    resolved: ResolvedAssignmentTree, skeleton: Skeleton
) -> dict[str, int]:
    """Module -> sum of `direct_source_file_count` over its owned paths."""
    counts = {n.path: n.direct_source_file_count for n in skeleton.iter_nodes()}
    return {
        m: sum(counts.get(p, 0) for p in paths)
        for m, paths in resolve_territories(resolved).items()
    }


def emitted_path_territories(
    emitted_paths: set[str], skeleton: Skeleton
) -> tuple[dict[str, int], list[str]]:
    """v1-comparable territories under the same physical rule.

    Assigns every skeleton path to its deepest emitted ancestor (or itself
    when emitted) and sums direct source counts. Paths with no emitted
    ancestor are returned separately as `unowned_optional` rather than dropped
    or assigned to a sibling.
    """
    territory: dict[str, int] = {p: 0 for p in emitted_paths}
    unowned: list[str] = []
    for node in skeleton.iter_nodes():
        owner = (
            node.path
            if node.path in emitted_paths
            else nearest_module_ancestor(node.path, emitted_paths)
        )
        if owner is None:
            unowned.append(node.path)
        else:
            territory[owner] += node.direct_source_file_count
    return territory, sorted(unowned)


def module_children(resolved: ResolvedAssignmentTree) -> dict[str, list[str]]:
    """Module -> sorted direct descendant module roots (nearest-ancestor rule)."""
    module_paths = resolved.module_paths()
    out: dict[str, list[str]] = {m: [] for m in module_paths}
    for m in module_paths:
        parent = nearest_module_ancestor(m, module_paths)
        if parent is not None:
            out[parent].append(m)
    return {m: sorted(cs) for m, cs in out.items()}


# ── Metadata batching ─────────────────────────────────────────────────────


class ModuleMetadataScope(BaseModel):
    """What one Stage-3B prompt tells the model about one requested module."""

    model_config = ConfigDict(extra="forbid")

    path: str
    origin: str
    keep_reason: str | None = None
    territory_node_count: int = Field(ge=0)
    territory_source_file_count: int = Field(ge=0)
    direct_source_file_count: int = Field(ge=0)
    child_module_paths: list[str] = Field(default_factory=list)
    representative_files: list[str] = Field(default_factory=list)


class MetadataBatch(BaseModel):
    """One Stage-3B call: the requested module set plus per-module scopes."""

    model_config = ConfigDict(extra="forbid")

    key: str
    module_paths: list[str] = Field(default_factory=list)
    weight: int = Field(ge=0)
    use_full_timeout: bool = False
    scopes: list[ModuleMetadataScope] = Field(default_factory=list)


class MetadataBatchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batches: list[MetadataBatch] = Field(default_factory=list)


def _territory_representative_files(
    owned_paths: list[str], skeleton: Skeleton
) -> list[str]:
    """Reading seeds for one territory: union of the owned nodes'
    representative files, re-ranked entry-file-first then lexical — the same
    ordering the skeleton uses — capped at `MAX_REPRESENTATIVE_FILES`."""
    nodes = {n.path: n for n in skeleton.iter_nodes()}
    pool = sorted(
        {
            f
            for p in owned_paths
            for f in (nodes[p].representative_files if p in nodes else [])
        }
    )
    entry = [f for f in pool if is_entry_file(f)]
    rest = [f for f in pool if not is_entry_file(f)]
    return (entry + rest)[:MAX_REPRESENTATIVE_FILES]


def derive_metadata_batches(
    resolved: ResolvedAssignmentTree,
    skeleton: Skeleton,
    config: ExtractorConfig,
) -> MetadataBatchPlan:
    """Deterministic Stage-3B batching, independent of Stage-3A shard shapes.

    Each final module territory is one indivisible item weighted by its owned
    skeleton-node count. `bin_count = min(n, enrich_max_shards,
    ceil(total_weight / enrich_subshard_threshold))` with a floor of one;
    items sorted by descending weight then path go to the currently
    least-loaded bin (ties by bin index). An empty module set produces no
    batch. A batch whose final weight exceeds the threshold uses the full
    Stage-3 timeout.
    """
    territories = resolve_territories(resolved)
    if not territories:
        return MetadataBatchPlan(batches=[])

    weights = {m: len(paths) for m, paths in territories.items()}
    total_weight = sum(weights.values())
    bin_count = max(
        1,
        min(
            len(weights),
            config.enrich_max_shards,
            math.ceil(total_weight / config.enrich_subshard_threshold),
        ),
    )

    bins: list[tuple[int, list[str]]] = [(0, []) for _ in range(bin_count)]
    for module, weight in sorted(weights.items(), key=lambda kv: (-kv[1], kv[0])):
        idx = min(range(bin_count), key=lambda i: (bins[i][0], i))
        load, members = bins[idx]
        bins[idx] = (load + weight, [*members, module])

    source_counts = territory_source_file_counts(resolved, skeleton)
    children = module_children(resolved)
    nodes = {n.path: n for n in skeleton.iter_nodes()}

    batches: list[MetadataBatch] = []
    for idx, (load, members) in enumerate(bins):
        if not members:
            continue
        module_paths = sorted(members)
        scopes = [
            ModuleMetadataScope(
                path=m,
                origin=resolved.origins[m],
                keep_reason=resolved.module_decisions[m].keep_reason,
                territory_node_count=len(territories[m]),
                territory_source_file_count=source_counts[m],
                direct_source_file_count=(
                    nodes[m].direct_source_file_count if m in nodes else 0
                ),
                child_module_paths=children[m],
                representative_files=_territory_representative_files(
                    territories[m], skeleton
                ),
            )
            for m in module_paths
        ]
        batches.append(
            MetadataBatch(
                key=f"batch_{idx:02d}",
                module_paths=module_paths,
                weight=load,
                use_full_timeout=load > config.enrich_subshard_threshold,
                scopes=scopes,
            )
        )
    return MetadataBatchPlan(batches=batches)


# ── Lints ─────────────────────────────────────────────────────────────────


class AssignmentLint(BaseModel):
    """One deterministic warning. Lints never change labels and never fail
    extraction."""

    model_config = ConfigDict(extra="forbid")

    code: Literal["tiny_leaf", "fragmented_children"]
    path: str
    territory_source_file_count: int | None = None
    leaf_child_count: int | None = None
    mean_leaf_territory: float | None = None
    message: str


def compute_assignment_lints(
    resolved: ResolvedAssignmentTree, skeleton: Skeleton
) -> list[AssignmentLint]:
    """`tiny_leaf` and `fragmented_children`, per the plan's exact definitions."""
    counts = territory_source_file_counts(resolved, skeleton)
    children = module_children(resolved)
    leaves = {m for m, cs in children.items() if not cs}
    lints: list[AssignmentLint] = []

    for m in sorted(leaves):
        if resolved.origins.get(m) == "top_level_anchor":
            continue
        if counts[m] < MIN_LEAF_FILES:
            lints.append(
                AssignmentLint(
                    code="tiny_leaf",
                    path=m,
                    territory_source_file_count=counts[m],
                    message=(
                        f"leaf module {m!r} owns only {counts[m]} source "
                        f"file(s) (< {MIN_LEAF_FILES})"
                    ),
                )
            )

    for m in sorted(children):
        leaf_children = [c for c in children[m] if c in leaves]
        if len(leaf_children) < 2:
            continue
        mean = sum(counts[c] for c in leaf_children) / len(leaf_children)
        if mean < MIN_LEAF_FILES:
            lints.append(
                AssignmentLint(
                    code="fragmented_children",
                    path=m,
                    leaf_child_count=len(leaf_children),
                    mean_leaf_territory=round(mean, 2),
                    message=(
                        f"module {m!r} has {len(leaf_children)} leaf-module "
                        f"children averaging {mean:.1f} source files "
                        f"(< {MIN_LEAF_FILES})"
                    ),
                )
            )
    return lints


__all__ = [
    "AssignmentLint",
    "MetadataBatch",
    "MetadataBatchPlan",
    "ModuleMetadataScope",
    "compute_assignment_lints",
    "derive_metadata_batches",
    "derive_module_forest",
    "derive_project_tree",
    "emitted_path_territories",
    "module_children",
    "resolve_assignments",
    "resolve_territories",
    "territory_source_file_counts",
    "verify_resolved_assignments",
]
