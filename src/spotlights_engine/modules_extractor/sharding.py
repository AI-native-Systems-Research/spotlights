"""Deterministic sharding for Stage-3A MODULE/PART assignments.

Each shard labels exactly its skeleton slice. Merging is a checked disjoint
union, so shard boundaries cannot change module ownership or tree structure.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.assignments import (
    AssignmentCoverageReport,
    compute_assignment_coverage,
)
from spotlights_engine.modules_extractor.coverage import CrossArtifactError
from spotlights_engine.modules_extractor.errors import ExtractorValidationError
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentLabel,
    AssignmentTree,
    ModuleDecision,
    Skeleton,
    SkeletonNode,
)

if TYPE_CHECKING:
    from spotlights_engine.modules_extractor.extractor import ExtractorConfig

_KEY_INVALID = re.compile(r"[^A-Za-z0-9_]")
_SPINE_SUFFIX = "__spine"

NOT_SPLIT_TOP_LEVEL_ONLY = "top_level_only_mode"
NOT_SPLIT_BELOW_THRESHOLD = "below_subshard_threshold"
NOT_SPLIT_MAX_DEPTH = "max_subshard_depth_reached"
NOT_SPLIT_BUDGET = "shard_budget_exhausted"
NOT_SPLIT_NO_PROMOTABLE = "no_promotable_children"


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    return descendant == ancestor or descendant.startswith(ancestor + "/")


def _parent_path(path: str) -> str:
    head, _, _ = path.rpartition("/")
    return head


def _in_scope(path: str, root: str, pruned: frozenset[str]) -> bool:
    return _is_ancestor(root, path) and not any(
        _is_ancestor(removed, path) for removed in pruned
    )


def _prune_node(node: SkeletonNode, pruned: frozenset[str]) -> SkeletonNode:
    """Copy ``node`` while removing delegated descendant subtrees.

    Count fields remain the raw full-subtree values because the assignment
    policy must be shard-boundary neutral.
    """
    return node.model_copy(
        update={
            "children": [
                _prune_node(child, pruned)
                for child in node.children
                if not any(_is_ancestor(path, child.path) for path in pruned)
            ]
        },
        deep=True,
    )


def slice_skeleton(
    skeleton: Skeleton,
    node: SkeletonNode,
    *,
    pruned_roots: frozenset[str] = frozenset(),
) -> Skeleton:
    root = node.path
    return Skeleton(
        source_root=skeleton.source_root,
        nodes=[_prune_node(node, pruned_roots)],
        ignored=list(skeleton.ignored),
        excluded=[
            path
            for path in skeleton.excluded
            if _in_scope(path, root, pruned_roots)
        ],
        organizational_only=[
            path
            for path in skeleton.organizational_only
            if _in_scope(path, root, pruned_roots)
        ],
        skipped_symlinks=[
            path
            for path in skeleton.skipped_symlinks
            if _in_scope(path, root, pruned_roots)
        ],
        inventory_fingerprint=skeleton.inventory_fingerprint,
    )


def node_source_file_count(node: SkeletonNode) -> int:
    return node.direct_source_file_count + sum(
        node_source_file_count(child) for child in node.children
    )


def assignment_node_weight(node: SkeletonNode) -> int:
    """Every inventoried path requires one output label."""
    return 1 + sum(assignment_node_weight(child) for child in node.children)


class AssignmentShard(BaseModel):
    """One Stage-3A assignment call and its exact skeleton scope."""

    model_config = ConfigDict(extra="forbid")

    key: str
    root_path: str
    subtree: Skeleton
    is_subshard: bool = False
    parent_key: str | None = None
    is_branch_root: bool = True
    depth: int = 0
    promoted_children: list[str] = Field(default_factory=list)

    @property
    def is_spine(self) -> bool:
        return bool(self.promoted_children)

    @property
    def promotion_parent(self) -> str:
        if not self.promoted_children:
            return self.root_path
        return _parent_path(self.promoted_children[0])


class AssignmentShardPlan(BaseModel):
    """The full deterministic Stage-3A partition."""

    model_config = ConfigDict(extra="forbid")

    shards: list[AssignmentShard] = Field(default_factory=list)
    branch_subtrees: dict[str, Skeleton] = Field(default_factory=dict)
    branch_roots: dict[str, str] = Field(default_factory=dict)
    not_split_reasons: dict[str, str] = Field(default_factory=dict)

    def branch_key_of(self, shard: AssignmentShard) -> str:
        return shard.parent_key or shard.key

    def shards_of_branch(self, branch_key: str) -> list[AssignmentShard]:
        return [
            shard
            for shard in self.shards
            if self.branch_key_of(shard) == branch_key
        ]

    def branch_keys(self) -> list[str]:
        return [
            key
            for key, _ in sorted(
                self.branch_roots.items(), key=lambda item: item[1]
            )
        ]

    def primary_shard(self) -> AssignmentShard | None:
        keys = self.branch_keys()
        if not keys:
            return None
        candidates = [
            shard
            for shard in self.shards_of_branch(keys[0])
            if shard.root_path == self.branch_roots[keys[0]]
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda shard: (shard.depth, shard.key))


def assignment_shard_weight(shard: AssignmentShard) -> int:
    return sum(assignment_node_weight(node) for node in shard.subtree.nodes)


def shard_source_file_count(shard: AssignmentShard) -> int:
    return sum(node_source_file_count(node) for node in shard.subtree.nodes)


def _base_key(root_path: str) -> str:
    return _KEY_INVALID.sub("", root_path.replace("/", "__"))


def _mint_key_literal(candidate: str, used: set[str]) -> str:
    key = candidate
    suffix = 2
    while key in used:
        key = f"{candidate}_{suffix}"
        suffix += 1
    used.add(key)
    return key


def _mint_key(root_path: str, used: set[str]) -> str:
    return _mint_key_literal(_base_key(root_path) or "shard", used)


class _Budget:
    def __init__(self, remaining: int) -> None:
        self.remaining = max(0, remaining)


def _promotable(node: SkeletonNode, child_min: int) -> list[SkeletonNode]:
    return [
        child
        for child in node.children
        if assignment_node_weight(child) >= child_min
    ]


def _promotion_order(children: list[SkeletonNode]) -> list[SkeletonNode]:
    return sorted(
        children,
        key=lambda child: (
            -assignment_node_weight(child),
            -node_source_file_count(child),
            child.path,
        ),
    )


def _split_target(
    node: SkeletonNode, config: ExtractorConfig
) -> SkeletonNode | None:
    """Prefer a multi-child split, otherwise the deepest legal one-child split."""
    current = node
    last_single: SkeletonNode | None = None
    while True:
        promotable = _promotable(current, config.enrich_subshard_child_min)
        if len(promotable) >= 2:
            return current
        if len(promotable) == 1:
            last_single = current
            current = promotable[0]
            continue
        return last_single


def _refuse_split_reason(
    node: SkeletonNode,
    *,
    depth: int,
    config: ExtractorConfig,
    budget: _Budget,
) -> str | None:
    if depth + 1 > config.enrich_subshard_max_depth:
        return NOT_SPLIT_MAX_DEPTH
    if assignment_node_weight(node) <= config.enrich_subshard_threshold:
        return NOT_SPLIT_BELOW_THRESHOLD
    if _split_target(node, config) is None:
        return NOT_SPLIT_NO_PROMOTABLE
    if budget.remaining < 1:
        return NOT_SPLIT_BUDGET
    return None


def _plan_node(
    skeleton: Skeleton,
    node: SkeletonNode,
    *,
    key: str,
    branch_key: str,
    branch_root: str,
    depth: int,
    config: ExtractorConfig,
    budget: _Budget,
    used_keys: set[str],
    not_split_reasons: dict[str, str],
    allow_split: bool,
) -> list[AssignmentShard]:
    reason = (
        _refuse_split_reason(node, depth=depth, config=config, budget=budget)
        if allow_split
        else NOT_SPLIT_TOP_LEVEL_ONLY
    )
    if reason is not None:
        not_split_reasons[key] = reason
        return [
            AssignmentShard(
                key=key,
                root_path=node.path,
                subtree=slice_skeleton(skeleton, node),
                is_subshard=depth > 0,
                parent_key=branch_key if depth > 0 else None,
                is_branch_root=node.path == branch_root,
                depth=depth,
            )
        ]

    target = _split_target(node, config)
    if target is None:  # pragma: no cover
        raise ExtractorValidationError(
            f"no split target for {node.path!r} after the split gate passed",
            stage="enrich",
        )
    promotable = _promotable(target, config.enrich_subshard_child_min)
    take = min(len(promotable), budget.remaining)
    promoted = sorted(_promotion_order(promotable)[:take], key=lambda c: c.path)
    budget.remaining -= len(promoted)

    promoted_roots = frozenset(child.path for child in promoted)
    spine_key = _mint_key_literal(f"{key}{_SPINE_SUFFIX}", used_keys)
    shards = [
        AssignmentShard(
            key=spine_key,
            root_path=node.path,
            subtree=slice_skeleton(
                skeleton, node, pruned_roots=promoted_roots
            ),
            is_subshard=True,
            parent_key=branch_key,
            is_branch_root=node.path == branch_root,
            depth=depth + 1,
            promoted_children=sorted(promoted_roots),
        )
    ]
    for child in promoted:
        shards.extend(
            _plan_node(
                skeleton,
                child,
                key=_mint_key(child.path, used_keys),
                branch_key=branch_key,
                branch_root=branch_root,
                depth=depth + 1,
                config=config,
                budget=budget,
                used_keys=used_keys,
                not_split_reasons=not_split_reasons,
                allow_split=True,
            )
        )
    return shards


def derive_assignment_shards(
    skeleton: Skeleton, config: ExtractorConfig
) -> AssignmentShardPlan:
    top_nodes = sorted(skeleton.nodes, key=lambda node: node.path)
    used_keys: set[str] = set()
    branch_keys = [(node, _mint_key(node.path, used_keys)) for node in top_nodes]
    plan = AssignmentShardPlan(
        branch_subtrees={
            key: slice_skeleton(skeleton, node) for node, key in branch_keys
        },
        branch_roots={key: node.path for node, key in branch_keys},
    )
    budget = _Budget(config.enrich_max_shards - len(branch_keys))
    allow_split = config.enrich_sharding == "auto"
    for node, key in branch_keys:
        plan.shards.extend(
            _plan_node(
                skeleton,
                node,
                key=key,
                branch_key=key,
                branch_root=node.path,
                depth=0,
                config=config,
                budget=budget,
                used_keys=used_keys,
                not_split_reasons=plan.not_split_reasons,
                allow_split=allow_split,
            )
        )
    keys = [shard.key for shard in plan.shards]
    if len(set(keys)) != len(keys):  # pragma: no cover
        raise ExtractorValidationError(
            f"shard keys are not unique: {sorted(keys)}", stage="enrich"
        )
    return plan


def covers_entire_skeleton(shard: AssignmentShard, skeleton: Skeleton) -> bool:
    return shard.subtree.all_paths() == skeleton.all_paths()


def has_several_source_roots(skeleton: Skeleton) -> bool:
    return len(skeleton.nodes) >= 2


def validate_assignment_shard_scope(
    shard: AssignmentShard, fragment: AssignmentTree
) -> None:
    allowed = shard.subtree.all_paths()
    for kind, keys in (
        ("assignment", fragment.assignments.keys()),
        ("module_decisions", fragment.module_decisions.keys()),
    ):
        for path in sorted(keys):
            if path in allowed:
                continue
            if any(_is_ancestor(root, path) for root in shard.promoted_children):
                raise CrossArtifactError(
                    f"shard {shard.key!r} has a {kind} key {path!r} inside a "
                    "subtree promoted to another shard; do not label removed "
                    "subtrees"
                )
            raise CrossArtifactError(
                f"shard {shard.key!r} has a {kind} key {path!r} outside its "
                f"scope {shard.root_path!r}"
            )


def merge_assignment_fragments(
    fragments: list[tuple[AssignmentShard, AssignmentTree]],
) -> AssignmentTree:
    assignments: dict[str, AssignmentLabel] = {}
    decisions: dict[str, ModuleDecision] = {}
    assignment_owner: dict[str, str] = {}
    for shard, fragment in sorted(
        fragments, key=lambda item: (item[0].root_path, item[0].key)
    ):
        for path, label in fragment.assignments.items():
            if path in assignments:
                raise ExtractorValidationError(
                    f"shards {assignment_owner[path]!r} and {shard.key!r} both "
                    f"label {path!r}; the shard partition is broken",
                    stage="enrich",
                )
            assignments[path] = label
            assignment_owner[path] = shard.key
        for path, decision in fragment.module_decisions.items():
            if path in decisions:
                raise ExtractorValidationError(
                    f"two shards both carry a module decision for {path!r}; "
                    "the shard partition is broken",
                    stage="enrich",
                )
            if fragment.assignments.get(path) != "MODULE":
                raise ExtractorValidationError(
                    f"shard {shard.key!r} carries a module decision for "
                    f"{path!r} without labeling it MODULE; fragment "
                    "validation should have rejected this",
                    stage="enrich",
                )
            decisions[path] = decision

    module_paths = {
        path for path, label in assignments.items() if label == "MODULE"
    }
    if set(decisions) != module_paths:
        missing = sorted(module_paths - set(decisions))
        extra = sorted(set(decisions) - module_paths)
        raise ExtractorValidationError(
            "merged assignment/module-decision maps are inconsistent "
            f"(missing decisions {missing}, orphan decisions {extra}); "
            "fragment validation should have rejected this",
            stage="enrich",
        )
    return AssignmentTree(
        assignments={key: assignments[key] for key in sorted(assignments)},
        module_decisions={key: decisions[key] for key in sorted(decisions)},
    )


def assignment_branch_coverage(
    plan: AssignmentShardPlan,
    branch_key: str,
    fragments: list[tuple[AssignmentShard, AssignmentTree]],
) -> AssignmentCoverageReport:
    union = merge_assignment_fragments(fragments)
    return compute_assignment_coverage(union, plan.branch_subtrees[branch_key])


def assignment_owning_shard(
    path: str, shards: list[AssignmentShard]
) -> AssignmentShard | None:
    for shard in shards:
        if path in shard.subtree.all_paths():
            return shard
    return None


def render_assignment_shard_plan_markdown(plan: AssignmentShardPlan) -> str:
    rows = [
        (branch_key, shard)
        for branch_key in plan.branch_keys()
        for shard in plan.shards_of_branch(branch_key)
    ]
    out = [
        "# Stage-3A assignment sharding",
        "",
        f"{len(plan.shards)} shard(s) across {len(plan.branch_roots)} "
        "top-level branch(es). One row per shard, grouped by branch.",
        "",
        "Sizes are each shard's own (possibly pruned) scope:",
        "",
        "- **nodes** — every inventory node the shard labels",
        "- **files** — source files across the shard's subtree",
        "",
        "| branch | shard | kind | root | anchor | depth | nodes | files | notes |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for branch_key, shard in rows:
        kind = "spine" if shard.is_spine else (
            "sub-shard" if shard.is_subshard else "branch"
        )
        notes: list[str] = []
        if shard.promoted_children:
            promoted = ", ".join(f"`{path}`" for path in shard.promoted_children)
            notes.append(
                f"promoted {len(shard.promoted_children)} child(ren): {promoted}"
            )
            if shard.promotion_parent != shard.root_path:
                notes.append(f"split at `{shard.promotion_parent}`")
        reason = plan.not_split_reasons.get(shard.key)
        if reason is not None:
            label = "not split" if not shard.is_subshard else "not split further"
            notes.append(f"{label}: {reason}")
        out.append(
            f"| `{plan.branch_roots[branch_key]}` | `{shard.key}` | {kind} "
            f"| `{shard.root_path}` | "
            f"{'yes' if shard.is_branch_root else 'no'} | {shard.depth} "
            f"| {assignment_shard_weight(shard)} "
            f"| {shard_source_file_count(shard)} "
            f"| {'; '.join(notes)} |"
        )
    out.append(
        f"| **total** | {len(plan.shards)} shard(s) | | | | | "
        f"{sum(assignment_shard_weight(shard) for shard in plan.shards)} | "
        f"{sum(shard_source_file_count(shard) for shard in plan.shards)} | |"
    )
    out.append("")
    return "\n".join(out)


__all__ = [
    "NOT_SPLIT_BELOW_THRESHOLD",
    "NOT_SPLIT_BUDGET",
    "NOT_SPLIT_MAX_DEPTH",
    "NOT_SPLIT_NO_PROMOTABLE",
    "NOT_SPLIT_TOP_LEVEL_ONLY",
    "AssignmentShard",
    "AssignmentShardPlan",
    "assignment_branch_coverage",
    "assignment_node_weight",
    "assignment_owning_shard",
    "assignment_shard_weight",
    "covers_entire_skeleton",
    "derive_assignment_shards",
    "has_several_source_roots",
    "merge_assignment_fragments",
    "node_source_file_count",
    "render_assignment_shard_plan_markdown",
    "shard_source_file_count",
    "slice_skeleton",
    "validate_assignment_shard_scope",
]
