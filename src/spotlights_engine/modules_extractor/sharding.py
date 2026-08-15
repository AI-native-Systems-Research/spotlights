"""Deterministic Stage-3/4 sharding: derive, validate, and merge enrichment shards.

Pure Python — no LLM, no filesystem writes — so every rule here is unit-testable
in isolation. See `design/module_extraction_fix_impl__top_level_plan.md`.

The base two-phase plan ran Stage 3 as one monolithic Claude call over the whole
repository. On a large monorepo that single request times out before anything is
persisted. This module partitions enrichment by **top-level skeleton node**, with
a size-gated, bounded *recursive sub-sharding* of any branch big enough to
reproduce the timeout on its own, and merges the fragments back.

Two contract-specific implementations coexist while the
`ExtractorConfig.contract` migration flag exists
(`design/module_extractor_simplified.md` §2.7):

- **Tree contract** (`EnrichShard`/`ShardPlan`): fragments are `EnrichedTree`s
  merged by concatenation + re-parenting, and the merged tree is handed to the
  **unchanged** `validate_enriched_tree` / `compute_coverage` in Stage 5.
  Three traps are created by splitting a branch — a single-child spine, a
  spine `main_file` that lands under a promoted child, and a chain-split spine
  that folds away the very node its promoted children re-attach to — and each
  is closed here at derivation or shard-validation time, because the merge is
  pure Python and Stage 5 has no repair. (A fourth, a spine with no legal
  `main_file` at all, is closed at the source: Rule 3 lets a pure container
  directory cite none.)

- **Assignment contract** (`AssignmentShard`/`AssignmentShardPlan`): fragments
  are label dictionaries and the merge is a checked disjoint union, so none of
  the tree traps exist. Weights count **all** inventory nodes (every path is
  output), a nested shard root may legally resolve `PART`, a split needs only
  one budget slot, and a one-promotable-child split is allowed — `_split
  target` selection becomes a packing optimization, not a correctness gate.

The two families share only the pure skeleton-slicing, key, and weight-neutral
helpers whose invariants match.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.modules_extractor.assignments import (
    AssignmentCoverageReport,
    compute_assignment_coverage,
)
from spotlights_engine.modules_extractor.coverage import (
    CoverageReport,
    CrossArtifactError,
    compute_coverage,
)
from spotlights_engine.modules_extractor.errors import ExtractorValidationError
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentLabel,
    AssignmentTree,
    EnrichedSubmodule,
    EnrichedTopModule,
    EnrichedTree,
    ModuleDecision,
    Skeleton,
    SkeletonNode,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle guard only
    from spotlights_engine.modules_extractor.extractor import ExtractorConfig

_KEY_INVALID = re.compile(r"[^A-Za-z0-9_]")
_SPINE_SUFFIX = "__spine"

# `not_split_reason` vocabulary recorded in `03_enrich/shards.json`.
NOT_SPLIT_TOP_LEVEL_ONLY = "top_level_only_mode"
NOT_SPLIT_BELOW_THRESHOLD = "below_subshard_threshold"
NOT_SPLIT_MAX_DEPTH = "max_subshard_depth_reached"
NOT_SPLIT_ONE_PROMOTABLE = "fewer_than_two_promotable_children"
NOT_SPLIT_BUDGET = "shard_budget_exhausted"
# Assignment mode only: Rule 4 is retired, so one promotable child suffices to
# split; the remaining refusal is having no promotable child at all.
NOT_SPLIT_NO_PROMOTABLE = "no_promotable_children"


# ── Models ────────────────────────────────────────────────────────────────


class EnrichShard(BaseModel):
    """One unit of Stage-3 enrichment: a subtree the model sees and owns.

    A shard may emit or fold only paths at or under `root_path`, which makes
    per-shard coverage a local property and the merge a disjoint union by
    construction.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    """Unique, artifact-safe id; also the shard's directory name."""

    root_path: str
    """The branch (or sub-branch) root this shard owns."""

    subtree: Skeleton
    """A `Skeleton` restricted to this shard's scope. For a spine, the promoted
    children's subtrees are pruned out."""

    is_subshard: bool = False
    parent_key: str | None = None
    """The enclosing top-level BRANCH key (never the enclosing *sub-shard*),
    or None for an un-split top-level shard. Grouping code keys on
    `shard.parent_key or shard.key`."""

    owns_root: bool = True
    """True for a shard that emits `root_path` as a *spine* — i.e. an un-split
    top-level shard or the spine of a split node. False for a child sub-shard,
    whose top module is demoted to a submodule at merge."""

    depth: int = 0
    """0 for a top-level shard, +1 per split level. Drives the bottom-up
    re-assembly order in `merge_fragments`."""

    promoted_children: list[str] = Field(default_factory=list)
    """Child roots delegated to child sub-shards and pruned from this shard's
    `subtree`; `[]` unless this shard is a spine. They are children of
    `promotion_parent`, which is usually — but not always — `root_path`."""

    @property
    def is_spine(self) -> bool:
        return self.owns_root and bool(self.promoted_children)

    @property
    def promotion_parent(self) -> str:
        """The node whose children were promoted away, i.e. the one directory in
        this shard's subtree whose `children` are incomplete.

        `root_path` for a shard split at its own root, and a descendant of it
        when derivation had to walk down a one-child chain to find a splittable
        node (see `_split_target`). `root_path` for a shard with nothing
        promoted, so callers need no special case.
        """
        if not self.promoted_children:
            return self.root_path
        return _parent_path(self.promoted_children[0])

    @property
    def rule4_exempt_paths(self) -> set[str]:
        """Paths whose zero-or-≥2-children rule this shard cannot check itself.

        Exactly the promotion parent: its guaranteeing children are pruned from
        this subtree and only come back at merge. Every *other* module here —
        including any chain node between `root_path` and the promotion parent —
        sees all of its children and is held to Rule 4 locally, which is what
        keeps the merged tree from acquiring a single-child parent that Stage 5
        would reject with no repair left.
        """
        return {self.promotion_parent} if self.is_spine else set()


class ShardPlan(BaseModel):
    """The derived plan. Carries the two derivation-time facts no individual
    shard holds: the *unpruned* per-branch subtree the branch precheck compares
    against, and the not-split ledger."""

    model_config = ConfigDict(extra="forbid")

    shards: list[EnrichShard] = Field(default_factory=list)
    branch_subtrees: dict[str, Skeleton] = Field(default_factory=dict)
    branch_roots: dict[str, str] = Field(default_factory=dict)
    """branch key -> the branch's `root_path`."""
    not_split_reasons: dict[str, str] = Field(default_factory=dict)

    def branch_key_of(self, shard: EnrichShard) -> str:
        return shard.parent_key or shard.key

    def shards_of_branch(self, branch_key: str) -> list[EnrichShard]:
        return [s for s in self.shards if self.branch_key_of(s) == branch_key]

    def branch_keys(self) -> list[str]:
        """Branch keys ordered by the branch's `root_path` (POSIX lexicographic)."""
        return [k for k, _ in sorted(self.branch_roots.items(), key=lambda kv: kv[1])]

    def primary_shard(self) -> EnrichShard | None:
        """The branch-owning shard of the first top-level branch by `root_path`.

        Selection is on `owns_root`/`depth`, never on lexicographic `key`:
        sorting `vllm`'s sub-shards by key gives `vllm__engine` < `vllm__spine`,
        which would make an arbitrary child primary.
        """
        keys = self.branch_keys()
        if not keys:
            return None
        owners = [
            s
            for s in self.shards_of_branch(keys[0])
            if s.owns_root and s.root_path == self.branch_roots[keys[0]]
        ]
        if not owners:
            return None
        return min(owners, key=lambda s: (s.depth, s.key))


# ── Skeleton slicing ──────────────────────────────────────────────────────


def _is_ancestor(ancestor: str, descendant: str) -> bool:
    """Physical-path ancestry, inclusive of equality."""
    return descendant == ancestor or descendant.startswith(ancestor + "/")


def _parent_path(path: str) -> str:
    """The POSIX parent of a source-root-relative path (`""` for a top-level
    one). Plain string surgery — these paths are always `/`-separated and
    already normalized by the skeleton."""
    head, _, _ = path.rpartition("/")
    return head


def _in_scope(path: str, root: str, pruned: frozenset[str]) -> bool:
    if not _is_ancestor(root, path):
        return False
    return not any(_is_ancestor(p, path) for p in pruned)


def _prune_node(node: SkeletonNode, pruned: frozenset[str]) -> SkeletonNode:
    """Deep copy of `node` with every pruned child subtree removed.

    Every other field — including `required`, `required_reasons`,
    `source_child_count`, and `subtree_source_file_count` — is copied
    **verbatim**. For a spine this deliberately
    leaves `source_child_count` disagreeing with `len(children)`: the reasons
    were derived from the original counts, and rewriting one field without the
    other would produce an incoherent node. The spine's `SCOPE` block is what
    explains the gap to the model.
    """
    return node.model_copy(
        update={
            "children": [
                _prune_node(c, pruned)
                for c in node.children
                if not any(_is_ancestor(p, c.path) for p in pruned)
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
    """A `Skeleton` carrying only `node` and its descendants.

    `source_root`, `ignored`, and `inventory_fingerprint` are copied verbatim
    for provenance — `ignored` is a global list of ignored directory *names*,
    not branch-relative paths, so it must not be "sliced" to `[]`, and the
    fingerprint is neither recomputed nor re-validated per shard.
    """
    root = node.path
    return Skeleton(
        source_root=skeleton.source_root,
        nodes=[_prune_node(node, pruned_roots)],
        ignored=list(skeleton.ignored),
        excluded=[p for p in skeleton.excluded if _in_scope(p, root, pruned_roots)],
        organizational_only=[
            p for p in skeleton.organizational_only if _in_scope(p, root, pruned_roots)
        ],
        skipped_symlinks=[
            p for p in skeleton.skipped_symlinks if _in_scope(p, root, pruned_roots)
        ],
        inventory_fingerprint=skeleton.inventory_fingerprint,
    )


# ── Weights ───────────────────────────────────────────────────────────────


def node_weight(node: SkeletonNode) -> int:
    """A node's deterministic size: the number of `required` nodes in its subtree."""
    return (1 if node.required else 0) + sum(node_weight(c) for c in node.children)


def node_source_file_count(node: SkeletonNode) -> int:
    """Total direct source files across the subtree — the promotion tie-break."""
    return node.direct_source_file_count + sum(
        node_source_file_count(c) for c in node.children
    )


def shard_source_file_count(shard: EnrichShard | AssignmentShard) -> int:
    """Total source files across the shard's (possibly pruned) subtree."""
    return sum(node_source_file_count(n) for n in shard.subtree.nodes)


def shard_weight(shard: EnrichShard) -> int:
    """The weight of what this shard actually enriches: the required-node count
    of its own (possibly pruned) subtree.

    For a spine this is the *residual* weight after the promoted children were
    pruned out — the number a per-shard deadline must be sized against, not the
    branch's original weight. Derivation aims to keep it at or below
    `enrich_subshard_threshold`, but four paths can leave it above: a wide flat
    branch with no promotable children, a split refused at the depth cap, a
    branch starved by the shard budget, and a spine left holding promotable
    children the budget could not take.
    """
    return sum(node_weight(n) for n in shard.subtree.nodes)


# ── Key derivation ────────────────────────────────────────────────────────


def _base_key(root_path: str) -> str:
    return _KEY_INVALID.sub("", root_path.replace("/", "__"))


def _mint_key_literal(candidate: str, used: set[str]) -> str:
    key = candidate
    n = 2
    while key in used:
        key = f"{candidate}_{n}"
        n += 1
    used.add(key)
    return key


def _mint_key(root_path: str, used: set[str]) -> str:
    """`root_path` with `/`→`__` and non-`[A-Za-z0-9_]` dropped, disambiguated
    with a numeric suffix on the rare collision."""
    return _mint_key_literal(_base_key(root_path) or "shard", used)


# ── Derivation ────────────────────────────────────────────────────────────


class _Budget:
    """Remaining *extra* shards derivation may create before hitting
    `enrich_max_shards`. Depth-2 recursion draws from whatever the shallower
    levels leave unused, so it can never exceed the cap either."""

    def __init__(self, remaining: int) -> None:
        self.remaining = max(0, remaining)


def _promotable(node: SkeletonNode, child_min: int) -> list[SkeletonNode]:
    return [c for c in node.children if node_weight(c) >= child_min]


def _promotion_order(children: list[SkeletonNode]) -> list[SkeletonNode]:
    """Heaviest first. Total and derived from the skeleton alone, so the chosen
    set is deterministic; the source-file count is the one tie-break."""
    return sorted(
        children,
        key=lambda c: (-node_weight(c), -node_source_file_count(c), c.path),
    )


def _split_target(node: SkeletonNode, config: ExtractorConfig) -> SkeletonNode | None:
    """The descendant whose children this split promotes, or None when there is
    no such node and `node` must stay whole.

    Usually `node` itself. But a node with exactly one promotable child cannot
    be split *at that node*: promoting the one child leaves a single-child spine
    that Stage-5 Rule 4 rejects with no possible repair. Refusing outright is
    what pinned vLLM's `rust/` — `rust` → `rust/src` → 13 crates, 71 required
    nodes — to one monolithic shard, which then blew its per-shard deadline. So
    walk *down* the one-child chain instead and promote from the first
    descendant that has two promotable children of its own. The chain nodes stay
    with the spine, which is what makes this safe: they keep every child they
    have, so the merged tree's Rule 4 is decided inside the shard, where the
    bounded repair can still fix it.

    Descending past a node with a single *child* is the one thing this must not
    do. Such a node would be left holding exactly one emitted child after the
    merge — the promoted subtrees re-attach under the descendant, not under it —
    and its only escape, folding the chain node away, is denied by
    `validate_promotion_parent`. That is a deadlock, so refuse the split
    instead; the branch stays monolithic exactly as before.

    The walk is unbounded and still cannot leave the spine large: a sibling
    heavy enough to matter is by definition promotable, so it would have ended
    the descent at the branch above. Every subtree the chain leaves with the
    spine weighs less than `enrich_subshard_child_min`.
    """
    current = node
    while True:
        promotable = _promotable(current, config.enrich_subshard_child_min)
        if len(promotable) >= 2:
            return current
        if len(promotable) == 1 and len(current.children) >= 2:
            current = promotable[0]
            continue
        return None


def _refuse_split_reason(
    node: SkeletonNode,
    *,
    depth: int,
    config: ExtractorConfig,
    budget: _Budget,
) -> str | None:
    """Why `node` must not be split, or None when all gates pass.

    One gate is a hard **correctness** requirement, not tuning: a split with no
    `_split_target` deterministically produces a single-child spine that Stage-5
    Rule 4 rejects unrecoverably. It is *not* satisfied by `node`'s own children
    alone — a one-child chain below it can still offer a legal split point.

    A root owning no direct source file used to be refused here as well, because
    its spine had no legal `main_file`: every file in the branch belongs to an
    emitted promoted child. Rule 3 now exempts a *pure container* directory — one
    holding no direct file at all — from citing anything, and a root that holds
    non-source files (a `docker/` of Dockerfiles) may cite those, so every such
    spine has a legal answer and the gate is gone. It was the one thing keeping
    the typical Go `pkg/` — often a repo's largest branch by far — from being
    sub-sharded at all.
    """
    if depth + 1 > config.enrich_subshard_max_depth:
        return NOT_SPLIT_MAX_DEPTH
    if node_weight(node) <= config.enrich_subshard_threshold:
        return NOT_SPLIT_BELOW_THRESHOLD
    if _split_target(node, config) is None:
        return NOT_SPLIT_ONE_PROMOTABLE
    if budget.remaining < 2:
        return NOT_SPLIT_BUDGET
    return None


def _plan_node(
    skeleton: Skeleton,
    node: SkeletonNode,
    *,
    key: str,
    branch_key: str,
    depth: int,
    config: ExtractorConfig,
    budget: _Budget,
    used_keys: set[str],
    not_split_reasons: dict[str, str],
    allow_split: bool,
) -> list[EnrichShard]:
    """Shards covering `node`'s subtree, splitting recursively when the
    three-part gate allows it."""
    if allow_split:
        reason = _refuse_split_reason(
            node, depth=depth, config=config, budget=budget
        )
    else:
        reason = NOT_SPLIT_TOP_LEVEL_ONLY

    if reason is not None:
        not_split_reasons[key] = reason
        return [
            EnrichShard(
                key=key,
                root_path=node.path,
                subtree=slice_skeleton(skeleton, node),
                is_subshard=depth > 0,
                parent_key=branch_key if depth > 0 else None,
                owns_root=depth == 0,
                depth=depth,
                promoted_children=[],
            )
        ]

    # The promotion parent is `node` itself unless derivation had to walk down a
    # one-child chain to reach a splittable node; `node` still owns the shard
    # either way, and the chain in between rides along with the spine.
    target = _split_target(node, config)
    if target is None:  # pragma: no cover - defensive; the gate above cleared it
        raise ExtractorValidationError(
            f"no split target for {node.path!r} after the split gate passed",
            stage="enrich",
        )
    promotable = _promotable(target, config.enrich_subshard_child_min)
    take = min(len(promotable), budget.remaining)
    promoted = sorted(_promotion_order(promotable)[:take], key=lambda c: c.path)
    budget.remaining -= len(promoted)

    promoted_roots = frozenset(c.path for c in promoted)
    spine_key = _mint_key_literal(f"{key}{_SPINE_SUFFIX}", used_keys)
    shards = [
        EnrichShard(
            key=spine_key,
            root_path=node.path,
            subtree=slice_skeleton(skeleton, node, pruned_roots=promoted_roots),
            is_subshard=True,
            parent_key=branch_key,
            owns_root=True,
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
                depth=depth + 1,
                config=config,
                budget=budget,
                used_keys=used_keys,
                not_split_reasons=not_split_reasons,
                allow_split=True,
            )
        )
    return shards


def derive_enrich_shards(skeleton: Skeleton, config: ExtractorConfig) -> ShardPlan:
    """Partition Stage-3 enrichment into shards, deterministically.

    The primary partition is the skeleton's top-level nodes, sorted by
    `root_path`. A branch large enough to reproduce the monolithic timeout on
    its own is additionally sub-sharded into a spine + child sub-shards, but
    only when the three-part gate (`_refuse_split_reason`) allows it. The split
    point need not be the branch root: `_split_target` walks down a one-child
    chain to the first node that can legally be split, so the promoted children
    may be deeper than the spine's own root.

    A repo with a single top-level source-bearing node under the size threshold
    yields exactly one shard — today's single call.
    """
    top_nodes = sorted(skeleton.nodes, key=lambda n: n.path)
    used_keys: set[str] = set()
    branch_keys: list[tuple[SkeletonNode, str]] = [
        (n, _mint_key(n.path, used_keys)) for n in top_nodes
    ]

    plan = ShardPlan(
        branch_subtrees={k: slice_skeleton(skeleton, n) for n, k in branch_keys},
        branch_roots={k: n.path for n, k in branch_keys},
    )

    # Each branch already contributes one shard; the cap bounds the *extra*
    # shards splitting may add.
    budget = _Budget(config.enrich_max_shards - len(branch_keys))
    allow_split = config.enrich_sharding == "auto"

    for node, key in branch_keys:
        plan.shards.extend(
            _plan_node(
                skeleton,
                node,
                key=key,
                branch_key=key,
                depth=0,
                config=config,
                budget=budget,
                used_keys=used_keys,
                not_split_reasons=plan.not_split_reasons,
                allow_split=allow_split,
            )
        )

    keys = [s.key for s in plan.shards]
    if len(set(keys)) != len(keys):  # pragma: no cover - defensive
        raise ExtractorValidationError(
            f"shard keys are not unique: {sorted(keys)}", stage="enrich"
        )
    return plan


def covers_entire_skeleton(
    shard: EnrichShard | AssignmentShard, skeleton: Skeleton
) -> bool:
    """True when this shard's scope is the whole repository.

    Such a shard gets `timeout_s`, not the (shorter) per-shard
    `enrich_timeout_s`: a per-shard budget is right for a *slice* of the repo
    and wrong for the whole of it, and "degenerates to today's single call"
    must not quietly mean "with a 3× shorter deadline".
    """
    return shard.subtree.all_paths() == skeleton.all_paths()


def has_several_source_roots(skeleton: Skeleton) -> bool:
    """True when the source root has >=2 top-level source-bearing folders,
    i.e. the multi-source-folder situation this change targets."""
    return len(skeleton.nodes) >= 2


# ── Shard ownership ───────────────────────────────────────────────────────


def owning_shard(path: str, shards: list[EnrichShard]) -> EnrichShard | None:
    """The shard whose scope owns `path`: the deepest shard root that is an
    ancestor of it. Used by the Stage-3 branch precheck."""
    best: EnrichShard | None = None
    for s in shards:
        if _is_ancestor(s.root_path, path) and (
            best is None or len(s.root_path) > len(best.root_path)
        ):
            best = s
    return best


# ── Per-shard (subtree-scoped) validation helpers ─────────────────────────


def validate_shard_scope(shard: EnrichShard, fragment: EnrichedTree) -> None:
    """The shard emitted/folded only inside its own scope, and exactly one
    top-level object rooted at `root_path`. Raises `CrossArtifactError` so the
    caller can spend a bounded shard repair on it."""
    if len(fragment.modules) != 1:
        raise CrossArtifactError(
            f"shard {shard.key!r} must emit exactly one top-level module rooted "
            f"at {shard.root_path!r}, got {len(fragment.modules)}"
        )
    top = fragment.modules[0]
    if top.path.strip("/") != shard.root_path:
        raise CrossArtifactError(
            f"shard {shard.key!r} top-level module is {top.path!r}, expected "
            f"{shard.root_path!r}"
        )
    for module in fragment.iter_all_modules():
        if not _is_ancestor(shard.root_path, module.path.strip("/")):
            raise CrossArtifactError(
                f"shard {shard.key!r} emitted {module.path!r}, which is outside "
                f"its scope {shard.root_path!r}"
            )
    for fold in fragment.folds:
        if not _is_ancestor(shard.root_path, fold.path.strip("/")):
            raise CrossArtifactError(
                f"shard {shard.key!r} folded {fold.path!r}, which is outside its "
                f"scope {shard.root_path!r}"
            )


def validate_spine_main_files(shard: EnrichShard, fragment: EnrichedTree) -> None:
    """No `main_file` may live under a promoted child root.

    Rule 3 checks the *nearest emitted* owner and does not require a main file
    to be in the skeleton inventory, so a spine can legally claim a file under a
    promoted child in its own pruned fragment — and then fail deterministically
    after the merge, where that child *is* emitted and no repair is possible.
    This shard-local predicate turns that unrecoverable Stage-5 rejection into a
    repairable shard error.
    """
    if not shard.promoted_children:
        return
    for module in fragment.iter_all_modules():
        for f in module.main_files:
            for child in shard.promoted_children:
                if _is_ancestor(child, f.path.strip("/")):
                    raise CrossArtifactError(
                        f"spine shard {shard.key!r} claims main_file {f.path!r}, "
                        f"which is owned by promoted child {child!r} (emitted by "
                        "another shard); choose a file outside that subtree"
                    )


def validate_promotion_parent(shard: EnrichShard, fragment: EnrichedTree) -> None:
    """A chain-split spine must *emit* the node whose children it promoted.

    `validate_shard_scope` already forces the root to be emitted, so this only
    bites when `_split_target` walked down a one-child chain: there the
    promotion parent is an ordinary module in the middle of the subtree, and the
    model is free to fold it into its parent. Folding it is unrecoverable after
    the merge — the promoted children re-attach to whatever emitted ancestor is
    left, and the fold itself then has to prove Rule-7 evidence for a directory
    whose entire content lives in subtrees another shard owns. Caught here, it
    is just another repairable shard error.
    """
    if not shard.promoted_children:
        return
    parent = shard.promotion_parent
    if parent == shard.root_path:
        return  # guaranteed by `validate_shard_scope`
    emitted = {m.path.strip("/") for m in fragment.iter_all_modules()}
    if parent in emitted:
        return
    folded = {f.path.strip("/") for f in fragment.folds}
    detail = "folded away" if parent in folded else "neither emitted nor folded"
    raise CrossArtifactError(
        f"spine shard {shard.key!r} must emit {parent!r} as a module ({detail}): "
        f"the subtrees removed from your SKELETON are re-attached as its "
        f"children, so it cannot be collapsed into its parent"
    )


# ── Merge ─────────────────────────────────────────────────────────────────


def _demote(module: EnrichedTopModule) -> EnrichedSubmodule:
    return EnrichedSubmodule(
        name=module.name,
        path=module.path,
        description=module.description,
        main_files=list(module.main_files),
        submodules=list(module.submodules),
    )


def _sort_submodules(module: EnrichedTopModule | EnrichedSubmodule) -> None:
    module.submodules.sort(key=lambda s: s.name)
    for sub in module.submodules:
        _sort_submodules(sub)


def assemble_branch(
    fragments: list[tuple[EnrichShard, EnrichedTree]],
    *,
    branch_root: str,
) -> EnrichedTopModule:
    """Re-assemble one branch's shard fragments into its single top-level module.

    Bottom-up by sub-shard `depth`: a depth-2 grandchild is grafted into its
    parent sub-shard's fragment *before* that parent is grafted upward, so it
    lands at its true object-tree position.

    Grafting is by physical-path ancestry, into the *nearest emitted module*
    above the sub-shard's root — which is the enclosing fragment's own top
    module when the promotion parent is the shard root, and a module nested
    inside that fragment when derivation walked down a one-child chain to split
    (`_split_target`). Attaching a chain-split child to the fragment root
    instead would put it at an object-tree position its physical path
    contradicts, which `validate_enriched_tree` rejects after the merge.
    """
    # Deep-copy every fragment module before grafting: this function is called
    # once per branch by the *dry-run* precheck and again by the authoritative
    # global merge, so mutating the caller's fragments in place would graft the
    # same children twice.
    by_root: dict[str, EnrichedTopModule] = {}
    emitted: dict[str, EnrichedTopModule | EnrichedSubmodule] = {}
    for shard, fragment in fragments:
        if not fragment.modules:  # pragma: no cover - validation catches this
            raise ExtractorValidationError(
                f"shard {shard.key!r} produced no top-level module", stage="enrich"
            )
        top = fragment.modules[0].model_copy(deep=True)
        by_root[shard.root_path] = top
        # Index the whole fragment, not just its root: a chain-split spine's
        # graft targets are nested inside it. Mutating one of those nested
        # objects stays visible after its own fragment is demoted upward,
        # because `_demote` copies the submodule *list*, not its elements.
        for module in _iter_modules(top):
            emitted[module.path.strip("/")] = module

    root_module = by_root.get(branch_root)
    if root_module is None:
        raise ExtractorValidationError(
            f"branch {branch_root!r} has no shard fragment for its own root",
            stage="enrich",
        )

    # Deepest *physical* path first. Shard `depth` alone is not enough: a spine
    # and the children it promotes share a depth (a depth-2 spine sits beside
    # its own depth-2 grandchildren), so ordering by shard depth would graft the
    # parent upward before its children had landed in it.
    for shard, _fragment in sorted(
        fragments,
        key=lambda sf: (-sf[0].root_path.count("/"), -sf[0].depth, sf[0].root_path),
    ):
        if shard.root_path == branch_root:
            continue
        target_path = _nearest_ancestor_root(shard.root_path, emitted)
        if target_path is None:
            raise ExtractorValidationError(
                f"sub-shard {shard.key!r} ({shard.root_path!r}) has no emitted "
                "ancestor to graft into; its promotion parent was neither "
                "emitted nor covered by an enclosing shard",
                stage="enrich",
            )
        emitted[target_path].submodules.append(
            _demote(by_root[shard.root_path])
        )

    _sort_submodules(root_module)
    return root_module


def _iter_modules(
    module: EnrichedTopModule | EnrichedSubmodule,
) -> Iterator[EnrichedTopModule | EnrichedSubmodule]:
    """`module` and every module nested under it, preorder."""
    yield module
    for sub in module.submodules:
        yield from _iter_modules(sub)


def _nearest_ancestor_root(
    path: str, roots: Mapping[str, EnrichedTopModule | EnrichedSubmodule]
) -> str | None:
    best: str | None = None
    for root in roots:
        if root != path and _is_ancestor(root, path):
            if best is None or len(root) > len(best):
                best = root
    return best


def branch_tree(
    fragments: list[tuple[EnrichShard, EnrichedTree]], *, branch_root: str
) -> EnrichedTree:
    """The branch-local merge used by the 3.5 precheck: one top-level module plus
    the branch's folds."""
    return EnrichedTree(
        modules=[assemble_branch(fragments, branch_root=branch_root)],
        folds=sorted(
            (f for _, frag in fragments for f in frag.folds), key=lambda f: f.path
        ),
    )


def merge_fragments(
    fragments: list[tuple[EnrichShard, EnrichedTree]],
    *,
    skeleton: Skeleton,
    plan: ShardPlan,
) -> EnrichedTree:
    """Deterministically merge every shard fragment into one `EnrichedTree`.

    Pure set/list concatenation with an ordering rule: no semantic inference, no
    added nodes. Given the same fragments it always produces byte-identical
    output, so wall-clock nondeterminism from the concurrent executor never
    reaches the artifact.

    The `root_path` sort here only stabilizes the *intermediate*
    `03_enrich/merged/enriched_tree.json`; the public artifact's order comes
    from the unchanged `ProjectTree` name-sorting validators, exactly as today.
    """
    _assert_disjoint(fragments, skeleton)

    grouped: dict[str, list[tuple[EnrichShard, EnrichedTree]]] = {}
    for shard, fragment in fragments:
        grouped.setdefault(plan.branch_key_of(shard), []).append((shard, fragment))

    modules: list[tuple[str, EnrichedTopModule]] = []
    for branch_key, branch_fragments in grouped.items():
        root = plan.branch_roots.get(branch_key, branch_key)
        modules.append((root, assemble_branch(branch_fragments, branch_root=root)))

    return EnrichedTree(
        modules=[m for _, m in sorted(modules, key=lambda rm: rm[0])],
        folds=sorted(
            (f for _, frag in fragments for f in frag.folds), key=lambda f: f.path
        ),
    )


def _assert_disjoint(
    fragments: list[tuple[EnrichShard, EnrichedTree]], skeleton: Skeleton
) -> None:
    """Emitted and fold path sets from distinct shards must not overlap.

    A violation is an internal error, not a model error: it would mean shard
    derivation broke its own partition invariant.
    """
    inventory = skeleton.all_paths()
    seen_emitted: dict[str, str] = {}
    seen_folded: dict[str, str] = {}
    for shard, fragment in fragments:
        for module in fragment.iter_all_modules():
            path = module.path.strip("/")
            if path in seen_emitted:
                raise ExtractorValidationError(
                    f"shards {seen_emitted[path]!r} and {shard.key!r} both emit "
                    f"{path!r}; the shard partition is broken",
                    stage="enrich",
                )
            seen_emitted[path] = shard.key
            if path not in inventory:
                raise ExtractorValidationError(
                    f"shard {shard.key!r} emitted {path!r}, which is not in the "
                    "skeleton inventory",
                    stage="enrich",
                )
        for fold in fragment.folds:
            path = fold.path.strip("/")
            if path in seen_folded:
                raise ExtractorValidationError(
                    f"shards {seen_folded[path]!r} and {shard.key!r} both fold "
                    f"{path!r}; the shard partition is broken",
                    stage="enrich",
                )
            seen_folded[path] = shard.key


# ── Human-readable sharding log ───────────────────────────────────────────


def _shard_kind(shard: EnrichShard) -> str:
    if shard.is_spine:
        return "spine"
    if shard.is_subshard:
        return "sub-shard"
    return "branch"


def _shard_notes(shard: EnrichShard, plan: ShardPlan) -> str:
    notes: list[str] = []
    if shard.promoted_children:
        promoted = ", ".join(f"`{p}`" for p in shard.promoted_children)
        notes.append(
            f"promoted {len(shard.promoted_children)} child(ren): {promoted}"
        )
        if shard.promotion_parent != shard.root_path:
            notes.append(f"split at `{shard.promotion_parent}`")
    reason = plan.not_split_reasons.get(shard.key)
    if reason is not None:
        label = "not split" if not shard.is_subshard else "not split further"
        notes.append(f"{label}: {reason}")
    return "; ".join(notes)


def render_shard_plan_markdown(plan: ShardPlan) -> str:
    """Render the derived shard plan as deterministic Markdown.

    The human companion to `03_enrich/shards.json`: one row per shard, grouped
    by top-level branch in derivation order (spine before the children it
    promoted), with each shard's size. Pure and timestamp-free, so identical
    plans produce byte-identical logs.
    """
    rows: list[tuple[str, EnrichShard]] = [
        (branch_key, shard)
        for branch_key in plan.branch_keys()
        for shard in plan.shards_of_branch(branch_key)
    ]

    out: list[str] = []
    out.append("# Stage-3 sharding")
    out.append("")
    out.append(
        f"{len(plan.shards)} shard(s) across {len(plan.branch_roots)} "
        "top-level branch(es). One row per shard, grouped by branch."
    )
    out.append("")
    out.append(
        "Sizes are the shard's own (possibly pruned) scope — a spine's row "
        "excludes the children it promoted to sub-shards:"
    )
    out.append("")
    out.append(
        "- **required** — required skeleton nodes the shard enriches "
        "(its deterministic weight, the number its deadline is sized against)"
    )
    out.append("- **nodes** — every directory node in the shard's subtree")
    out.append("- **files** — source files across the shard's subtree")
    out.append("")
    out.append(
        "| branch | shard | kind | root | depth | required | nodes | files "
        "| notes |"
    )
    out.append("|---|---|---|---|---:|---:|---:|---:|---|")
    for branch_key, shard in rows:
        inventory = len(shard.subtree.all_paths())
        out.append(
            f"| `{plan.branch_roots[branch_key]}` | `{shard.key}` "
            f"| {_shard_kind(shard)} | `{shard.root_path}` | {shard.depth} "
            f"| {shard_weight(shard)} | {inventory} "
            f"| {shard_source_file_count(shard)} "
            f"| {_shard_notes(shard, plan)} |"
        )
    out.append(
        f"| **total** | {len(plan.shards)} shard(s) | | | "
        f"| {sum(shard_weight(s) for s in plan.shards)} "
        f"| {sum(len(s.subtree.all_paths()) for s in plan.shards)} "
        f"| {sum(shard_source_file_count(s) for s in plan.shards)} | |"
    )
    out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# ── Per-branch coverage precheck ──────────────────────────────────────────


def branch_coverage(
    plan: ShardPlan,
    branch_key: str,
    fragments: list[tuple[EnrichShard, EnrichedTree]],
) -> CoverageReport:
    """Branch-local coverage over the *re-assembled* branch module vs. the
    branch's full required subset.

    Computed on the branch-local merge, not on individual fragments: a required
    node owned by a child sub-shard is `emitted` only in that child's fragment,
    and only the branch merge brings every shard's emitted/folded sets into one
    union. The skeleton passed is the **unpruned** branch slice — a spine's own
    subtree has the promoted children removed, so using it here would shrink
    `required` to the spine's share and the precheck would pass vacuously.
    """
    root = plan.branch_roots[branch_key]
    return compute_coverage(
        branch_tree(fragments, branch_root=root), plan.branch_subtrees[branch_key]
    )


# ══════════════════════════════════════════════════════════════════════════
# Assignment-contract sharding (v2)
# ══════════════════════════════════════════════════════════════════════════


def assignment_node_weight(node: SkeletonNode) -> int:
    """All-inventory-node weight: every path is output in assignment mode, so
    the deterministic size of a subtree is its full node count, not only its
    `required` nodes. (This changes the documented semantics of the shared
    `enrich_subshard_*` fields in assignment mode.)"""
    return 1 + sum(assignment_node_weight(c) for c in node.children)


class AssignmentShard(BaseModel):
    """One unit of Stage-3A assignment: a subtree slice the model labels.

    A shard labels exactly `subtree.all_paths()`, which makes per-shard
    coverage a local property and the merge a checked disjoint union. There is
    no `owns_root`, no Rule-4 exemption set, and no spine main-file guard: a
    nested shard's root may legally resolve `PART`, and main files do not
    exist in Stage 3A.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    root_path: str
    subtree: Skeleton
    is_subshard: bool = False
    parent_key: str | None = None
    """The enclosing top-level BRANCH key (never the enclosing sub-shard), or
    None for an un-split top-level shard."""

    is_branch_root: bool = True
    """True when `root_path` is a true top-level skeleton path — the one case
    where the shard must label its root `MODULE` (structural anchor)."""

    depth: int = 0
    promoted_children: list[str] = Field(default_factory=list)

    @property
    def is_spine(self) -> bool:
        return bool(self.promoted_children)

    @property
    def promotion_parent(self) -> str:
        """The node whose children were promoted away; `root_path` when
        nothing was promoted. Still meaningful because the packing-only
        `_assignment_split_target` may descend a one-promotable-child chain."""
        if not self.promoted_children:
            return self.root_path
        return _parent_path(self.promoted_children[0])


class AssignmentShardPlan(BaseModel):
    """The derived assignment-mode plan; same branch bookkeeping as v1."""

    model_config = ConfigDict(extra="forbid")

    shards: list[AssignmentShard] = Field(default_factory=list)
    branch_subtrees: dict[str, Skeleton] = Field(default_factory=dict)
    branch_roots: dict[str, str] = Field(default_factory=dict)
    not_split_reasons: dict[str, str] = Field(default_factory=dict)

    def branch_key_of(self, shard: AssignmentShard) -> str:
        return shard.parent_key or shard.key

    def shards_of_branch(self, branch_key: str) -> list[AssignmentShard]:
        return [s for s in self.shards if self.branch_key_of(s) == branch_key]

    def branch_keys(self) -> list[str]:
        return [k for k, _ in sorted(self.branch_roots.items(), key=lambda kv: kv[1])]

    def primary_shard(self) -> AssignmentShard | None:
        """The shard whose `root_path` equals the first branch's root (then by
        depth/key) — a bookkeeping choice for telemetry, not a semantic label."""
        keys = self.branch_keys()
        if not keys:
            return None
        owners = [
            s
            for s in self.shards_of_branch(keys[0])
            if s.root_path == self.branch_roots[keys[0]]
        ]
        if not owners:
            return None
        return min(owners, key=lambda s: (s.depth, s.key))


def assignment_shard_weight(shard: AssignmentShard) -> int:
    """The all-node weight of the shard's own (possibly pruned) subtree."""
    return sum(assignment_node_weight(n) for n in shard.subtree.nodes)


def _assignment_promotable(
    node: SkeletonNode, child_min: int
) -> list[SkeletonNode]:
    return [c for c in node.children if assignment_node_weight(c) >= child_min]


def _assignment_promotion_order(children: list[SkeletonNode]) -> list[SkeletonNode]:
    return sorted(
        children,
        key=lambda c: (
            -assignment_node_weight(c),
            -node_source_file_count(c),
            c.path,
        ),
    )


def _assignment_split_target(
    node: SkeletonNode, config: ExtractorConfig
) -> SkeletonNode | None:
    """The descendant whose children this split promotes, or None when the
    subtree has no promotable child at all.

    A packing optimization, not a correctness gate: with Rule 4 retired, a
    spine holding exactly one promoted child is legal. Walk down a
    single-promotable-child chain and prefer the first descendant with at
    least two promotable children (avoids spending the depth budget on
    wrapper-only spines); otherwise return the deepest node that still has
    one promotable child.
    """
    current = node
    last_single: SkeletonNode | None = None
    while True:
        promotable = _assignment_promotable(
            current, config.enrich_subshard_child_min
        )
        if len(promotable) >= 2:
            return current
        if len(promotable) == 1:
            last_single = current
            current = promotable[0]
            continue
        return last_single


def _assignment_refuse_split_reason(
    node: SkeletonNode,
    *,
    depth: int,
    config: ExtractorConfig,
    budget: _Budget,
) -> str | None:
    """Why `node` must not be split in assignment mode, or None.

    Unlike the v1 gate, a split requires only **one** remaining budget slot
    (a spine plus one promoted child is a legal outcome), and having a single
    promotable child is not a refusal."""
    if depth + 1 > config.enrich_subshard_max_depth:
        return NOT_SPLIT_MAX_DEPTH
    if assignment_node_weight(node) <= config.enrich_subshard_threshold:
        return NOT_SPLIT_BELOW_THRESHOLD
    if _assignment_split_target(node, config) is None:
        return NOT_SPLIT_NO_PROMOTABLE
    if budget.remaining < 1:
        return NOT_SPLIT_BUDGET
    return None


def _plan_assignment_node(
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
    if allow_split:
        reason = _assignment_refuse_split_reason(
            node, depth=depth, config=config, budget=budget
        )
    else:
        reason = NOT_SPLIT_TOP_LEVEL_ONLY

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
                promoted_children=[],
            )
        ]

    target = _assignment_split_target(node, config)
    if target is None:  # pragma: no cover - defensive; the gate cleared it
        raise ExtractorValidationError(
            f"no split target for {node.path!r} after the split gate passed",
            stage="enrich",
        )
    promotable = _assignment_promotable(target, config.enrich_subshard_child_min)
    take = min(len(promotable), budget.remaining)
    promoted = sorted(
        _assignment_promotion_order(promotable)[:take], key=lambda c: c.path
    )
    budget.remaining -= len(promoted)

    promoted_roots = frozenset(c.path for c in promoted)
    spine_key = _mint_key_literal(f"{key}{_SPINE_SUFFIX}", used_keys)
    shards = [
        AssignmentShard(
            key=spine_key,
            root_path=node.path,
            subtree=slice_skeleton(skeleton, node, pruned_roots=promoted_roots),
            is_subshard=True,
            parent_key=branch_key,
            is_branch_root=node.path == branch_root,
            depth=depth + 1,
            promoted_children=sorted(promoted_roots),
        )
    ]
    for child in promoted:
        shards.extend(
            _plan_assignment_node(
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
    """Partition Stage-3A assignment into shards, deterministically.

    Same top-level partition and bounded recursive sub-sharding as the v1
    planner, with the assignment-mode differences: all-node weights, a
    one-slot split cost, and one-promotable-child splits. Stage 3A always
    needs one shard per top-level branch, so `enrich_max_shards` budgets only
    the *extra* recursive shards (effective ceiling
    `max(enrich_max_shards, top_level_count)`).
    """
    top_nodes = sorted(skeleton.nodes, key=lambda n: n.path)
    used_keys: set[str] = set()
    branch_keys: list[tuple[SkeletonNode, str]] = [
        (n, _mint_key(n.path, used_keys)) for n in top_nodes
    ]

    plan = AssignmentShardPlan(
        branch_subtrees={k: slice_skeleton(skeleton, n) for n, k in branch_keys},
        branch_roots={k: n.path for n, k in branch_keys},
    )

    budget = _Budget(config.enrich_max_shards - len(branch_keys))
    allow_split = config.enrich_sharding == "auto"

    for node, key in branch_keys:
        plan.shards.extend(
            _plan_assignment_node(
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

    keys = [s.key for s in plan.shards]
    if len(set(keys)) != len(keys):  # pragma: no cover - defensive
        raise ExtractorValidationError(
            f"shard keys are not unique: {sorted(keys)}", stage="enrich"
        )
    return plan


def validate_assignment_shard_scope(
    shard: AssignmentShard, fragment: AssignmentTree
) -> None:
    """The fragment labeled only keys inside the shard's slice inventory.

    Checks assignment **and** module-decision keys, not merely path prefixes:
    a promoted subtree's paths are absent from the spine's allowed inventory
    and may not appear even though they sit under `root_path`. Raises
    `CrossArtifactError` so the caller can spend a bounded shard repair.
    """
    allowed = shard.subtree.all_paths()
    for kind, keys in (
        ("assignment", fragment.assignments.keys()),
        ("module_decisions", fragment.module_decisions.keys()),
    ):
        for path in sorted(keys):
            if path in allowed:
                continue
            if any(_is_ancestor(p, path) for p in shard.promoted_children):
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
    """Checked disjoint union of fragment label/decision maps.

    Deterministic: fragments are processed in (root_path, key) order and the
    result maps are key-sorted, so wall-clock nondeterminism from the
    concurrent executor never reaches the artifact. Any overlap or cross-set
    inconsistency is an internal partition error, not a model error.
    """
    assignments: dict[str, AssignmentLabel] = {}
    decisions: dict[str, ModuleDecision] = {}
    assignment_owner: dict[str, str] = {}
    for shard, fragment in sorted(
        fragments, key=lambda sf: (sf[0].root_path, sf[0].key)
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
        assignments={k: assignments[k] for k in sorted(assignments)},
        module_decisions={k: decisions[k] for k in sorted(decisions)},
    )


def assignment_branch_coverage(
    plan: AssignmentShardPlan,
    branch_key: str,
    fragments: list[tuple[AssignmentShard, AssignmentTree]],
) -> AssignmentCoverageReport:
    """Branch-local totality precheck: the union of the branch's fragment
    labels vs. the branch's **unpruned** subtree inventory. The cheap
    losslessness check and missing-path blame localizer; its firing indicates
    a sharding bug, not a bad model response."""
    union = merge_assignment_fragments(fragments)
    return compute_assignment_coverage(union, plan.branch_subtrees[branch_key])


def assignment_owning_shard(
    path: str, shards: list[AssignmentShard]
) -> AssignmentShard | None:
    """The shard whose slice inventory contains `path` (unique by partition)."""
    for s in shards:
        if path in s.subtree.all_paths():
            return s
    return None


def render_assignment_shard_plan_markdown(plan: AssignmentShardPlan) -> str:
    """Deterministic Markdown companion to the assignment `shards.json`."""
    rows: list[tuple[str, AssignmentShard]] = [
        (branch_key, shard)
        for branch_key in plan.branch_keys()
        for shard in plan.shards_of_branch(branch_key)
    ]

    out: list[str] = []
    out.append("# Stage-3A assignment sharding")
    out.append("")
    out.append(
        f"{len(plan.shards)} shard(s) across {len(plan.branch_roots)} "
        "top-level branch(es). One row per shard, grouped by branch."
    )
    out.append("")
    out.append(
        "Sizes are the shard's own (possibly pruned) scope — a spine's row "
        "excludes the children it promoted to sub-shards:"
    )
    out.append("")
    out.append(
        "- **nodes** — every inventory node the shard labels (its "
        "deterministic weight; assignment mode outputs all paths)"
    )
    out.append("- **files** — source files across the shard's subtree")
    out.append("")
    out.append("| branch | shard | kind | root | anchor | depth | nodes | files | notes |")
    out.append("|---|---|---|---|---|---:|---:|---:|---|")
    for branch_key, shard in rows:
        kind = (
            "spine"
            if shard.is_spine
            else ("sub-shard" if shard.is_subshard else "branch")
        )
        notes: list[str] = []
        if shard.promoted_children:
            promoted = ", ".join(f"`{p}`" for p in shard.promoted_children)
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
            f"| `{plan.branch_roots[branch_key]}` | `{shard.key}` "
            f"| {kind} | `{shard.root_path}` "
            f"| {'yes' if shard.is_branch_root else 'no'} | {shard.depth} "
            f"| {assignment_shard_weight(shard)} "
            f"| {shard_source_file_count(shard)} "
            f"| {'; '.join(notes)} |"
        )
    out.append(
        f"| **total** | {len(plan.shards)} shard(s) | | | | "
        f"| {sum(assignment_shard_weight(s) for s in plan.shards)} "
        f"| {sum(shard_source_file_count(s) for s in plan.shards)} | |"
    )
    out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


__all__ = [
    "NOT_SPLIT_BELOW_THRESHOLD",
    "NOT_SPLIT_BUDGET",
    "NOT_SPLIT_MAX_DEPTH",
    "NOT_SPLIT_NO_PROMOTABLE",
    "NOT_SPLIT_ONE_PROMOTABLE",
    "NOT_SPLIT_TOP_LEVEL_ONLY",
    "AssignmentShard",
    "AssignmentShardPlan",
    "EnrichShard",
    "ShardPlan",
    "assemble_branch",
    "assignment_branch_coverage",
    "assignment_node_weight",
    "assignment_owning_shard",
    "assignment_shard_weight",
    "branch_coverage",
    "branch_tree",
    "covers_entire_skeleton",
    "derive_assignment_shards",
    "derive_enrich_shards",
    "has_several_source_roots",
    "merge_assignment_fragments",
    "merge_fragments",
    "node_source_file_count",
    "node_weight",
    "owning_shard",
    "render_assignment_shard_plan_markdown",
    "render_shard_plan_markdown",
    "shard_source_file_count",
    "shard_weight",
    "slice_skeleton",
    "validate_assignment_shard_scope",
    "validate_promotion_parent",
    "validate_shard_scope",
    "validate_spine_main_files",
]
