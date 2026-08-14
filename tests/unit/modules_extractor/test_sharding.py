"""Shard derivation, merge determinism, and the shard-local validators.

Everything here is pure Python — no Claude, no Codex. The few tests that need a
filesystem build a tiny synthetic repo, because `validate_enriched_tree` checks
paths against the real filesystem.

The fixtures deliberately encode the two traps that splitting a branch creates
(single-child spine, spine `main_file` under a promoted child), since those are
the failure modes that the pure Python merge and the repair-less Stage 5 cannot
recover from. A third — a spine root with no file of its own to cite — is not a
trap any more: such a root cites nothing.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.coverage import (
    CrossArtifactError,
    compute_coverage,
    validate_enriched_tree,
)
from spotlights_engine.modules_extractor.errors import ExtractorValidationError
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.sharding import (
    NOT_SPLIT_BELOW_THRESHOLD,
    NOT_SPLIT_ONE_PROMOTABLE,
    NOT_SPLIT_TOP_LEVEL_ONLY,
    EnrichShard,
    ShardPlan,
    branch_coverage,
    covers_entire_skeleton,
    derive_enrich_shards,
    has_several_source_roots,
    merge_fragments,
    node_weight,
    owning_shard,
    render_shard_plan_markdown,
    shard_source_file_count,
    shard_weight,
    validate_promotion_parent,
    validate_shard_scope,
    validate_spine_main_files,
)
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    Skeleton,
    SkeletonNode,
)
from spotlights_engine.schemas.project import Repository

# ── Synthetic skeleton builders (no filesystem) ───────────────────────────


def _node(
    path: str,
    *,
    required: bool = True,
    files: int = 2,
    children: tuple[SkeletonNode, ...] = (),
    rep: list[str] | None = None,
) -> SkeletonNode:
    return SkeletonNode(
        path=path,
        direct_source_file_count=files,
        source_child_count=len(children),
        representative_files=(
            rep if rep is not None else [f"{path}/main.py"]
        ),
        required=required,
        required_reasons=["two_or_more_direct_source_files"] if required else [],
        children=list(children),
    )


def _skel(*nodes: SkeletonNode, source_root: str = "") -> Skeleton:
    return Skeleton(
        source_root=source_root,
        nodes=list(nodes),
        ignored=["node_modules", "vendor"],
        excluded=["docs"],
        organizational_only=[],
        skipped_symlinks=[],
        inventory_fingerprint="fp-v1",
    )


def _cfg(**kwargs) -> ExtractorConfig:
    return ExtractorConfig(**kwargs)


# ── Derivation: the top-level partition ───────────────────────────────────


def test_two_branches_yield_two_shards_sorted_by_root_path() -> None:
    plan = derive_enrich_shards(_skel(_node("zeta"), _node("alpha")), _cfg())

    assert [s.key for s in plan.shards] == ["alpha", "zeta"]
    assert [s.root_path for s in plan.shards] == ["alpha", "zeta"]
    assert all(s.depth == 0 and s.owns_root and not s.is_subshard for s in plan.shards)
    assert all(s.parent_key is None for s in plan.shards)


def test_single_top_level_branch_degenerates_to_one_shard() -> None:
    skeleton = _skel(_node("pkg"))
    plan = derive_enrich_shards(skeleton, _cfg())

    assert len(plan.shards) == 1
    shard = plan.shards[0]
    assert shard.key == "pkg"
    # Its scope is the whole repository, so it must keep today's deadline.
    assert covers_entire_skeleton(shard, skeleton)


def test_shard_keys_are_artifact_safe_and_unique() -> None:
    plan = derive_enrich_shards(
        _skel(_node("src/my-pkg"), _node("src/my.pkg")), _cfg()
    )
    keys = [s.key for s in plan.shards]
    assert keys == ["src__mypkg", "src__mypkg_2"]
    assert len(set(keys)) == len(keys)


# ── The several-source-roots predicate ────────────────────────────────────


def test_has_several_source_roots_keys_on_node_count() -> None:
    """0/1 top-level nodes → single-source-root; >=2 → the multi-folder case."""
    assert has_several_source_roots(_skel()) is False
    assert has_several_source_roots(_skel(_node("pkg"))) is False
    assert has_several_source_roots(_skel(_node("alpha"), _node("beta"))) is True
    assert (
        has_several_source_roots(
            _skel(_node("alpha"), _node("beta"), _node("gamma"))
        )
        is True
    )


# ── Derivation: the size-gated split and its three gates ──────────────────


def _heavy_branch(n_children: int = 4, *, rep: list[str] | None = None) -> SkeletonNode:
    """A branch whose weight is spread across `n_children` promotable children."""
    children = tuple(
        _node(
            f"pkg/c{i}",
            children=tuple(_node(f"pkg/c{i}/g{j}") for j in range(3)),
        )
        for i in range(n_children)
    )
    return _node("pkg", children=children, rep=rep)


def test_heavy_branch_splits_into_spine_plus_child_subshards() -> None:
    skeleton = _skel(_heavy_branch())
    plan = derive_enrich_shards(
        skeleton, _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2)
    )

    keys = sorted(s.key for s in plan.shards)
    assert keys == ["pkg__c0", "pkg__c1", "pkg__c2", "pkg__c3", "pkg__spine"]

    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.owns_root and spine.is_subshard and spine.depth == 1
    assert spine.promoted_children == [f"pkg/c{i}" for i in range(4)]
    assert spine.parent_key == "pkg"

    child = next(s for s in plan.shards if s.key == "pkg__c0")
    assert not child.owns_root and child.is_subshard and child.depth == 1
    assert child.parent_key == "pkg"

    # The sub-shards partition the branch's required set: every required node is
    # owned by exactly one shard.
    branch_required = skeleton.required_paths()
    owned: dict[str, list[str]] = {}
    for path in branch_required:
        owner = owning_shard(path, plan.shards)
        assert owner is not None
        owned.setdefault(owner.key, []).append(path)
    assert sum(len(v) for v in owned.values()) == len(branch_required)
    union = {p for s in plan.shards for p in s.subtree.required_paths()}
    assert union == branch_required


def test_light_children_stay_with_the_spine() -> None:
    branch = _node(
        "pkg",
        children=(
            _node("pkg/heavy1", children=(_node("pkg/heavy1/x"),)),
            _node("pkg/heavy2", children=(_node("pkg/heavy2/x"),)),
            _node("pkg/light", required=False, files=1),
        ),
    )
    plan = derive_enrich_shards(
        _skel(branch), _cfg(enrich_subshard_threshold=3, enrich_subshard_child_min=2)
    )
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.promoted_children == ["pkg/heavy1", "pkg/heavy2"]
    assert "pkg/light" in spine.subtree.all_paths()


def test_shard_weight_is_the_residual_after_pruning() -> None:
    """A spine's weight is what it actually enriches, not the branch total."""
    plan = derive_enrich_shards(
        _skel(_heavy_branch()),
        _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2),
    )
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert shard_weight(spine) == 1  # `pkg` alone; all four children promoted
    child = next(s for s in plan.shards if s.key == "pkg__c0")
    assert shard_weight(child) == 4


def test_budget_starved_spine_keeps_its_leftover_weight() -> None:
    """With the shard budget able to take only two of four promotable children,
    the other two stay in the spine. `shard_weight` reports that residual —
    over the threshold — which is what entitles the spine to the full
    (monolithic) deadline instead of the short per-shard one."""
    cfg = _cfg(
        enrich_subshard_threshold=4, enrich_subshard_child_min=2, enrich_max_shards=3
    )
    plan = derive_enrich_shards(_skel(_heavy_branch()), cfg)

    assert len(plan.shards) == 3
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.promoted_children == ["pkg/c0", "pkg/c1"]
    assert shard_weight(spine) == 9  # `pkg` + the two unpromoted children
    assert shard_weight(spine) > cfg.enrich_subshard_threshold


def test_sharding_markdown_lists_every_shard_with_its_sizes() -> None:
    """`03_enrich/sharding.md`: one row per shard, each with its three sizes."""
    skeleton = _skel(_heavy_branch(), _node("tools"))
    cfg = _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2)
    plan = derive_enrich_shards(skeleton, cfg)

    text = render_shard_plan_markdown(plan)
    # Deterministic: an identical plan renders byte-identical bytes.
    assert text == render_shard_plan_markdown(derive_enrich_shards(skeleton, cfg))

    rows = {
        cells[1]: cells
        for line in text.splitlines()
        if line.startswith("| `")
        for cells in [[c.strip() for c in line.strip("|").split("|")]]
    }
    assert set(rows) == {f"`{s.key}`" for s in plan.shards}
    for shard in plan.shards:
        cells = rows[f"`{shard.key}`"]
        assert cells[0] == f"`{plan.branch_roots[plan.branch_key_of(shard)]}`"
        assert cells[3] == f"`{shard.root_path}`"
        assert cells[4] == str(shard.depth)
        assert cells[5] == str(shard_weight(shard))
        assert cells[6] == str(len(shard.subtree.all_paths()))
        assert cells[7] == str(shard_source_file_count(shard))

    # The spine names what it promoted; an un-split branch names why.
    assert "promoted 4 child(ren)" in " ".join(rows["`pkg__spine`"])
    assert NOT_SPLIT_BELOW_THRESHOLD in " ".join(rows["`tools`"])


def test_wide_flat_branch_stays_whole_and_over_threshold() -> None:
    """A branch over the threshold whose children are all below `child_min` has
    no legal split point: it stays monolithic at its full weight, and the
    deadline selection must compensate with the full `timeout_s`."""
    branch = _node("pkg", children=tuple(_node(f"pkg/c{i}") for i in range(10)))
    cfg = _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2)
    plan = derive_enrich_shards(_skel(branch), cfg)

    assert [s.key for s in plan.shards] == ["pkg"]
    assert plan.not_split_reasons["pkg"] == NOT_SPLIT_ONE_PROMOTABLE
    assert shard_weight(plan.shards[0]) == 11
    assert shard_weight(plan.shards[0]) > cfg.enrich_subshard_threshold


def test_branch_with_one_promotable_child_is_not_split() -> None:
    """The Rule-4 split gate.

    Splitting *at `pkg`* would graft a single child under the spine, and the
    merged tree would carry a single-child parent that Stage 5 rejects with no
    possible LLM recovery on the pure-Python merge. Descending into `pkg/heavy`
    is no help either: its own children are all below `child_min`, so there is
    no legal split point anywhere down the chain.
    """
    branch = _node(
        "pkg",
        children=(
            _node("pkg/heavy", children=tuple(_node(f"pkg/heavy/g{j}") for j in range(4))),
            _node("pkg/light", required=False, files=1),
        ),
    )
    plan = derive_enrich_shards(
        _skel(branch), _cfg(enrich_subshard_threshold=2, enrich_subshard_child_min=2)
    )

    assert [s.key for s in plan.shards] == ["pkg"]
    assert plan.not_split_reasons["pkg"] == NOT_SPLIT_ONE_PROMOTABLE


def _chain_branch() -> SkeletonNode:
    """vLLM's `rust/`: one light sibling, one heavy child, the split point a
    level further down. Refusing to split this is what blew the 1800s per-shard
    deadline on a 71-node branch."""
    return _node(
        "rust",
        files=0,
        rep=[],
        children=(
            _node("rust/proto"),
            _node(
                "rust/src",
                files=0,
                rep=[],
                children=tuple(
                    _node(f"rust/src/c{i}", children=(_node(f"rust/src/c{i}/deep"),))
                    for i in range(3)
                ),
            ),
        ),
    )


def test_one_promotable_child_descends_to_the_splittable_node() -> None:
    """The split point may be a descendant, not just the branch root."""
    plan = derive_enrich_shards(
        _skel(_chain_branch()),
        _cfg(enrich_subshard_threshold=2, enrich_subshard_child_min=2),
    )

    assert sorted(s.key for s in plan.shards) == [
        "rust__spine",
        "rust__src__c0",
        "rust__src__c1",
        "rust__src__c2",
    ]
    spine = next(s for s in plan.shards if s.key == "rust__spine")
    assert spine.root_path == "rust"
    assert spine.promoted_children == [f"rust/src/c{i}" for i in range(3)]
    assert spine.promotion_parent == "rust/src"
    # The chain rides along with the spine; only the promoted grandchildren go.
    assert spine.subtree.all_paths() == {"rust", "rust/proto", "rust/src"}
    # Every promoted subtree is a child sub-shard of the same branch.
    for i in range(3):
        child = next(s for s in plan.shards if s.key == f"rust__src__c{i}")
        assert child.parent_key == "rust" and not child.owns_root


def test_only_the_promotion_parent_defers_rule_4() -> None:
    """The chain node above the split point keeps every child it has, so its
    Rule 4 is decided inside the shard — where the bounded repair can still fix
    it. Exempting it too would let a folded `rust/proto` through, and the merged
    tree would then carry `rust` with `rust/src` as its only child, which Stage
    5 rejects with no repair left."""
    plan = derive_enrich_shards(
        _skel(_chain_branch()),
        _cfg(enrich_subshard_threshold=2, enrich_subshard_child_min=2),
    )
    spine = next(s for s in plan.shards if s.key == "rust__spine")
    assert spine.rule4_exempt_paths == {"rust/src"}
    assert "rust" not in spine.rule4_exempt_paths


def test_a_single_child_chain_node_refuses_the_split() -> None:
    """The one descent that must not happen.

    `pkg` holds nothing but the chain, so splitting below it would leave `pkg`
    with exactly one emitted child after the merge — the promoted subtrees
    re-attach under `pkg/src`, not under `pkg` — and folding `pkg/src` away is
    denied by `validate_promotion_parent`. Unsplittable beats deadlocked.
    """
    branch = _node(
        "pkg",
        files=0,
        rep=[],
        children=(
            _node(
                "pkg/src",
                files=0,
                rep=[],
                children=tuple(
                    _node(f"pkg/src/c{i}", children=(_node(f"pkg/src/c{i}/deep"),))
                    for i in range(3)
                ),
            ),
        ),
    )
    plan = derive_enrich_shards(
        _skel(branch), _cfg(enrich_subshard_threshold=2, enrich_subshard_child_min=2)
    )
    assert [s.key for s in plan.shards] == ["pkg"]
    assert plan.not_split_reasons["pkg"] == NOT_SPLIT_ONE_PROMOTABLE


def test_namespace_only_branch_root_is_split() -> None:
    """A namespace-only root (a Go `pkg/`) is splittable like any other.

    It was refused while a spine had to cite ≥1 `main_file` it owned; Rule 3 now
    lets a pure container cite none, so the largest branch in a typical Go repo
    is no longer pinned to a single monolithic enrichment call.
    """
    plan = derive_enrich_shards(
        _skel(_heavy_branch(rep=[])),
        _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2),
    )
    assert "pkg__spine" in {s.key for s in plan.shards}
    assert "pkg" not in plan.not_split_reasons


def test_init_only_root_is_split() -> None:
    """`direct_source_file_count == 0` but `representative_files != []`.

    A namespace package carrying only `__init__.py` splits — as it did before
    the no-direct-source-file gate was removed.
    """
    branch = _heavy_branch()
    branch = branch.model_copy(
        update={"direct_source_file_count": 0, "representative_files": ["pkg/__init__.py"]}
    )
    plan = derive_enrich_shards(
        _skel(branch), _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2)
    )
    assert "pkg__spine" in {s.key for s in plan.shards}


def test_branch_below_threshold_is_not_split() -> None:
    plan = derive_enrich_shards(
        _skel(_heavy_branch()), _cfg(enrich_subshard_threshold=1000)
    )
    assert [s.key for s in plan.shards] == ["pkg"]
    assert plan.not_split_reasons["pkg"] == NOT_SPLIT_BELOW_THRESHOLD


def test_top_level_only_mode_never_subshards() -> None:
    plan = derive_enrich_shards(
        _skel(_heavy_branch()),
        _cfg(enrich_sharding="top_level_only", enrich_subshard_threshold=1),
    )
    assert [s.key for s in plan.shards] == ["pkg"]
    assert plan.not_split_reasons["pkg"] == NOT_SPLIT_TOP_LEVEL_ONLY


# ── Derivation: the shard-count cap ───────────────────────────────────────


def _weighted_branch() -> SkeletonNode:
    """Four promotable children with strictly decreasing weight (4, 3, 2, 1)."""
    children = tuple(
        _node(
            f"pkg/c{i}",
            children=tuple(_node(f"pkg/c{i}/g{j}") for j in range(4 - i - 1)),
        )
        for i in range(4)
    )
    return _node("pkg", children=children)


def test_shard_cap_promotes_only_the_heaviest_children() -> None:
    branch = _weighted_branch()
    assert [node_weight(c) for c in branch.children] == [4, 3, 2, 1]

    # 1 branch + 2 promoted = 3 shards total.
    plan = derive_enrich_shards(
        _skel(branch),
        _cfg(
            enrich_subshard_threshold=4,
            enrich_subshard_child_min=1,
            enrich_max_shards=3,
            enrich_subshard_max_depth=1,
        ),
    )
    assert len(plan.shards) == 3
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.promoted_children == ["pkg/c0", "pkg/c1"]
    # The un-promoted children stayed with the spine.
    assert "pkg/c2" in spine.subtree.all_paths()
    assert "pkg/c3" in spine.subtree.all_paths()


def test_shard_cap_is_stable_across_runs() -> None:
    cfg = _cfg(
        enrich_subshard_threshold=4,
        enrich_subshard_child_min=1,
        enrich_max_shards=3,
        enrich_subshard_max_depth=1,
    )
    first = derive_enrich_shards(_skel(_weighted_branch()), cfg)
    second = derive_enrich_shards(_skel(_weighted_branch()), cfg)
    assert [s.key for s in first.shards] == [s.key for s in second.shards]


def test_depth_two_recursion_draws_from_the_remaining_budget() -> None:
    """A grandchild split must not push the total past `enrich_max_shards`."""
    deep = _node(
        "pkg",
        children=(
            _node(
                "pkg/big",
                children=tuple(
                    _node(f"pkg/big/g{j}", children=(_node(f"pkg/big/g{j}/x"),))
                    for j in range(4)
                ),
            ),
            _node("pkg/other", children=(_node("pkg/other/y"),)),
        ),
    )
    cfg = _cfg(
        enrich_subshard_threshold=2,
        enrich_subshard_child_min=2,
        enrich_subshard_max_depth=2,
        enrich_max_shards=5,
    )
    plan = derive_enrich_shards(_skel(deep), cfg)
    assert len(plan.shards) <= cfg.enrich_max_shards
    assert max(s.depth for s in plan.shards) == 2


def test_a_depth_two_shard_is_never_split_again() -> None:
    deep = _node(
        "pkg",
        children=(
            _node(
                "pkg/big",
                children=tuple(
                    _node(
                        f"pkg/big/g{j}",
                        children=tuple(_node(f"pkg/big/g{j}/h{k}") for k in range(3)),
                    )
                    for j in range(2)
                ),
            ),
            _node("pkg/other", children=(_node("pkg/other/y"),)),
        ),
    )
    plan = derive_enrich_shards(
        _skel(deep),
        _cfg(
            enrich_subshard_threshold=1,
            enrich_subshard_child_min=1,
            enrich_subshard_max_depth=2,
            enrich_max_shards=64,
        ),
    )
    assert max(s.depth for s in plan.shards) == 2


# ── Derivation: subtree slicing ───────────────────────────────────────────


def test_spine_subtree_prunes_promoted_children_and_copies_globals_verbatim() -> None:
    skeleton = _skel(_heavy_branch())
    plan = derive_enrich_shards(
        skeleton, _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2)
    )
    spine = next(s for s in plan.shards if s.key == "pkg__spine")

    assert spine.subtree.all_paths() == {"pkg"}
    for i in range(4):
        child = next(s for s in plan.shards if s.key == f"pkg__c{i}")
        assert f"pkg/c{i}" in child.subtree.all_paths()
        assert f"pkg/c{i}" not in spine.subtree.all_paths()

    # `ignored` is a global list of directory *names*, not branch-relative
    # paths, so slicing must not empty it.
    assert spine.subtree.ignored == skeleton.ignored
    assert spine.subtree.source_root == skeleton.source_root
    assert spine.subtree.inventory_fingerprint == skeleton.inventory_fingerprint

    # The spine's root keeps `source_child_count` verbatim even though the
    # promoted children were pruned — rewriting it without the reasons would
    # produce an incoherent node.
    root = spine.subtree.nodes[0]
    assert root.source_child_count == 4
    assert root.children == []


def test_branch_subtrees_are_unpruned() -> None:
    plan = derive_enrich_shards(
        _skel(_heavy_branch()),
        _cfg(enrich_subshard_threshold=4, enrich_subshard_child_min=2),
    )
    branch = plan.branch_subtrees["pkg"]
    assert "pkg/c0" in branch.all_paths()
    assert branch.required_paths() == {
        "pkg",
        *(f"pkg/c{i}" for i in range(4)),
        *(f"pkg/c{i}/g{j}" for i in range(4) for j in range(3)),
    }


# ── Merge ─────────────────────────────────────────────────────────────────


def _mod(path: str, *, subs=(), files=None) -> dict:
    return {
        "name": path.rsplit("/", 1)[-1],
        "path": path,
        "description": f"Module {path}.",
        # `files=[]` is meaningful (a pure container cites nothing), so only a
        # missing argument takes the default.
        "main_files": (
            files if files is not None
            else [{"path": f"{path}/main.py", "role": "Entry."}]
        ),
        "submodules": list(subs),
    }


def _sub(path: str, *, subs=(), files=None) -> dict:
    return _mod(path, subs=subs, files=files)


def _frag(*modules: dict, folds=()) -> EnrichedTree:
    return EnrichedTree.model_validate({"modules": list(modules), "folds": list(folds)})


def _plan_for(shards: list[EnrichShard], branch_roots: dict[str, str]) -> ShardPlan:
    return ShardPlan(shards=shards, branch_roots=branch_roots)


def _shard(key: str, root: str, **kwargs) -> EnrichShard:
    return EnrichShard(
        key=key,
        root_path=root,
        subtree=_skel(_node(root)),
        **kwargs,
    )


def test_merge_is_order_independent() -> None:
    a = _shard("alpha", "alpha")
    b = _shard("beta", "beta")
    plan = _plan_for([a, b], {"alpha": "alpha", "beta": "beta"})
    pairs = [
        (a, _frag(_mod("alpha"))),
        (b, _frag(_mod("beta"))),
    ]
    skeleton = _skel(_node("alpha"), _node("beta"))
    expected = merge_fragments(pairs, skeleton=skeleton, plan=plan).model_dump_json()

    rng = random.Random(7)
    for _ in range(5):
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        got = merge_fragments(shuffled, skeleton=skeleton, plan=plan)
        assert got.model_dump_json() == expected
    assert [m.path for m in merge_fragments(pairs, skeleton=skeleton, plan=plan).modules] == [
        "alpha",
        "beta",
    ]


def test_merge_reassembles_subshards_and_demotes_them() -> None:
    spine = _shard(
        "pkg__spine",
        "pkg",
        is_subshard=True,
        parent_key="pkg",
        depth=1,
        promoted_children=["pkg/a", "pkg/b"],
    )
    a = _shard("pkg__a", "pkg/a", is_subshard=True, parent_key="pkg", owns_root=False, depth=1)
    b = _shard("pkg__b", "pkg/b", is_subshard=True, parent_key="pkg", owns_root=False, depth=1)
    plan = _plan_for([spine, a, b], {"pkg": "pkg"})
    skeleton = _skel(_node("pkg", children=(_node("pkg/a"), _node("pkg/b"))))

    merged = merge_fragments(
        [
            (spine, _frag(_mod("pkg"))),
            (a, _frag(_mod("pkg/a"))),
            (b, _frag(_mod("pkg/b"))),
        ],
        skeleton=skeleton,
        plan=plan,
    )

    assert len(merged.modules) == 1
    top = merged.modules[0]
    assert top.path == "pkg"
    assert [s.path for s in top.submodules] == ["pkg/a", "pkg/b"]


def test_merge_grafts_a_depth_two_grandchild_bottom_up() -> None:
    spine = _shard(
        "pkg__spine", "pkg", is_subshard=True, parent_key="pkg", depth=1,
        promoted_children=["pkg/a", "pkg/b"],
    )
    a_spine = _shard(
        "pkg__a__spine", "pkg/a", is_subshard=True, parent_key="pkg", depth=2,
        promoted_children=["pkg/a/g1", "pkg/a/g2"],
    )
    g1 = _shard(
        "pkg__a__g1", "pkg/a/g1", is_subshard=True, parent_key="pkg",
        owns_root=False, depth=2,
    )
    g2 = _shard(
        "pkg__a__g2", "pkg/a/g2", is_subshard=True, parent_key="pkg",
        owns_root=False, depth=2,
    )
    b = _shard("pkg__b", "pkg/b", is_subshard=True, parent_key="pkg", owns_root=False, depth=1)
    plan = _plan_for([spine, a_spine, g1, g2, b], {"pkg": "pkg"})
    skeleton = _skel(
        _node(
            "pkg",
            children=(
                _node("pkg/a", children=(_node("pkg/a/g1"), _node("pkg/a/g2"))),
                _node("pkg/b"),
            ),
        )
    )

    merged = merge_fragments(
        [
            (b, _frag(_mod("pkg/b"))),
            (g2, _frag(_mod("pkg/a/g2"))),
            (spine, _frag(_mod("pkg"))),
            (g1, _frag(_mod("pkg/a/g1"))),
            (a_spine, _frag(_mod("pkg/a"))),
        ],
        skeleton=skeleton,
        plan=plan,
    )

    top = merged.modules[0]
    assert [s.path for s in top.submodules] == ["pkg/a", "pkg/b"]
    a_module = top.submodules[0]
    assert [s.path for s in a_module.submodules] == ["pkg/a/g1", "pkg/a/g2"]


def test_merge_never_manufactures_a_single_child_parent(tmp_path) -> None:
    """The split gate guarantees ≥2 grafted children; assert the merge honors it."""
    spine = _shard(
        "pkg__spine", "pkg", is_subshard=True, parent_key="pkg", depth=1,
        promoted_children=["pkg/a", "pkg/b"],
    )
    a = _shard("pkg__a", "pkg/a", is_subshard=True, parent_key="pkg", owns_root=False, depth=1)
    b = _shard("pkg__b", "pkg/b", is_subshard=True, parent_key="pkg", owns_root=False, depth=1)
    plan = _plan_for([spine, a, b], {"pkg": "pkg"})
    skeleton = _skel(_node("pkg", children=(_node("pkg/a"), _node("pkg/b"))))
    merged = merge_fragments(
        [(spine, _frag(_mod("pkg"))), (a, _frag(_mod("pkg/a"))), (b, _frag(_mod("pkg/b")))],
        skeleton=skeleton,
        plan=plan,
    )
    assert len(merged.modules[0].submodules) >= 2


def test_merge_rejects_two_shards_emitting_the_same_path() -> None:
    a = _shard("alpha", "alpha")
    b = _shard("beta", "beta")
    plan = _plan_for([a, b], {"alpha": "alpha", "beta": "beta"})
    with pytest.raises(ExtractorValidationError, match="partition is broken"):
        merge_fragments(
            [(a, _frag(_mod("alpha"))), (b, _frag(_mod("alpha")))],
            skeleton=_skel(_node("alpha"), _node("beta")),
            plan=plan,
        )


def test_merge_rejects_a_missing_graft_target() -> None:
    a = _shard(
        "pkg__a", "pkg/a", is_subshard=True, parent_key="pkg", owns_root=False, depth=1
    )
    plan = _plan_for([a], {"pkg": "pkg"})
    with pytest.raises(ExtractorValidationError, match="no shard fragment for its own root"):
        merge_fragments(
            [(a, _frag(_mod("pkg/a")))],
            skeleton=_skel(_node("pkg", children=(_node("pkg/a"),))),
            plan=plan,
        )


# ── Partition + merge, as properties over many random shapes ──────────────
#
# The fixture tests above pin specific shapes; these assert the two invariants
# the whole completeness argument rests on — the partition is exhaustive and
# disjoint, and the merge neither invents nor drops a node — across the shard
# derivation's whole configuration space.


def _random_node(rng: random.Random, path: str, depth: int) -> SkeletonNode:
    kids = (
        [_random_node(rng, f"{path}/c{i}", depth + 1)
         for i in range(rng.choice([0, 0, 1, 2, 2, 3]))]
        if depth < 4
        else []
    )
    return SkeletonNode(
        path=path,
        direct_source_file_count=rng.choice([0, 1, 2, 3]),
        source_child_count=len(kids),
        # An empty `representative_files` is the namespace-only root that must
        # never be split, so let the generator produce both.
        representative_files=([f"{path}/f.py"] if rng.random() > 0.2 else []),
        required=rng.random() > 0.3,
        required_reasons=[],
        children=kids,
    )


def _random_case(seed: int) -> tuple[Skeleton, ExtractorConfig, random.Random]:
    rng = random.Random(seed)
    skeleton = _skel(
        *(_random_node(rng, f"t{i}", 0) for i in range(rng.choice([1, 1, 2, 3])))
    )
    config = _cfg(
        enrich_subshard_threshold=rng.choice([1, 2, 4, 40]),
        enrich_subshard_child_min=rng.choice([1, 2, 3]),
        enrich_subshard_max_depth=rng.choice([1, 2]),
        enrich_max_shards=rng.choice([2, 3, 5, 24]),
    )
    return skeleton, config, rng


@pytest.mark.parametrize("seed", range(60))
def test_shards_partition_the_inventory_exhaustively_and_disjointly(seed: int) -> None:
    skeleton, config, _ = _random_case(seed)
    plan = derive_enrich_shards(skeleton, config)

    owner_of: dict[str, str] = {}
    for shard in plan.shards:
        for path in shard.subtree.all_paths():
            assert path not in owner_of, (
                f"{path!r} is claimed by both {owner_of.get(path)!r} and {shard.key!r}"
            )
            owner_of[path] = shard.key
    assert set(owner_of) == skeleton.all_paths()

    # `owning_shard` (used by the branch precheck) must agree with that
    # partition, and must be total.
    for path in skeleton.all_paths():
        found = owning_shard(path, plan.shards)
        assert found is not None and found.key == owner_of[path]

    # No split may produce a single-child spine (Stage-5 Rule 4 would reject the
    # merged tree with no possible recovery). A spine whose root owns no source
    # file of its own is fine: it cites nothing.
    for shard in plan.shards:
        if shard.promoted_children:
            assert shard.owns_root
            assert len(shard.promoted_children) >= 2
        assert shard.depth <= config.enrich_subshard_max_depth
    assert len({s.key for s in plan.shards}) == len(plan.shards)
    assert len(plan.shards) <= max(config.enrich_max_shards, len(skeleton.nodes))


def _emit_everything(node: SkeletonNode, *, top: bool) -> dict:
    return _mod(node.path, subs=[_emit_everything(c, top=False) for c in node.children])


@pytest.mark.parametrize("seed", range(60))
def test_merge_is_loss_free_and_order_independent(seed: int) -> None:
    skeleton, config, rng = _random_case(seed)
    plan = derive_enrich_shards(skeleton, config)
    pairs = [
        (s, _frag(_emit_everything(s.subtree.nodes[0], top=True))) for s in plan.shards
    ]

    union: set[str] = set()
    for _, fragment in pairs:
        union |= fragment.emitted_paths()

    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    assert merged.emitted_paths() == union == skeleton.all_paths()
    assert compute_coverage(merged, skeleton).missing == []
    assert len(merged.modules) == len(plan.branch_roots)

    shuffled = list(pairs)
    rng.shuffle(shuffled)
    assert (
        merge_fragments(shuffled, skeleton=skeleton, plan=plan).model_dump_json()
        == merged.model_dump_json()
    )


# ── Shard-local validators ────────────────────────────────────────────────


def test_shard_scope_requires_exactly_one_top_level_module_at_the_root() -> None:
    shard = _shard("alpha", "alpha")
    with pytest.raises(CrossArtifactError, match="exactly one top-level module"):
        validate_shard_scope(shard, _frag(_mod("alpha"), _mod("beta")))
    with pytest.raises(CrossArtifactError, match="expected 'alpha'"):
        validate_shard_scope(shard, _frag(_mod("beta")))


def test_shard_scope_rejects_an_out_of_scope_emit() -> None:
    shard = _shard("alpha", "alpha")
    with pytest.raises(CrossArtifactError, match="outside"):
        validate_shard_scope(shard, _frag(_mod("alpha", subs=[_sub("beta/x")])))


def test_shard_scope_rejects_an_out_of_scope_fold() -> None:
    shard = _shard("alpha", "alpha")
    fold = {
        "path": "beta/x",
        "into": "alpha",
        "reason": "r",
        "evidence_files": ["beta/x/f.py"],
    }
    with pytest.raises(CrossArtifactError, match="outside its scope"):
        validate_shard_scope(shard, _frag(_mod("alpha"), folds=[fold]))


def test_spine_main_files_may_not_reach_into_a_promoted_child() -> None:
    spine = _shard(
        "pkg__spine", "pkg", is_subshard=True, parent_key="pkg", depth=1,
        promoted_children=["pkg/a"],
    )
    frag = _frag(
        _mod("pkg", files=[{"path": "pkg/a/core.py", "role": "Core."}])
    )
    with pytest.raises(CrossArtifactError) as exc:
        validate_spine_main_files(spine, frag)
    assert "pkg/a/core.py" in str(exc.value)
    assert "pkg/a" in str(exc.value)


def test_primary_shard_is_the_branch_owner_not_the_first_key() -> None:
    """`vllm__engine` sorts before `vllm__spine`, so a lexicographic-key rule
    would make an arbitrary child primary."""
    spine = _shard(
        "vllm__spine", "vllm", is_subshard=True, parent_key="vllm", depth=1,
        promoted_children=["vllm/engine", "vllm/v1"],
    )
    engine = _shard(
        "vllm__engine", "vllm/engine", is_subshard=True, parent_key="vllm",
        owns_root=False, depth=1,
    )
    v1 = _shard(
        "vllm__v1", "vllm/v1", is_subshard=True, parent_key="vllm",
        owns_root=False, depth=1,
    )
    csrc = _shard("zcsrc", "zcsrc")
    plan = _plan_for([engine, spine, v1, csrc], {"vllm": "vllm", "zcsrc": "zcsrc"})
    assert plan.primary_shard() is spine


# ── The spine hazards, against a real filesystem ─────────────────────────


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _spine_repo(tmp_path: Path) -> Path:
    """`pkg` owns one direct file, two promotable children, two light leaves."""
    _write(tmp_path / "pkg" / "root.py")
    _write(tmp_path / "pkg" / "a" / "a1.py")
    _write(tmp_path / "pkg" / "a" / "a2.py")
    _write(tmp_path / "pkg" / "b" / "b1.py")
    _write(tmp_path / "pkg" / "b" / "b2.py")
    _write(tmp_path / "pkg" / "light1" / "l.py")
    _write(tmp_path / "pkg" / "light2" / "m.py")
    return tmp_path


SPINE_CFG = dict(enrich_subshard_threshold=2, enrich_subshard_child_min=1)


def _spine_plan(repo: Path) -> tuple[Skeleton, ShardPlan]:
    skeleton = build_skeleton(repo, "")
    return skeleton, derive_enrich_shards(skeleton, _cfg(**SPINE_CFG))


def test_spine_repo_splits_as_designed(tmp_path) -> None:
    skeleton, plan = _spine_plan(_spine_repo(tmp_path))
    assert sorted(s.key for s in plan.shards) == ["pkg__a", "pkg__b", "pkg__spine"]
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.promoted_children == ["pkg/a", "pkg/b"]
    assert "pkg/light1" in spine.subtree.all_paths()


def _spine_fragment(main_files=None) -> EnrichedTree:
    """Spine emits `pkg` + exactly ONE light child, folding the other."""
    return _frag(
        _mod(
            "pkg",
            subs=[_sub("pkg/light1", files=[{"path": "pkg/light1/l.py", "role": "L."}])],
            files=main_files
            or [
                {"path": "pkg/root.py", "role": "Root."},
                {"path": "pkg/light2/m.py", "role": "Folded helper."},
            ],
        ),
        folds=[
            {
                "path": "pkg/light2",
                "into": "pkg",
                "reason": "single-file helper",
                "evidence_files": ["pkg/light2/m.py"],
            }
        ],
    )


def _repository() -> Repository:
    return Repository(name="demo", summary="A demo repo.", source_root="")


def test_single_child_spine_passes_locally_and_the_merge_passes_rule_4(tmp_path) -> None:
    """Rule-4 deferral.

    The spine's pruned subtree makes it look like a single-child parent; the
    merged branch module has three children. Without `rule4_exempt_paths` this
    fixture would burn the shard's one repair and then hard-fail.
    """
    repo = _spine_repo(tmp_path)
    skeleton, plan = _spine_plan(repo)
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    fragment = _spine_fragment()
    assert len(fragment.modules[0].submodules) == 1

    # Local (subtree-scoped) validation passes only because the spine's root is
    # exempt from Rule 4.
    validate_enriched_tree(
        fragment,
        repo,
        _repository(),
        spine.subtree,
        rule4_exempt_paths={spine.root_path},
    )
    with pytest.raises(CrossArtifactError, match="single child"):
        validate_enriched_tree(
            fragment,
            repo,
            _repository(),
            spine.subtree,
        )

    # The merged tree passes the *unchanged* full Rule-4 check.
    pairs = [
        (spine, fragment),
        (next(s for s in plan.shards if s.key == "pkg__a"), _frag(_mod("pkg/a", files=[
            {"path": "pkg/a/a1.py", "role": "A1."}]))),
        (next(s for s in plan.shards if s.key == "pkg__b"), _frag(_mod("pkg/b", files=[
            {"path": "pkg/b/b1.py", "role": "B1."}]))),
    ]
    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    assert len(merged.modules[0].submodules) == 3
    validate_enriched_tree(merged, repo, _repository(), skeleton)
    assert compute_coverage(merged, skeleton).missing == []


def test_branch_precheck_uses_the_unpruned_branch_skeleton(tmp_path) -> None:
    repo = _spine_repo(tmp_path)
    _, plan = _spine_plan(repo)
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    a = next(s for s in plan.shards if s.key == "pkg__a")
    b = next(s for s in plan.shards if s.key == "pkg__b")

    complete = [
        (spine, _spine_fragment()),
        (a, _frag(_mod("pkg/a", files=[{"path": "pkg/a/a1.py", "role": "A1."}]))),
        (b, _frag(_mod("pkg/b", files=[{"path": "pkg/b/b1.py", "role": "B1."}]))),
    ]
    assert branch_coverage(plan, "pkg", complete).missing == []

    # Drop the `pkg/b` sub-shard: the branch-local union is now short a required
    # path, which the spine's own (pruned) coverage could never have seen.
    assert branch_coverage(plan, "pkg", complete[:2]).missing == ["pkg/b"]


def test_spine_main_file_under_a_promoted_child_is_fatal_after_the_merge(tmp_path) -> None:
    """The trap the shard-local predicate exists to catch.

    In the spine's own pruned fragment the file is legal (Rule 3 checks the
    nearest *emitted* owner, and `pkg/a` is not emitted there). After the merge
    `pkg/a` *is* emitted, so it becomes the owner and Stage 5 fails with no
    possible LLM recovery.
    """
    repo = _spine_repo(tmp_path)
    skeleton, plan = _spine_plan(repo)
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    a = next(s for s in plan.shards if s.key == "pkg__a")
    b = next(s for s in plan.shards if s.key == "pkg__b")

    bad = _spine_fragment(
        main_files=[
            {"path": "pkg/a/a2.py", "role": "Reaching into a promoted child."},
            {"path": "pkg/light2/m.py", "role": "Folded helper."},
        ]
    )
    # Legal in the spine's own subtree...
    validate_enriched_tree(
        bad,
        repo,
        _repository(),
        spine.subtree,
        rule4_exempt_paths={spine.root_path},
    )
    # ...and fatal after the merge.
    merged = merge_fragments(
        [
            (spine, bad),
            (a, _frag(_mod("pkg/a", files=[{"path": "pkg/a/a1.py", "role": "A1."}]))),
            (b, _frag(_mod("pkg/b", files=[{"path": "pkg/b/b1.py", "role": "B1."}]))),
        ],
        skeleton=skeleton,
        plan=plan,
    )
    with pytest.raises(CrossArtifactError, match="belongs to emitted descendant"):
        validate_enriched_tree(merged, repo, _repository(), skeleton)

    # The shard-local predicate is what stands between the two.
    with pytest.raises(CrossArtifactError, match="promoted child"):
        validate_spine_main_files(spine, bad)


def test_a_split_namespace_root_cites_nothing_and_survives_the_merge(tmp_path) -> None:
    """A branch root owning no direct source file is splittable end to end.

    Every file in the branch belongs to an emitted promoted child, so the spine's
    only legal answer is to cite nothing — and it must stay legal through the
    merge, where those children *are* emitted. Citing one of their files instead
    is still the fatal shape `validate_spine_main_files` catches.
    """
    repo = tmp_path
    _write(repo / "pkg" / "a" / "a1.py")
    _write(repo / "pkg" / "a" / "a2.py")
    _write(repo / "pkg" / "b" / "b1.py")
    _write(repo / "pkg" / "b" / "b2.py")
    skeleton = build_skeleton(repo, "")
    assert skeleton.nodes[0].representative_files == []  # a pure container

    plan = derive_enrich_shards(skeleton, _cfg(**SPINE_CFG))
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    a = next(s for s in plan.shards if s.key == "pkg__a")
    b = next(s for s in plan.shards if s.key == "pkg__b")
    assert spine.promoted_children == ["pkg/a", "pkg/b"]

    empty_spine = _frag(_mod("pkg", files=[]))
    children = [
        (a, _frag(_mod("pkg/a", files=[{"path": "pkg/a/a1.py", "role": "A1."}]))),
        (b, _frag(_mod("pkg/b", files=[{"path": "pkg/b/b1.py", "role": "B1."}]))),
    ]

    # Legal in the spine's own subtree...
    validate_enriched_tree(
        empty_spine,
        repo,
        _repository(),
        spine.subtree,
        rule4_exempt_paths={spine.root_path},
    )
    validate_spine_main_files(spine, empty_spine)

    # ...and legal after the merge, with full coverage.
    merged = merge_fragments(
        [(spine, empty_spine), *children], skeleton=skeleton, plan=plan
    )
    validate_enriched_tree(merged, repo, _repository(), skeleton)
    assert compute_coverage(merged, skeleton).missing == []

    # Reaching into a promoted child instead is still rejected.
    grabby = _frag(_mod("pkg", files=[{"path": "pkg/a/a2.py", "role": "Not mine."}]))
    with pytest.raises(CrossArtifactError, match="promoted child"):
        validate_spine_main_files(spine, grabby)


# ── The chain split, against a real filesystem ────────────────────────────


CHAIN_CFG = dict(enrich_subshard_threshold=2, enrich_subshard_child_min=2)


def _chain_repo(tmp_path: Path) -> Path:
    """vLLM's `rust/` in miniature: `pkg` → `pkg/src` → two heavy crates.

    `pkg` keeps siblings of its own (`proto`, `tools`) — that is what makes the
    chain splittable at all — and `pkg/src` keeps a light leaf that stays with
    the spine.
    """
    for sibling in ("proto", "tools"):
        _write(tmp_path / "pkg" / sibling / "s1.py")
        _write(tmp_path / "pkg" / sibling / "s2.py")
    for crate in ("a", "b"):
        _write(tmp_path / "pkg" / "src" / crate / f"{crate}1.py")
        _write(tmp_path / "pkg" / "src" / crate / f"{crate}2.py")
        for deep in ("deep1", "deep2"):
            _write(tmp_path / "pkg" / "src" / crate / deep / "d1.py")
            _write(tmp_path / "pkg" / "src" / crate / deep / "d2.py")
    _write(tmp_path / "pkg" / "src" / "light" / "l.py")
    return tmp_path


def _chain_plan(repo: Path) -> tuple[Skeleton, ShardPlan]:
    skeleton = build_skeleton(repo, "")
    return skeleton, derive_enrich_shards(skeleton, _cfg(**CHAIN_CFG))


def _chain_fragments(plan: ShardPlan) -> list[tuple[EnrichShard, EnrichedTree]]:
    """What a well-behaved model returns: the spine emits the whole chain
    (`pkg` → `pkg/src`) plus the paths left with it, and each crate its own."""
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    spine_fragment = _frag(
        _mod(
            "pkg",
            files=[],
            subs=[
                _sub(s, files=[{"path": f"pkg/{s.rsplit('/', 1)[-1]}/s1.py", "role": "S."}])
                for s in ("pkg/proto", "pkg/tools")
            ]
            + [
                _sub(
                    "pkg/src",
                    files=[],
                    subs=[
                        _sub(
                            "pkg/src/light",
                            files=[{"path": "pkg/src/light/l.py", "role": "L."}],
                        )
                    ],
                ),
            ],
        )
    )
    return [
        (spine, spine_fragment),
        *(
            (
                next(s for s in plan.shards if s.key == f"pkg__src__{crate}"),
                _frag(
                    _mod(
                        f"pkg/src/{crate}",
                        files=[{"path": f"pkg/src/{crate}/{crate}1.py", "role": "C."}],
                        subs=[
                            _sub(
                                f"pkg/src/{crate}/{deep}",
                                files=[
                                    {
                                        "path": f"pkg/src/{crate}/{deep}/d1.py",
                                        "role": "D.",
                                    }
                                ],
                            )
                            for deep in ("deep1", "deep2")
                        ],
                    )
                ),
            )
            for crate in ("a", "b")
        ),
    ]


def test_chain_repo_splits_at_the_descendant(tmp_path) -> None:
    _, plan = _chain_plan(_chain_repo(tmp_path))
    assert sorted(s.key for s in plan.shards) == [
        "pkg__spine",
        "pkg__src__a",
        "pkg__src__b",
    ]
    spine = next(s for s in plan.shards if s.key == "pkg__spine")
    assert spine.promoted_children == ["pkg/src/a", "pkg/src/b"]
    assert spine.promotion_parent == "pkg/src"
    assert "pkg/src/light" in spine.subtree.all_paths()


def test_chain_split_children_graft_under_the_promotion_parent(tmp_path) -> None:
    """The merge's object-tree position has to follow the physical path.

    Grafting these into the fragment's root module — which is all the merge had
    to do while every promoted child was an immediate child of the shard root —
    would put `pkg/src/a` beside `pkg/src` instead of inside it, and
    `validate_enriched_tree` rejects exactly that shape.
    """
    repo = _chain_repo(tmp_path)
    skeleton, plan = _chain_plan(repo)
    pairs = _chain_fragments(plan)

    merged = merge_fragments(pairs, skeleton=skeleton, plan=plan)
    root = merged.modules[0]
    assert [s.path for s in root.submodules] == ["pkg/proto", "pkg/src", "pkg/tools"]
    src = next(s for s in root.submodules if s.path == "pkg/src")
    assert [s.path for s in src.submodules] == [
        "pkg/src/a",
        "pkg/src/b",
        "pkg/src/light",
    ]

    validate_enriched_tree(merged, repo, _repository(), skeleton)
    assert compute_coverage(merged, skeleton).missing == []


def test_chain_split_spine_defers_rule_4_only_for_the_promotion_parent(tmp_path) -> None:
    """`pkg/src` looks single-child inside the shard and is exempt; `pkg` does
    not and is not. Fold both of `pkg`'s siblings away and the shard must fail
    *locally*, while it can still be repaired — the merge cannot give `pkg` a
    second child, because the promoted subtrees land under `pkg/src`."""
    repo = _chain_repo(tmp_path)
    _, plan = _chain_plan(repo)
    spine, fragment = _chain_fragments(plan)[0]

    validate_enriched_tree(
        fragment, repo, _repository(), spine.subtree,
        rule4_exempt_paths=spine.rule4_exempt_paths,
    )
    with pytest.raises(CrossArtifactError, match="single child"):
        validate_enriched_tree(fragment, repo, _repository(), spine.subtree)

    folded_siblings = _frag(
        _mod(
            "pkg",
            files=[
                {"path": "pkg/proto/s1.py", "role": "Folded."},
                {"path": "pkg/tools/s1.py", "role": "Folded."},
            ],
            subs=[_sub("pkg/src", files=[], subs=[
                _sub("pkg/src/light", files=[{"path": "pkg/src/light/l.py", "role": "L."}])
            ])],
        ),
        folds=[
            {
                "path": f"pkg/{sibling}",
                "into": "pkg",
                "reason": "thin generated shim",
                "evidence_files": [f"pkg/{sibling}/s1.py"],
            }
            for sibling in ("proto", "tools")
        ],
    )
    with pytest.raises(CrossArtifactError, match="single child.*'pkg'"):
        validate_enriched_tree(
            folded_siblings, repo, _repository(), spine.subtree,
            rule4_exempt_paths=spine.rule4_exempt_paths,
        )


def test_chain_split_spine_may_not_fold_its_promotion_parent(tmp_path) -> None:
    """Folding `pkg/src` passes every subtree-scoped rule and is unrecoverable
    after the merge, so it is caught where a repair is still possible."""
    repo = _chain_repo(tmp_path)
    _, plan = _chain_plan(repo)
    spine = next(s for s in plan.shards if s.key == "pkg__spine")

    folded_parent = _frag(
        _mod(
            "pkg",
            files=[{"path": "pkg/src/light/l.py", "role": "Folded."}],
            subs=[
                _sub(s, files=[{"path": f"pkg/{s.rsplit('/', 1)[-1]}/s1.py", "role": "S."}])
                for s in ("pkg/proto", "pkg/tools")
            ],
        ),
        folds=[
            {
                "path": "pkg/src",
                "into": "pkg",
                "reason": "container",
                "evidence_files": ["pkg/src/light/l.py"],
            },
            {
                "path": "pkg/src/light",
                "into": "pkg",
                "reason": "single-file helper",
                "evidence_files": ["pkg/src/light/l.py"],
            },
        ],
    )
    # Locally legal — `pkg/src`'s real content is in subtrees another shard owns
    # and is simply not visible here.
    validate_enriched_tree(
        folded_parent, repo, _repository(), spine.subtree,
        rule4_exempt_paths=spine.rule4_exempt_paths,
    )
    with pytest.raises(CrossArtifactError, match="must emit 'pkg/src'"):
        validate_promotion_parent(spine, folded_parent)

    # An ordinary spine, whose promotion parent is its own root, is unaffected —
    # `validate_shard_scope` already forces that module to exist.
    root_split = next(
        s for s in _spine_plan(_spine_repo(tmp_path / "other"))[1].shards
        if s.key == "pkg__spine"
    )
    validate_promotion_parent(root_split, _spine_fragment())
