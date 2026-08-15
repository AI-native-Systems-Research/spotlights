"""Derivation tests: resolution, tree building, batching, territories, lints."""

from __future__ import annotations

import pytest

from spotlights_engine.modules_extractor.coverage import CrossArtifactError
from spotlights_engine.modules_extractor.derive import (
    compute_assignment_lints,
    derive_metadata_batches,
    derive_module_forest,
    derive_project_tree,
    module_children,
    resolve_assignments,
    resolve_territories,
    territory_source_file_counts,
    verify_resolved_assignments,
)
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleDecision,
    ModuleInfo,
    Skeleton,
    SkeletonNode,
)
from spotlights_engine.schemas.project import Repository


def _node(
    path: str,
    *,
    direct: int = 1,
    children: tuple[SkeletonNode, ...] = (),
    required: bool = True,
) -> SkeletonNode:
    return SkeletonNode(
        path=path,
        direct_source_file_count=direct,
        subtree_source_file_count=direct
        + sum(c.subtree_source_file_count for c in children),
        source_child_count=len(children),
        representative_files=[f"{path}/file_a.py", f"{path}/main.py"][
            : (2 if direct else 0)
        ],
        required=required,
        required_reasons=["synthetic"] if required else [],
        children=list(children),
    )


def _skeleton(*nodes: SkeletonNode, source_root: str = "pkg") -> Skeleton:
    return Skeleton(
        source_root=source_root,
        nodes=list(nodes),
        inventory_fingerprint="fp",
    )


def _tree(
    assignments: dict[str, str], reasons: dict[str, str | None] | None = None
) -> AssignmentTree:
    decisions = {
        p: ModuleDecision(keep_reason=(reasons or {}).get(p))
        for p, label in assignments.items()
        if label == "MODULE"
    }
    return AssignmentTree(assignments=assignments, module_decisions=decisions)


# The worked kv_offload example from the design, scaled to the fixture sizes.
def _kv_skeleton() -> Skeleton:
    return _skeleton(
        _node(
            "pkg/kv_offload",
            direct=4,
            children=(
                _node(
                    "pkg/kv_offload/cpu",
                    direct=6,
                    children=(_node("pkg/kv_offload/cpu/policies", direct=4),),
                ),
                _node(
                    "pkg/kv_offload/tiering",
                    direct=5,
                    children=(
                        _node("pkg/kv_offload/tiering/fs", direct=3),
                        _node("pkg/kv_offload/tiering/obj", direct=2),
                        _node(
                            "pkg/kv_offload/tiering/p2p",
                            direct=3,
                            children=(
                                _node("pkg/kv_offload/tiering/p2p/control", direct=2),
                                _node("pkg/kv_offload/tiering/p2p/data", direct=2),
                                _node("pkg/kv_offload/tiering/p2p/session", direct=2),
                            ),
                        ),
                    ),
                ),
            ),
        )
    )


def _kv_assignments() -> AssignmentTree:
    assignments = {
        "pkg/kv_offload": "MODULE",  # 33 files, anchor
        "pkg/kv_offload/cpu": "MODULE",  # 10 files, independent
        "pkg/kv_offload/cpu/policies": "PART",
        "pkg/kv_offload/tiering": "MODULE",  # 19 files > threshold
        "pkg/kv_offload/tiering/fs": "MODULE",  # 3 files, independent
        "pkg/kv_offload/tiering/obj": "PART",
        "pkg/kv_offload/tiering/p2p": "PART",
        "pkg/kv_offload/tiering/p2p/control": "PART",
        "pkg/kv_offload/tiering/p2p/data": "PART",
        "pkg/kv_offload/tiering/p2p/session": "PART",
    }
    return _tree(
        assignments,
        {
            "pkg/kv_offload/cpu": "Self-contained CPU tier developed independently",
            "pkg/kv_offload/tiering/fs": "Storage backend against the tier interface",
        },
    )


# ── resolve_assignments ───────────────────────────────────────────────────


def test_resolution_owners_and_origins() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)

    assert resolved.owners["pkg/kv_offload/cpu/policies"] == "pkg/kv_offload/cpu"
    assert resolved.owners["pkg/kv_offload/tiering/obj"] == "pkg/kv_offload/tiering"
    assert (
        resolved.owners["pkg/kv_offload/tiering/p2p/data"]
        == "pkg/kv_offload/tiering"
    )
    assert resolved.owners["pkg/kv_offload/tiering/fs"] == "pkg/kv_offload/tiering/fs"

    assert resolved.origins["pkg/kv_offload"] == "top_level_anchor"
    assert resolved.origins["pkg/kv_offload/tiering"] == "size"
    assert resolved.origins["pkg/kv_offload/cpu"] == "independent"
    assert resolved.origins["pkg/kv_offload/tiering/obj"] == "part"
    assert resolved.collision_precedence == {}
    assert resolved.merge_threshold == 15


def test_resolution_rejects_non_total_assignments() -> None:
    skel = _kv_skeleton()
    tree = _kv_assignments()
    partial = AssignmentTree(
        assignments={
            p: label
            for p, label in tree.assignments.items()
            if p != "pkg/kv_offload/tiering/obj"
        },
        module_decisions=dict(tree.module_decisions),
    )
    with pytest.raises(CrossArtifactError, match="not total"):
        resolve_assignments(partial, skel, 15)


def test_resolution_rejects_label_policy_violations() -> None:
    skel = _kv_skeleton()
    tree = _kv_assignments()
    bad = AssignmentTree(
        assignments={**tree.assignments, "pkg/kv_offload/tiering": "PART"},
        module_decisions={
            p: d
            for p, d in tree.module_decisions.items()
            if p != "pkg/kv_offload/tiering"
        },
    )
    with pytest.raises(CrossArtifactError, match="must be MODULE"):
        resolve_assignments(bad, skel, 15)


def test_resolution_fails_fast_on_top_level_collision() -> None:
    skel = _skeleton(
        _node("top-x", direct=3), _node("top_x", direct=5), source_root=""
    )
    tree = _tree({"top-x": "MODULE", "top_x": "MODULE"})
    with pytest.raises(CrossArtifactError, match="top-level"):
        resolve_assignments(tree, skel, 15)


def test_verify_rejects_tampered_owner_map() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    verify_resolved_assignments(resolved, skel)  # clean copy passes

    tampered = resolved.model_copy(deep=True)
    tampered.owners["pkg/kv_offload/tiering/obj"] = "pkg/kv_offload"
    with pytest.raises(CrossArtifactError, match="owners disagree"):
        verify_resolved_assignments(tampered, skel)

    tampered2 = resolved.model_copy(deep=True)
    tampered2.origins["pkg/kv_offload/cpu"] = "size"
    with pytest.raises(CrossArtifactError, match="origins disagree"):
        verify_resolved_assignments(tampered2, skel)


# ── Tree derivation ───────────────────────────────────────────────────────


def _metadata_for(resolved) -> dict[str, ModuleInfo]:
    return {
        m: ModuleInfo.model_validate(
            {"description": f"Module at {m}.", "main_files": []}
        )
        for m in resolved.module_paths()
    }


def test_derived_tree_nests_by_nearest_module_ancestor() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    tree = derive_project_tree(
        Repository(name="r", summary="s", source_root="pkg"),
        resolved,
        _metadata_for(resolved),
    )
    qns = [qn for qn, _ in tree.walk()]
    assert qns == [
        "kv_offload",
        "kv_offload/cpu",
        "kv_offload/tiering",
        "kv_offload/tiering/fs",
    ]
    top = tree.modules[0]
    assert [s.name for s in top.submodules] == ["cpu", "tiering"]
    assert [s.name for s in top.submodules[1].submodules] == ["fs"]


def test_single_child_chains_are_preserved() -> None:
    # Level 2 retires Rule 4: a module with exactly one child module is legal.
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/only", direct=20),
            ),
        )
    )
    assignments = {"pkg/app": "MODULE", "pkg/app/only": "MODULE"}
    resolved = resolve_assignments(_tree(assignments), skel, 15)
    tree = derive_project_tree(
        Repository(name="r", summary="s", source_root="pkg"),
        resolved,
        _metadata_for(resolved),
    )
    assert len(tree.modules) == 1
    assert len(tree.modules[0].submodules) == 1
    assert tree.modules[0].submodules[0].name == "only"


def test_derive_module_forest_is_path_only_and_tolerates_external_parts() -> None:
    # A nested fragment: its local root is PART (owner lives outside), and its
    # local modules still get the public-name gate.
    fragment = _tree(
        {
            "pkg/app/sub": "PART",
            "pkg/app/sub/x": "MODULE",
            "pkg/app/sub/y": "MODULE",
        },
        {"pkg/app/sub/x": "why", "pkg/app/sub/y": "why"},
    )
    forest = derive_module_forest(
        Repository(name="r", summary="s", source_root="pkg"), fragment
    )
    assert sorted(m.path for m in forest.modules) == [
        "pkg/app/sub/x",
        "pkg/app/sub/y",
    ]


def test_derive_module_forest_rejects_qn_collision() -> None:
    fragment = _tree(
        {"pkg/app/foo-bar": "MODULE", "pkg/app/foo_bar": "MODULE"},
        {"pkg/app/foo-bar": "why", "pkg/app/foo_bar": "why"},
    )
    with pytest.raises(ValueError, match="duplicate qualified name"):
        derive_module_forest(
            Repository(name="r", summary="s", source_root="pkg"), fragment
        )


def test_derive_project_tree_requires_exact_metadata_keys() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    metadata = _metadata_for(resolved)
    metadata.pop("pkg/kv_offload/cpu")
    with pytest.raises(CrossArtifactError, match="metadata union"):
        derive_project_tree(
            Repository(name="r", summary="s", source_root="pkg"),
            resolved,
            metadata,
        )


def test_empty_skeleton_resolves_and_derives_empty_tree() -> None:
    skel = _skeleton(source_root="pkg")
    resolved = resolve_assignments(
        AssignmentTree(assignments={}, module_decisions={}), skel, 15
    )
    tree = derive_project_tree(
        Repository(name="r", summary="s", source_root="pkg"), resolved, {}
    )
    assert tree.modules == []
    plan = derive_metadata_batches(resolved, skel, ExtractorConfig())
    assert plan.batches == []


def test_thousand_node_scale() -> None:
    tops = []
    assignments: dict[str, str] = {}
    reasons: dict[str, str | None] = {}
    for i in range(10):
        mids = []
        for j in range(10):
            leaves = tuple(
                _node(f"pkg/t{i}/m{j}/l{k}", direct=1) for k in range(10)
            )
            mids.append(_node(f"pkg/t{i}/m{j}", direct=2, children=leaves))
            assignments[f"pkg/t{i}/m{j}"] = "MODULE"  # 12 files ≤ 15
            reasons[f"pkg/t{i}/m{j}"] = "independent mid-tier unit"
            for k in range(10):
                assignments[f"pkg/t{i}/m{j}/l{k}"] = "PART"
        tops.append(_node(f"pkg/t{i}", direct=1, children=tuple(mids)))
        assignments[f"pkg/t{i}"] = "MODULE"
    skel = _skeleton(*tops)
    assert len(skel.all_paths()) == 1110

    resolved = resolve_assignments(_tree(assignments, reasons), skel, 15)
    tree = derive_project_tree(
        Repository(name="r", summary="s", source_root="pkg"),
        resolved,
        _metadata_for(resolved),
    )
    assert len(list(tree.walk())) == 110
    plan = derive_metadata_batches(resolved, skel, ExtractorConfig())
    assert sum(len(b.module_paths) for b in plan.batches) == 110
    assert sum(b.weight for b in plan.batches) == 1110


# ── Territories and metadata batching ─────────────────────────────────────


def test_territory_counts_partition_the_inventory() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    territories = resolve_territories(resolved)
    all_owned = sorted(p for paths in territories.values() for p in paths)
    assert all_owned == sorted(skel.all_paths())

    counts = territory_source_file_counts(resolved, skel)
    assert counts["pkg/kv_offload"] == 4
    assert counts["pkg/kv_offload/cpu"] == 10
    assert counts["pkg/kv_offload/tiering"] == 16  # 5 + obj 2 + p2p subtree 9
    assert counts["pkg/kv_offload/tiering/fs"] == 3
    assert sum(counts.values()) == 33


def test_metadata_batching_is_deterministic_and_bounded() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    config = ExtractorConfig(enrich_subshard_threshold=4, enrich_max_shards=3)
    plan = derive_metadata_batches(resolved, skel, config)
    plan2 = derive_metadata_batches(resolved, skel, config)
    assert plan.model_dump() == plan2.model_dump()

    all_paths = sorted(p for b in plan.batches for p in b.module_paths)
    assert all_paths == sorted(resolved.module_paths())
    # 10 territory nodes total across 4 modules; threshold 4 -> 3 bins.
    assert len(plan.batches) == 3
    # tiering owns 6 nodes (> threshold 4): its batch gets the full timeout.
    tiering_batch = next(
        b for b in plan.batches if "pkg/kv_offload/tiering" in b.module_paths
    )
    assert tiering_batch.use_full_timeout

    scope = next(
        s
        for b in plan.batches
        for s in b.scopes
        if s.path == "pkg/kv_offload/tiering"
    )
    assert scope.child_module_paths == ["pkg/kv_offload/tiering/fs"]
    assert scope.territory_node_count == 6
    assert scope.territory_source_file_count == 16
    assert scope.keep_reason is None
    assert scope.origin == "size"


def test_metadata_scope_seeds_are_entry_first() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    plan = derive_metadata_batches(resolved, skel, ExtractorConfig())
    scope = next(
        s
        for b in plan.batches
        for s in b.scopes
        if s.path == "pkg/kv_offload/cpu"
    )
    # main.py entries sort before the lexical rest, capped at 5.
    assert scope.representative_files[0].endswith("main.py")
    assert len(scope.representative_files) <= 5


# ── Lints ─────────────────────────────────────────────────────────────────


def test_tiny_leaf_and_fragmented_children_lints() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=8,
            children=(
                _node("pkg/app/a", direct=2),
                _node("pkg/app/b", direct=1),
                _node("pkg/app/big", direct=20),
            ),
        )
    )
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/a": "MODULE",
        "pkg/app/b": "MODULE",
        "pkg/app/big": "MODULE",
    }
    reasons = {"pkg/app/a": "why", "pkg/app/b": "why"}
    resolved = resolve_assignments(_tree(assignments, reasons), skel, 15)
    lints = compute_assignment_lints(resolved, skel)

    by_code = {}
    for lint in lints:
        by_code.setdefault(lint.code, []).append(lint)
    assert {lint.path for lint in by_code["tiny_leaf"]} == {
        "pkg/app/a",
        "pkg/app/b",
    }
    # Mean over app's *leaf* children only: a(2), b(1), big(20) -> 23/3 >= 3,
    # so no fragmented_children fires for pkg/app.
    assert "fragmented_children" not in by_code


def test_fragmented_children_fires_on_tiny_leaf_cluster() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=30,
            children=(
                _node("pkg/app/a", direct=2),
                _node("pkg/app/b", direct=1),
            ),
        )
    )
    assignments = {"pkg/app": "MODULE", "pkg/app/a": "MODULE", "pkg/app/b": "MODULE"}
    reasons = {"pkg/app/a": "why", "pkg/app/b": "why"}
    resolved = resolve_assignments(_tree(assignments, reasons), skel, 15)
    lints = compute_assignment_lints(resolved, skel)
    fragmented = [lint for lint in lints if lint.code == "fragmented_children"]
    assert [lint.path for lint in fragmented] == ["pkg/app"]
    assert fragmented[0].leaf_child_count == 2
    assert fragmented[0].mean_leaf_territory == 1.5


def test_top_level_anchor_is_exempt_from_tiny_leaf() -> None:
    skel = _skeleton(_node("pkg/tiny", direct=1))
    resolved = resolve_assignments(_tree({"pkg/tiny": "MODULE"}), skel, 15)
    assert compute_assignment_lints(resolved, skel) == []


def test_module_children_map() -> None:
    skel = _kv_skeleton()
    resolved = resolve_assignments(_kv_assignments(), skel, 15)
    children = module_children(resolved)
    assert children["pkg/kv_offload"] == [
        "pkg/kv_offload/cpu",
        "pkg/kv_offload/tiering",
    ]
    assert children["pkg/kv_offload/tiering"] == ["pkg/kv_offload/tiering/fs"]
    assert children["pkg/kv_offload/tiering/fs"] == []
