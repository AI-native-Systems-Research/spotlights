"""Assignment-contract sharding invariants kept separate from the v1 planner."""

from __future__ import annotations

import pytest

from spotlights_engine.modules_extractor.derive import (
    derive_metadata_batches,
    resolve_assignments,
)
from spotlights_engine.modules_extractor.errors import ExtractorValidationError
from spotlights_engine.modules_extractor.extractor import ExtractorConfig
from spotlights_engine.modules_extractor.sharding import (
    assignment_node_weight,
    derive_assignment_shards,
    merge_assignment_fragments,
    validate_assignment_shard_scope,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleDecision,
    Skeleton,
    SkeletonNode,
)


def _node(
    path: str,
    *,
    direct: int = 1,
    required: bool = True,
    children: tuple[SkeletonNode, ...] = (),
) -> SkeletonNode:
    return SkeletonNode(
        path=path,
        direct_source_file_count=direct,
        subtree_source_file_count=direct
        + sum(child.subtree_source_file_count for child in children),
        source_child_count=len(children),
        representative_files=[f"{path}/main.py"] if direct else [],
        required=required,
        required_reasons=["synthetic"] if required else [],
        children=list(children),
    )


def _skeleton(*nodes: SkeletonNode) -> Skeleton:
    return Skeleton(source_root="pkg", nodes=list(nodes), inventory_fingerprint="fp")


def _split_skeleton() -> Skeleton:
    return _skeleton(
        _node(
            "pkg/app",
            children=(
                _node(
                    "pkg/app/feature",
                    required=False,
                    children=(
                        _node("pkg/app/feature/impl", required=False),
                    ),
                ),
            ),
        )
    )


def _config(**overrides) -> ExtractorConfig:
    values = {
        "contract": "assignments",
        "enrich_subshard_threshold": 1,
        "enrich_subshard_child_min": 2,
        "enrich_subshard_max_depth": 2,
        "enrich_max_shards": 2,
    }
    values.update(overrides)
    return ExtractorConfig(**values)


def test_assignment_weight_counts_optional_inventory_nodes() -> None:
    root = _split_skeleton().nodes[0]
    assert assignment_node_weight(root) == 3
    # Only the root is legacy-required; assignment mode still weighs all 3.
    assert sum(1 for node in root.children if node.required) == 0


def test_one_promotable_child_can_split_with_one_extra_budget_slot() -> None:
    plan = derive_assignment_shards(_split_skeleton(), _config())
    assert len(plan.shards) == 2
    spine = next(shard for shard in plan.shards if shard.is_spine)
    child = next(shard for shard in plan.shards if not shard.is_spine)
    assert spine.root_path == "pkg/app"
    assert spine.promoted_children == ["pkg/app/feature"]
    assert spine.is_branch_root
    assert child.root_path == "pkg/app/feature"
    assert not child.is_branch_root


def test_nested_shard_root_may_be_part_and_union_is_order_independent() -> None:
    skeleton = _split_skeleton()
    plan = derive_assignment_shards(skeleton, _config())
    spine = next(shard for shard in plan.shards if shard.is_spine)
    child = next(shard for shard in plan.shards if not shard.is_spine)
    spine_fragment = AssignmentTree(
        assignments={"pkg/app": "MODULE"},
        module_decisions={"pkg/app": ModuleDecision(keep_reason=None)},
    )
    child_fragment = AssignmentTree(
        assignments={
            "pkg/app/feature": "PART",
            "pkg/app/feature/impl": "PART",
        },
        module_decisions={},
    )
    validate_assignment_shard_scope(child, child_fragment)

    pairs = [(spine, spine_fragment), (child, child_fragment)]
    forward = merge_assignment_fragments(pairs)
    reverse = merge_assignment_fragments(list(reversed(pairs)))
    assert forward.model_dump() == reverse.model_dump()
    assert set(forward.assignments) == skeleton.all_paths()


def test_merge_rejects_module_without_its_decision_record() -> None:
    plan = derive_assignment_shards(_split_skeleton(), _config())
    spine = next(shard for shard in plan.shards if shard.is_spine)
    fragment = AssignmentTree(
        assignments={"pkg/app": "MODULE"}, module_decisions={}
    )
    with pytest.raises(ExtractorValidationError, match="missing decisions"):
        merge_assignment_fragments([(spine, fragment)])


def test_promoted_part_territory_returns_to_external_owner_metadata_scope() -> None:
    skeleton = _split_skeleton()
    plan = derive_assignment_shards(skeleton, _config())
    spine = next(shard for shard in plan.shards if shard.is_spine)
    child = next(shard for shard in plan.shards if not shard.is_spine)
    merged = merge_assignment_fragments(
        [
            (
                spine,
                AssignmentTree(
                    assignments={"pkg/app": "MODULE"},
                    module_decisions={
                        "pkg/app": ModuleDecision(keep_reason=None)
                    },
                ),
            ),
            (
                child,
                AssignmentTree(
                    assignments={
                        "pkg/app/feature": "PART",
                        "pkg/app/feature/impl": "PART",
                    },
                    module_decisions={},
                ),
            ),
        ]
    )
    resolved = resolve_assignments(merged, skeleton, 15)
    metadata_plan = derive_metadata_batches(resolved, skeleton, _config())
    scope = metadata_plan.batches[0].scopes[0]
    assert scope.path == "pkg/app"
    assert scope.territory_node_count == 3
    assert "pkg/app/feature/main.py" in scope.representative_files


def test_promoted_module_gets_its_own_metadata_territory() -> None:
    skeleton = _split_skeleton()
    assignments = AssignmentTree(
        assignments={
            "pkg/app": "MODULE",
            "pkg/app/feature": "MODULE",
            "pkg/app/feature/impl": "PART",
        },
        module_decisions={
            "pkg/app": ModuleDecision(keep_reason=None),
            "pkg/app/feature": ModuleDecision(
                keep_reason="Independently developed feature subsystem"
            ),
        },
    )
    resolved = resolve_assignments(assignments, skeleton, 15)
    plan = derive_metadata_batches(resolved, skeleton, _config())
    scopes = {scope.path: scope for batch in plan.batches for scope in batch.scopes}
    assert scopes["pkg/app"].territory_node_count == 1
    assert scopes["pkg/app"].child_module_paths == ["pkg/app/feature"]
    assert scopes["pkg/app/feature"].territory_node_count == 2


def test_top_level_branch_floor_can_exceed_configured_shard_cap() -> None:
    skeleton = _skeleton(
        _node("pkg/a"),
        _node("pkg/b"),
        _node("pkg/c"),
    )
    plan = derive_assignment_shards(
        skeleton,
        _config(enrich_sharding="top_level_only", enrich_max_shards=2),
    )
    assert len(plan.shards) == 3
    assert all(shard.is_branch_root for shard in plan.shards)
