"""v2 (assignment-contract) decision report: build, render, offline dispatch."""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.modules_extractor.assignments import (
    AssignmentIssue,
    compute_assignment_coverage,
)
from spotlights_engine.modules_extractor.derive import (
    compute_assignment_lints,
    resolve_assignments,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleDecision,
    Skeleton,
    SkeletonNode,
)
from spotlights_engine.modules_extractor.tree_report import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    build_tree_decision_report_v2,
    detect_report_version,
    load_any_report_from_run_dir,
    load_report_v2_from_run_dir,
    render_markdown_v2,
    report_json_text,
)


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
        representative_files=[],
        required=required,
        required_reasons=["synthetic"] if required else [],
        children=list(children),
    )


def _skeleton() -> Skeleton:
    return Skeleton(
        source_root="pkg",
        nodes=[
            _node(
                "pkg/app",
                direct=2,
                children=(
                    _node("pkg/app/big", direct=20),
                    _node("pkg/app/small", direct=2),
                ),
            )
        ],
        inventory_fingerprint="fp",
    )


def _tree() -> AssignmentTree:
    return AssignmentTree(
        assignments={
            "pkg/app": "MODULE",
            "pkg/app/big": "MODULE",
            "pkg/app/small": "PART",
        },
        module_decisions={
            "pkg/app": ModuleDecision(keep_reason=None),
            "pkg/app/big": ModuleDecision(keep_reason=None),
        },
    )


def _build(**over):
    skeleton = _skeleton()
    tree = _tree()
    resolved = resolve_assignments(tree, skeleton, 15)
    kwargs = dict(
        assignments=tree,
        resolved=resolved,
        metadata=None,
        coverage=compute_assignment_coverage(tree, skeleton),
        lints=compute_assignment_lints(resolved, skeleton),
        issues=None,
        decision=None,
        merge_threshold=15,
    )
    kwargs.update(over)
    return build_tree_decision_report_v2(skeleton, **kwargs)


def test_summary_reconciles_with_the_inventory() -> None:
    report = _build()
    s = report.summary
    assert s.total_nodes == 3
    assert s.module_parent + s.module_leaf + s.part + s.missing == s.total_nodes
    assert s.module_parent == 1  # pkg/app owns the nested big module
    assert s.module_leaf == 1
    assert s.part == 1
    assert s.missing == 0

    nodes = {n.path: n for n in report.iter_nodes()}
    assert nodes["pkg/app"].decision == "module_parent"
    assert nodes["pkg/app/big"].decision == "module_leaf"
    assert nodes["pkg/app/small"].decision == "part"
    assert nodes["pkg/app/small"].owner == "pkg/app"
    assert nodes["pkg/app"].origin == "top_level_anchor"
    assert nodes["pkg/app/big"].origin == "size"
    assert nodes["pkg/app"].territory_source_file_count == 4  # app + small
    assert nodes["pkg/app/big"].territory_source_file_count == 20


def test_missing_and_invalid_are_orthogonal() -> None:
    skeleton = _skeleton()
    partial = AssignmentTree(
        assignments={"pkg/app": "MODULE", "pkg/app/big": "PART"},
        module_decisions={"pkg/app": ModuleDecision(keep_reason=None)},
    )
    issues = [
        AssignmentIssue(
            code="large_path_not_module",
            path="pkg/app/big",
            detail="pkg/app/big must be MODULE",
        )
    ]
    report = build_tree_decision_report_v2(
        skeleton,
        assignments=partial,
        resolved=None,
        metadata=None,
        coverage=compute_assignment_coverage(partial, skeleton, invalid=issues),
        lints=None,
        issues=issues,
        decision=None,
        merge_threshold=15,
    )
    nodes = {n.path: n for n in report.iter_nodes()}
    # The raw label drives the bucket; the issue rides as an invalid code.
    assert nodes["pkg/app/big"].decision == "part"
    assert nodes["pkg/app/big"].invalid_codes == ["large_path_not_module"]
    assert nodes["pkg/app/small"].decision == "missing"
    s = report.summary
    assert s.missing == 1
    assert s.invalid_nodes == 1
    assert s.module_parent + s.module_leaf + s.part + s.missing == s.total_nodes


def test_report_without_any_assignments_marks_everything_missing() -> None:
    report = build_tree_decision_report_v2(
        _skeleton(),
        assignments=None,
        resolved=None,
        metadata=None,
        coverage=None,
        lints=None,
        issues=None,
        decision=None,
        merge_threshold=15,
    )
    assert report.summary.missing == report.summary.total_nodes == 3


def test_markdown_is_deterministic_and_carries_the_header() -> None:
    report = _build()
    md1 = render_markdown_v2(report)
    md2 = render_markdown_v2(_build())
    assert md1 == md2
    assert "**merge threshold**: 15" in md1
    assert "**contract**: `assignments`" in md1
    assert "`●` module parent" in md1
    assert report_json_text(report) == report_json_text(_build())


def test_extra_assignment_keys_surface_at_the_root() -> None:
    skeleton = _skeleton()
    tree = AssignmentTree(
        assignments={**_tree().assignments, "pkg/ghost": "PART"},
        module_decisions=dict(_tree().module_decisions),
    )
    report = build_tree_decision_report_v2(
        skeleton,
        assignments=tree,
        resolved=None,
        metadata=None,
        coverage=None,
        lints=None,
        issues=None,
        decision=None,
        merge_threshold=15,
    )
    assert report.extra_assignments == ["pkg/ghost"]
    assert "Extra assignment keys" in render_markdown_v2(report)


# ── Offline loading and dispatch ──────────────────────────────────────────


def _fake_v2_run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "modules_extractor"
    (run / "02_skeleton").mkdir(parents=True)
    (run / "02_skeleton" / "skeleton.json").write_text(
        json.dumps(_skeleton().model_dump(mode="json")), encoding="utf-8"
    )
    resolved = resolve_assignments(_tree(), _skeleton(), 15)
    (run / "resolved_assignments.json").write_text(
        json.dumps(resolved.model_dump(mode="json")), encoding="utf-8"
    )
    (run / "assignment_tree.json").write_text(
        json.dumps(_tree().model_dump(mode="json")), encoding="utf-8"
    )
    return run


def test_offline_v2_rebuild_from_artifacts(tmp_path: Path) -> None:
    run = _fake_v2_run_dir(tmp_path)
    assert detect_report_version(run) == SCHEMA_VERSION_V2
    report = load_report_v2_from_run_dir(run)
    assert report.schema_version == SCHEMA_VERSION_V2
    assert report.merge_threshold == 15  # from the resolved artifact
    nodes = {n.path: n for n in report.iter_nodes()}
    assert nodes["pkg/app/small"].owner == "pkg/app"


def test_offline_dispatch_prefers_existing_report_version(tmp_path: Path) -> None:
    run = _fake_v2_run_dir(tmp_path)
    (run / "tree_decisions.json").write_text(
        json.dumps({"schema_version": "tree_decisions.v2"}), encoding="utf-8"
    )
    assert detect_report_version(run) == SCHEMA_VERSION_V2
    (run / "tree_decisions.json").write_text(
        json.dumps({"schema_version": "tree_decisions.v1"}), encoding="utf-8"
    )
    assert detect_report_version(run) == SCHEMA_VERSION


def test_offline_dispatch_preserves_a_complete_best_effort_report(
    tmp_path: Path,
) -> None:
    run = _fake_v2_run_dir(tmp_path)
    partial_tree = AssignmentTree(
        assignments={"pkg/app": "MODULE", "pkg/app/big": "MODULE"},
        module_decisions=dict(_tree().module_decisions),
    )
    partial = build_tree_decision_report_v2(
        _skeleton(),
        assignments=partial_tree,
        resolved=None,
        metadata=None,
        coverage=compute_assignment_coverage(partial_tree, _skeleton()),
        lints=None,
        issues=None,
        decision=None,
        merge_threshold=15,
    )
    (run / "tree_decisions.json").write_text(
        report_json_text(partial), encoding="utf-8"
    )
    loaded = load_any_report_from_run_dir(run)
    assert loaded.model_dump() == partial.model_dump()


def test_offline_v2_rejects_tampered_resolved_artifact(tmp_path: Path) -> None:
    run = _fake_v2_run_dir(tmp_path)
    resolved = json.loads((run / "resolved_assignments.json").read_text())
    resolved["owners"]["pkg/app/small"] = "pkg/app/big"
    (run / "resolved_assignments.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )
    report = load_report_v2_from_run_dir(run)
    # The tampered artifact is dropped, flagged at the root, and the report is
    # rebuilt from the raw labels (owner recomputed correctly).
    assert any(i.code == "resolved_artifact_mismatch" for i in report.root_issues)
    nodes = {n.path: n for n in report.iter_nodes()}
    assert nodes["pkg/app/small"].owner == "pkg/app"
