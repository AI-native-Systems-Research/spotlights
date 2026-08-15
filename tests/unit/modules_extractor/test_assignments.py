"""Assignment-contract validation: preflight, V1–V4, and v2 coverage.

Filesystem-backed tests build small synthetic repos (as `test_coverage` does);
pure label-policy tests build `Skeleton` models directly, since V2/V3 never
touch the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from spotlights_engine.modules_extractor.assignments import (
    compute_assignment_coverage,
    compute_collision_precedence,
    compute_metadata_coverage,
    validate_assignment_labels,
    validate_assignment_paths,
    validate_metadata_entries,
)
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ModuleDecision,
    ModuleInfo,
    ModuleMetadataBatch,
    ResolvedAssignmentTree,
    Skeleton,
    SkeletonNode,
)


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


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


def _skeleton(*nodes: SkeletonNode, source_root: str = "pkg") -> Skeleton:
    return Skeleton(
        source_root=source_root,
        nodes=list(nodes),
        inventory_fingerprint="fp",
    )


def _tree(
    assignments: dict[str, str], decisions: dict[str, str | None] | None = None
) -> AssignmentTree:
    return AssignmentTree(
        assignments=assignments,
        module_decisions={
            p: ModuleDecision(keep_reason=r) for p, r in (decisions or {}).items()
        },
    )


def _auto_decisions(assignments: dict[str, str]) -> dict[str, str | None]:
    return {p: None for p, label in assignments.items() if label == "MODULE"}


# ── Schema: key normalization is collision-safe ───────────────────────────


def test_key_normalization_rejects_colliding_raw_keys() -> None:
    with pytest.raises(ValidationError, match="normalize to"):
        AssignmentTree.model_validate(
            {
                "assignments": {"a/b": "MODULE", "a//b/": "PART"},
                "module_decisions": {},
            }
        )


def test_keys_are_normalized_on_parse() -> None:
    tree = AssignmentTree.model_validate(
        {
            "assignments": {"./a/b/": "MODULE"},
            "module_decisions": {"a//b": {"keep_reason": "independent unit"}},
        }
    )
    assert set(tree.assignments) == {"a/b"}
    assert set(tree.module_decisions) == {"a/b"}


def test_directory_literally_named_module_is_unambiguous() -> None:
    # Values are enum labels, never paths, so a real directory named `MODULE`
    # parses cleanly as a key.
    tree = AssignmentTree.model_validate(
        {
            "assignments": {"pkg/MODULE": "PART", "pkg": "MODULE"},
            "module_decisions": {"pkg": {"keep_reason": None}},
        }
    )
    assert tree.assignments["pkg/MODULE"] == "PART"


def test_keep_reason_is_stripped_and_non_empty() -> None:
    with pytest.raises(ValidationError):
        ModuleDecision(keep_reason="   ")
    assert ModuleDecision(keep_reason="  real reason  ").keep_reason == "real reason"


def test_resolved_tree_requires_key_agreement() -> None:
    with pytest.raises(ValidationError, match="owners keys"):
        ResolvedAssignmentTree(
            assignments={"pkg/a": "MODULE"},
            module_decisions={"pkg/a": ModuleDecision(keep_reason=None)},
            owners={},
            origins={"pkg/a": "top_level_anchor"},
            merge_threshold=15,
        )
    with pytest.raises(ValidationError, match="module_decisions keys"):
        ResolvedAssignmentTree(
            assignments={"pkg/a": "MODULE"},
            module_decisions={},
            owners={"pkg/a": "pkg/a"},
            origins={"pkg/a": "top_level_anchor"},
            merge_threshold=15,
        )


# ── Collision preflight ───────────────────────────────────────────────────


def test_nested_collision_class_picks_largest_subtree_winner() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/foo-bar", direct=8),
                _node("pkg/app/foo_bar", direct=3),
            ),
        )
    )
    precedence = compute_collision_precedence(skel)
    assert precedence.forced_part == {"pkg/app/foo_bar": "pkg/app/foo-bar"}
    assert not precedence.fatal


def test_collision_tie_breaks_lexically() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/foo-bar", direct=3),
                _node("pkg/app/foo_bar", direct=3),
            ),
        )
    )
    precedence = compute_collision_precedence(skel)
    # Equal subtree counts: lexically smallest path wins ("-" < "_").
    assert precedence.forced_part == {"pkg/app/foo_bar": "pkg/app/foo-bar"}


def test_top_level_collision_is_fatal() -> None:
    skel = _skeleton(
        _node("top-x", direct=3),
        _node("top_x", direct=5),
        source_root="",
    )
    precedence = compute_collision_precedence(skel)
    assert precedence.fatal
    assert precedence.top_level_collisions == [["top-x", "top_x"]]
    assert precedence.forced_part == {}


# ── V1 — real inventoried paths ───────────────────────────────────────────


def test_v1_rejects_uninventoried_and_unreal_paths(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "app" / "a.py")
    _write(tmp_path / "pkg" / "app" / "b.py")
    skel = build_skeleton(tmp_path, "pkg")
    assignments = {"pkg/app": "MODULE", "pkg/ghost": "PART"}
    tree = _tree(assignments, {"pkg/app": None, "pkg/ghost": None})
    issues = validate_assignment_paths(tree, skel, tmp_path)
    codes = {(i.code, i.path) for i in issues}
    assert ("assignment_path_not_inventoried", "pkg/ghost") in codes
    assert ("decision_path_not_inventoried", "pkg/ghost") in codes
    assert all(i.path != "pkg/app" for i in issues)


def test_v1_rechecks_real_directory_for_decision_only_key(tmp_path: Path) -> None:
    skel = _skeleton(_node("pkg/gone"))
    tree = _tree({}, {"pkg/gone": None})
    issues = validate_assignment_paths(tree, skel, tmp_path)
    assert ("decision_path_not_a_directory", "pkg/gone") in {
        (issue.code, issue.path) for issue in issues
    }


# ── V2 — totality coverage ────────────────────────────────────────────────


def test_v2_coverage_equations() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/x", direct=2),
                _node("pkg/app/y", direct=1, required=False),
            ),
        )
    )
    tree = _tree(
        {"pkg/app": "MODULE", "pkg/app/x": "PART", "pkg/oops": "PART"},
        {"pkg/app": None},
    )
    cov = compute_assignment_coverage(tree, skel)
    assert cov.schema_version == "coverage.v2"
    assert cov.missing == ["pkg/app/y"]
    assert cov.extra == ["pkg/oops"]
    assert cov.assigned == ["pkg/app", "pkg/app/x"]
    assert cov.modules == ["pkg/app"]
    assert cov.parts == ["pkg/app/x"]
    assert cov.required_missing == []
    assert cov.optional_inventory == ["pkg/app/y"]
    assert not cov.ok


# ── V3 — label policy ─────────────────────────────────────────────────────


def _threshold_skeleton() -> Skeleton:
    return _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/big", direct=16),
                _node("pkg/app/mid", direct=15),
                _node("pkg/app/small", direct=14),
            ),
        )
    )


def _v3(tree: AssignmentTree, skel: Skeleton, **over):
    kwargs = dict(
        merge_threshold=15,
        precedence=compute_collision_precedence(skel),
        anchor_paths={n.path for n in skel.nodes},
        require_owner_in_scope=True,
    )
    kwargs.update(over)
    return validate_assignment_labels(tree, skel, **kwargs)


def test_threshold_boundary_16_must_be_module() -> None:
    skel = _threshold_skeleton()
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/big": "PART",
        "pkg/app/mid": "PART",
        "pkg/app/small": "PART",
    }
    issues = _v3(_tree(assignments, _auto_decisions(assignments)), skel)
    assert [(i.code, i.path) for i in issues] == [
        ("large_path_not_module", "pkg/app/big")
    ]


def test_threshold_boundary_15_and_14_may_be_part() -> None:
    skel = _threshold_skeleton()
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/big": "MODULE",
        "pkg/app/mid": "PART",
        "pkg/app/small": "PART",
    }
    issues = _v3(_tree(assignments, _auto_decisions(assignments)), skel)
    assert issues == []


def test_small_module_requires_keep_reason() -> None:
    skel = _threshold_skeleton()
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/big": "MODULE",
        "pkg/app/mid": "MODULE",
        "pkg/app/small": "PART",
    }
    # mid (15 <= threshold) has no keep_reason -> rejected.
    issues = _v3(_tree(assignments, _auto_decisions(assignments)), skel)
    assert [(i.code, i.path) for i in issues] == [
        ("keep_reason_missing", "pkg/app/mid")
    ]
    # With a reason it passes.
    decisions = _auto_decisions(assignments)
    decisions["pkg/app/mid"] = "Self-contained backend against the tier interface"
    assert _v3(_tree(assignments, decisions), skel) == []


def test_top_level_anchor_must_be_module_and_needs_no_reason() -> None:
    skel = _skeleton(_node("pkg/app", direct=1))  # tiny anchor, below threshold
    issues = _v3(_tree({"pkg/app": "PART"}, {}), skel)
    assert [(i.code, i.path) for i in issues] == [
        ("top_level_not_module", "pkg/app")
    ]
    assert _v3(_tree({"pkg/app": "MODULE"}, {"pkg/app": None}), skel) == []


def test_collision_loser_may_stay_part_even_when_large() -> None:
    skel = _skeleton(
        _node(
            "pkg/app",
            direct=2,
            children=(
                _node("pkg/app/foo-bar", direct=20),
                _node("pkg/app/foo_bar", direct=16),
            ),
        )
    )
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/foo-bar": "MODULE",
        "pkg/app/foo_bar": "PART",  # 16 > threshold, but forced by collision
    }
    issues = _v3(_tree(assignments, _auto_decisions(assignments)), skel)
    assert issues == []
    # The loser labeled MODULE is rejected with the collision code.
    assignments["pkg/app/foo_bar"] = "MODULE"
    issues = _v3(_tree(assignments, _auto_decisions(assignments)), skel)
    assert ("forced_part_violated", "pkg/app/foo_bar") in {
        (i.code, i.path) for i in issues
    }


def test_decision_bijection_is_exact() -> None:
    skel = _threshold_skeleton()
    assignments = {
        "pkg/app": "MODULE",
        "pkg/app/big": "MODULE",
        "pkg/app/mid": "PART",
        "pkg/app/small": "PART",
    }
    # Missing decision for big; orphan decision for small.
    tree = _tree(assignments, {"pkg/app": None, "pkg/app/small": "orphan"})
    codes = {(i.code, i.path) for i in _v3(tree, skel)}
    assert ("decision_missing", "pkg/app/big") in codes
    assert ("decision_orphaned", "pkg/app/small") in codes


def test_part_needs_module_ancestor_globally_but_not_in_nested_shard() -> None:
    skel = _skeleton(
        _node("pkg/app/sub", direct=2, children=(_node("pkg/app/sub/leaf", direct=1),))
    )
    assignments = {"pkg/app/sub": "PART", "pkg/app/sub/leaf": "PART"}
    tree = _tree(assignments, {})
    # Global view: no module anywhere above -> rejected.
    global_issues = validate_assignment_labels(
        tree,
        skel,
        merge_threshold=15,
        precedence=compute_collision_precedence(skel),
        anchor_paths=set(),
        require_owner_in_scope=True,
    )
    assert {i.code for i in global_issues} == {"part_without_module_ancestor"}
    # Nested-shard view: provisionally legal; the union resolves the owner.
    shard_issues = validate_assignment_labels(
        tree,
        skel,
        merge_threshold=15,
        precedence=compute_collision_precedence(skel),
        anchor_paths=set(),
        require_owner_in_scope=False,
    )
    assert shard_issues == []


# ── V4 — metadata keys and main files ─────────────────────────────────────


def _metadata_repo(tmp_path: Path) -> Path:
    _write(tmp_path / "pkg" / "app" / "engine.py")
    _write(tmp_path / "pkg" / "app" / "runner.py")
    _write(tmp_path / "pkg" / "app" / "inner" / "i1.py")
    _write(tmp_path / "pkg" / "app" / "inner" / "i2.py")
    return tmp_path


def test_v4_exact_keys_and_nearest_owner(tmp_path: Path) -> None:
    _metadata_repo(tmp_path)
    module_paths = {"pkg/app", "pkg/app/inner"}
    requested = {"pkg/app"}
    batch = ModuleMetadataBatch.model_validate(
        {
            "modules": {
                "pkg/app": {
                    "description": "App runtime.",
                    "main_files": [
                        {"path": "pkg/app/engine.py", "role": "Engine."},
                        # Owned by the nested module -> rejected.
                        {"path": "pkg/app/inner/i1.py", "role": "Inner."},
                    ],
                },
                # Not requested in this batch.
                "pkg/app/inner": {
                    "description": "Inner.",
                    "main_files": [{"path": "pkg/app/inner/i1.py", "role": "I."}],
                },
            }
        }
    )
    issues = validate_metadata_entries(
        batch.modules, requested, module_paths, tmp_path
    )
    codes = {(i.code, i.path) for i in issues}
    assert ("metadata_key_not_requested", "pkg/app/inner") in codes
    assert ("main_file_not_owned", "pkg/app") in codes

    coverage = compute_metadata_coverage(batch.modules, requested)
    assert coverage.extra == ["pkg/app/inner"]
    assert coverage.missing == []
    assert coverage.ok


def test_v4_missing_requested_keys_are_coverage_not_validation() -> None:
    coverage = compute_metadata_coverage({"pkg/app"}, {"pkg/app", "pkg/lib"})
    assert coverage.missing == ["pkg/lib"]
    assert not coverage.ok


def test_v4_double_claim_across_batches(tmp_path: Path) -> None:
    _metadata_repo(tmp_path)
    module_paths = {"pkg/app", "pkg/app/inner"}
    claimed: dict[str, str] = {}
    first = {
        "pkg/app": ModuleInfo.model_validate(
            {
                "description": "App.",
                "main_files": [{"path": "pkg/app/engine.py", "role": "E."}],
            }
        )
    }
    assert (
        validate_metadata_entries(
            first, {"pkg/app"}, module_paths, tmp_path, claimed=claimed
        )
        == []
    )
    # A second batch claiming the same file for another module is caught by
    # the threaded `claimed` map (defense in depth; disjoint territories make
    # it impossible for accepted batches).
    second = {
        "pkg/app/inner": ModuleInfo.model_validate(
            {
                "description": "Inner.",
                "main_files": [{"path": "pkg/app/engine.py", "role": "E."}],
            }
        )
    }
    issues = validate_metadata_entries(
        second, {"pkg/app/inner"}, module_paths, tmp_path, claimed=claimed
    )
    # Outside the module dir entirely -> outside_module fires first.
    assert issues and issues[0].code == "main_file_outside_module"


def test_v4_pure_container_and_nonsource_carveouts(tmp_path: Path) -> None:
    # cmd/ holds only sub-directories (pure container: zero main files legal).
    _write(tmp_path / "cmd" / "epp" / "main.go")
    _write(tmp_path / "cmd" / "epp" / "run.go")
    # docker/ holds no source-extension file, but real files of other kinds.
    _write(tmp_path / "docker" / "Dockerfile", "FROM scratch\n")
    _write(tmp_path / "docker" / "bake.hcl", 'target "x" {}\n')
    _write(tmp_path / "docker" / "entry" / "run.sh", "echo hi\n")
    skel = build_skeleton(tmp_path, "")
    assert skel  # both trees walked

    module_paths = {"cmd", "cmd/epp", "docker"}
    modules = {
        "cmd": ModuleInfo.model_validate(
            {"description": "Container of binaries.", "main_files": []}
        ),
        "cmd/epp": ModuleInfo.model_validate(
            {
                "description": "Binary.",
                "main_files": [{"path": "cmd/epp/main.go", "role": "Entry."}],
            }
        ),
        "docker": ModuleInfo.model_validate(
            {
                "description": "Container builds.",
                "main_files": [
                    {"path": "docker/Dockerfile", "role": "Primary image."}
                ],
            }
        ),
    }
    issues = validate_metadata_entries(
        modules, set(module_paths), module_paths, tmp_path
    )
    assert issues == []

    # A source-bearing module citing nothing is rejected.
    modules["cmd/epp"] = ModuleInfo.model_validate(
        {"description": "Binary.", "main_files": []}
    )
    issues = validate_metadata_entries(
        modules, set(module_paths), module_paths, tmp_path
    )
    assert {(i.code, i.path) for i in issues} == {
        ("main_files_required", "cmd/epp")
    }


def test_v4_source_bearing_module_accepts_any_real_file(
    tmp_path: Path,
) -> None:
    # No extension gate: a README next to source files is a valid citation.
    _write(tmp_path / "pkg" / "app" / "engine.py")
    _write(tmp_path / "pkg" / "app" / "README.md", "# hi\n")
    module_paths = {"pkg/app"}
    modules = {
        "pkg/app": ModuleInfo.model_validate(
            {
                "description": "App.",
                "main_files": [{"path": "pkg/app/README.md", "role": "Docs."}],
            }
        )
    }
    issues = validate_metadata_entries(
        modules, module_paths, module_paths, tmp_path
    )
    assert issues == []

    # Files must still exist for real.
    modules["pkg/app"] = ModuleInfo.model_validate(
        {
            "description": "App.",
            "main_files": [{"path": "pkg/app/missing.py", "role": "Ghost."}],
        }
    )
    issues = validate_metadata_entries(
        modules, module_paths, module_paths, tmp_path
    )
    assert issues and issues[0].code == "main_file_not_a_file"
