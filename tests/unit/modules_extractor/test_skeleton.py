"""Stage 2 (deterministic skeleton) and Stage 1 filesystem-validation tests.

Covers the synthetic-repository shape matrix from the plan's Testing section:
two direct files, one direct file, an ``__init__``-only leaf, a file-plus-child,
a namespace with two single-file children, a single-child passthrough, and a
branch with no independently-required node that must be promoted so its
directories have an owner. Also covers pruning/audit lists, fingerprint
sensitivity, and stable ordering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.coverage import (
    CrossArtifactError,
    validate_source_root_decision,
)
from spotlights_engine.modules_extractor.skeleton import (
    build_skeleton,
    compute_fingerprint,
    scan_source_files,
)
from spotlights_engine.modules_extractor.stage_schemas import SourceRootDecision


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _shape_repo(root: Path) -> None:
    """Build the shape matrix under ``src/``."""
    # two direct files (required owner) with a single-file foldable child leaf
    _write(root / "src" / "twofile" / "a.py")
    _write(root / "src" / "twofile" / "b.py")
    # single-file leaf under a required ancestor -> foldable, NOT promoted
    _write(root / "src" / "twofile" / "onefile" / "only.py")
    # __init__-only leaf (not source-bearing: __init__ does not count as a
    # source file, and there are no children, so the walker prunes it)
    _write(root / "src" / "initonly" / "__init__.py")
    # file plus a child (required: direct + branch)
    _write(root / "src" / "withchild" / "top.py")
    _write(root / "src" / "withchild" / "sub" / "s.py")
    # namespace with two single-file children (required namespace owner)
    _write(root / "src" / "ns" / "left" / "l.py")
    _write(root / "src" / "ns" / "right" / "r.py")
    # single-child passthrough branch that must be promoted
    _write(root / "src" / "passthrough" / "deep" / "leaf.py")


def test_shape_matrix_required_classification(tmp_path: Path) -> None:
    _shape_repo(tmp_path)
    skel = build_skeleton(tmp_path, "src")

    inventory = skel.all_paths()
    required = skel.required_paths()

    # Everything source-bearing is inventoried.
    for expected in (
        "src/twofile",
        "src/twofile/onefile",
        "src/withchild",
        "src/withchild/sub",
        "src/ns",
        "src/ns/left",
        "src/ns/right",
        "src/passthrough",
        "src/passthrough/deep",
    ):
        assert expected in inventory, expected

    # A pure __init__-only leaf carries no source file and no children, so the
    # walker treats it as non-source-bearing and prunes it entirely.
    assert "src/initonly" not in inventory

    # two direct files -> required
    assert "src/twofile" in required
    # file + child -> required
    assert "src/withchild" in required
    # namespace owner of two children -> required
    assert "src/ns" in required
    # single-file leaf beneath a required ancestor -> foldable, not promoted
    assert "src/twofile/onefile" not in required


def test_organizational_only_namespace_recorded(tmp_path: Path) -> None:
    # A directory with only __init__.py plus a single source-bearing child is
    # organizational-only (fewer than two children, no direct source file).
    _write(tmp_path / "src" / "wrapper" / "__init__.py")
    _write(tmp_path / "src" / "wrapper" / "inner" / "a.py")
    _write(tmp_path / "src" / "wrapper" / "inner" / "b.py")
    skel = build_skeleton(tmp_path, "src")
    assert "src/wrapper" in skel.organizational_only


def test_passthrough_branch_is_promoted(tmp_path: Path) -> None:
    _shape_repo(tmp_path)
    skel = build_skeleton(tmp_path, "src")
    required = skel.required_paths()
    # The passthrough branch (src/passthrough/deep/leaf.py) has no naturally
    # required node; the highest node of that branch is promoted so its
    # directories have an emittable owner.
    assert "src/passthrough" in required
    node = next(n for n in skel.iter_nodes() if n.path == "src/passthrough")
    assert "promoted_branch_owner" in node.required_reasons


def test_ignored_dirs_and_symlinks_pruned_and_audited(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    # Ignored dirs
    _write(tmp_path / "src" / "node_modules" / "junk.py")
    _write(tmp_path / "src" / "__pycache__" / "c.py")
    _write(tmp_path / "src" / ".hidden" / "h.py")

    if sys.platform != "win32":
        # directory symlink (loop-ish) and source-file symlink
        (tmp_path / "src" / "linkdir").symlink_to(tmp_path / "src" / "pkg")
        _write(tmp_path / "src" / "real.py")
        (tmp_path / "src" / "linked.py").symlink_to(tmp_path / "src" / "real.py")

    skel = build_skeleton(tmp_path, "src")
    inventory = skel.all_paths()
    assert "src/pkg" in inventory
    # ignored dirs never appear
    assert not any("node_modules" in p for p in inventory)
    assert not any("__pycache__" in p for p in inventory)
    assert not any(".hidden" in p for p in inventory)

    if sys.platform != "win32":
        assert not any("linkdir" in p for p in inventory)
        # symlinks recorded in the dedicated audit list
        assert "src/linkdir" in skel.skipped_symlinks
        assert "src/linked.py" in skel.skipped_symlinks


def test_source_root_validation_rejects_nonexistent_root(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    scan = scan_source_files(tmp_path, "")
    decision = SourceRootDecision.model_validate(
        {
            "repository": {
                "name": "r",
                "summary": "s",
                "source_root": "does_not_exist",
            },
            "excluded_source_paths": [],
        }
    )
    with pytest.raises(CrossArtifactError, match="not an existing"):
        validate_source_root_decision(decision, tmp_path, scan.files)


def test_source_root_validation_unaccounted_sibling_fails_then_exclusion_passes(
    tmp_path: Path,
) -> None:
    # source_root=src, but there's a source-bearing sibling `tests/` outside it.
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    _write(tmp_path / "tests" / "test_a.py")
    scan = scan_source_files(tmp_path, "")

    unaccounted = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src"},
            "excluded_source_paths": [],
        }
    )
    with pytest.raises(CrossArtifactError, match="outside source_root"):
        validate_source_root_decision(unaccounted, tmp_path, scan.files)

    accounted = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src"},
            "excluded_source_paths": [
                {
                    "path": "tests",
                    "reason": "centralized_tests",
                    "explanation": "Top-level test suite.",
                }
            ],
        }
    )
    validate_source_root_decision(accounted, tmp_path, scan.files)


def test_source_root_validation_exclusion_covering_no_source_fails(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    (tmp_path / "emptydir").mkdir()
    scan = scan_source_files(tmp_path, "")
    decision = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src"},
            "excluded_source_paths": [
                {
                    "path": "emptydir",
                    "reason": "not_product_source",
                    "explanation": "Nothing here.",
                }
            ],
        }
    )
    with pytest.raises(CrossArtifactError, match="covers no detected source"):
        validate_source_root_decision(decision, tmp_path, scan.files)


def test_source_root_validation_file_directly_at_nonempty_root_must_be_excluded(
    tmp_path: Path,
) -> None:
    # A source file sitting directly at a non-empty source_root (src/main.py)
    # cannot be a directory module. It is under the root, so the "outside
    # source_root" check does not catch it; it must still be explicitly
    # classified or the decision must fail, otherwise it is silently dropped.
    _write(tmp_path / "src" / "main.py")
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    scan = scan_source_files(tmp_path, "")

    unaccounted = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src"},
            "excluded_source_paths": [],
        }
    )
    with pytest.raises(CrossArtifactError, match="directly at source_root"):
        validate_source_root_decision(unaccounted, tmp_path, scan.files)

    # Explicitly classifying it (repository_level_file) makes it auditable.
    accounted = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src"},
            "excluded_source_paths": [
                {
                    "path": "src/main.py",
                    "reason": "repository_level_file",
                    "explanation": "Entry file at the source root.",
                }
            ],
        }
    )
    validate_source_root_decision(accounted, tmp_path, scan.files)


def test_source_root_validation_exclusion_containing_root_fails(
    tmp_path: Path,
) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py")
    _write(tmp_path / "src" / "pkg" / "b.py")
    scan = scan_source_files(tmp_path, "")
    decision = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "src/pkg"},
            "excluded_source_paths": [
                {
                    "path": "src",
                    "reason": "not_product_source",
                    "explanation": "Contains the root.",
                }
            ],
        }
    )
    with pytest.raises(CrossArtifactError, match="contains the source_root"):
        validate_source_root_decision(decision, tmp_path, scan.files)


def test_source_root_validation_overlapping_exclusions_fail(tmp_path: Path) -> None:
    _write(tmp_path / "pkg" / "a.py")
    _write(tmp_path / "pkg" / "b.py")
    _write(tmp_path / "extra" / "nested" / "c.py")
    _write(tmp_path / "extra" / "nested" / "d.py")
    scan = scan_source_files(tmp_path, "")
    decision = SourceRootDecision.model_validate(
        {
            "repository": {"name": "r", "summary": "s", "source_root": "pkg"},
            "excluded_source_paths": [
                {
                    "path": "extra",
                    "reason": "not_product_source",
                    "explanation": "outer",
                },
                {
                    "path": "extra/nested",
                    "reason": "not_product_source",
                    "explanation": "inner",
                },
            ],
        }
    )
    with pytest.raises(CrossArtifactError, match="antichain"):
        validate_source_root_decision(decision, tmp_path, scan.files)


def test_fingerprint_changes_on_add_delete_and_edit(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "pkg" / "a.py", "a = 1\n")
    _write(tmp_path / "src" / "pkg" / "b.py", "b = 1\n")
    scan1 = scan_source_files(tmp_path, "")
    fp1 = compute_fingerprint(tmp_path, scan1.files, extra_paths=[])

    # Content edit (same size) must change the fingerprint.
    _write(tmp_path / "src" / "pkg" / "a.py", "a = 2\n")
    scan2 = scan_source_files(tmp_path, "")
    fp2 = compute_fingerprint(tmp_path, scan2.files, extra_paths=[])
    assert fp1 != fp2

    # Add a file.
    _write(tmp_path / "src" / "pkg" / "c.py", "c = 1\n")
    scan3 = scan_source_files(tmp_path, "")
    fp3 = compute_fingerprint(tmp_path, scan3.files, extra_paths=[])
    assert fp3 != fp2

    # Delete a file.
    os.remove(tmp_path / "src" / "pkg" / "c.py")
    scan4 = scan_source_files(tmp_path, "")
    fp4 = compute_fingerprint(tmp_path, scan4.files, extra_paths=[])
    assert fp4 == fp2  # back to the two-file content state


def test_scan_and_skeleton_order_is_stable(tmp_path: Path) -> None:
    # Create in a deliberately non-sorted order.
    for name in ("zeta", "alpha", "mike"):
        _write(tmp_path / "src" / name / "a.py")
        _write(tmp_path / "src" / name / "b.py")
    skel1 = build_skeleton(tmp_path, "src")
    order1 = [n.path for n in skel1.iter_nodes()]
    # Rebuild — order must be identical (sorted, not creation-dependent).
    skel2 = build_skeleton(tmp_path, "src")
    order2 = [n.path for n in skel2.iter_nodes()]
    assert order1 == order2
    assert order1 == sorted(order1)
