"""Coverage-equation and cross-artifact validation regression tests.

Mirrors the plan's Testing/"Coverage regression" bullets. A synthetic repo
stands in for the vllm layout: ``pkg/core`` (required, two files) with a
required nested ``pkg/core/kv_offload`` and a single-file foldable
``pkg/core/util``. The coverage equation is ``missing = required - emitted -
folded`` with no ancestor inference and no implicit folds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.coverage import (
    CrossArtifactError,
    compute_coverage,
    validate_enriched_tree,
)
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import EnrichedTree
from spotlights_engine.schemas.project import Repository


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    # pkg/core: two direct files + two required nested dirs (kv_offload,
    # scheduler; a parent needs zero or >=2 children) + a foldable util leaf.
    _write(tmp_path / "pkg" / "core" / "engine.py")
    _write(tmp_path / "pkg" / "core" / "runner.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "a.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "b.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s1.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s2.py")
    _write(tmp_path / "pkg" / "core" / "util" / "helper.py")
    return tmp_path


def _repository() -> Repository:
    return Repository(name="r", summary="s", source_root="pkg")


def _full_tree() -> dict:
    return {
        "modules": [
            {
                "name": "core",
                "path": "pkg/core",
                "description": "Core runtime.",
                "depends_on": [],
                "main_files": [
                    {"path": "pkg/core/engine.py", "role": "Engine."},
                    {"path": "pkg/core/util/helper.py", "role": "Helper (folded)."},
                ],
                "submodules": [
                    {
                        "name": "kv_offload",
                        "path": "pkg/core/kv_offload",
                        "description": "KV offloading.",
                        "main_files": [
                            {"path": "pkg/core/kv_offload/a.py", "role": "A."}
                        ],
                    },
                    {
                        "name": "scheduler",
                        "path": "pkg/core/scheduler",
                        "description": "Scheduling.",
                        "main_files": [
                            {"path": "pkg/core/scheduler/s1.py", "role": "S1."}
                        ],
                    },
                ],
            }
        ],
        "folds": [
            {
                "path": "pkg/core/util",
                "into": "pkg/core",
                "reason": "single-file helper",
                "evidence_files": ["pkg/core/util/helper.py"],
            }
        ],
    }


def test_full_valid_tree_has_no_missing(tmp_path: Path) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    tree = EnrichedTree.model_validate(_full_tree())
    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert cov.missing == []
    assert cov.ok
    assert "pkg/core/kv_offload" in cov.emitted
    assert "pkg/core/util" in cov.folded


def test_deleting_required_node_from_tree_is_missing(tmp_path: Path) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    # Drop the emitted kv_offload submodule entirely.
    data["modules"][0]["submodules"] = []
    tree = EnrichedTree.model_validate(data)
    cov = compute_coverage(tree, skel)
    assert "pkg/core/kv_offload" in cov.missing


def test_unverified_fold_string_does_not_clear_missing(tmp_path: Path) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["submodules"] = []
    # Try to "cover" kv_offload with a fold whose evidence is not in main_files
    # and whose target does not list it.
    data["folds"].append(
        {
            "path": "pkg/core/kv_offload",
            "into": "pkg/core",
            "reason": "hand-wave",
            "evidence_files": ["pkg/core/kv_offload/a.py"],
        }
    )
    tree = EnrichedTree.model_validate(data)
    # compute_coverage sees the fold string and would clear missing, but
    # validate_enriched_tree must reject the invalid fold first.
    with pytest.raises(CrossArtifactError, match="no valid evidence file"):
        validate_enriched_tree(tree, tmp_path, _repository(), skel)


def test_emitting_only_descendant_does_not_cover_ancestor(tmp_path: Path) -> None:
    # pkg/core is required; emit only the descendant kv_offload, not core.
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = {
        "modules": [
            {
                "name": "kv_offload",
                "path": "pkg/core/kv_offload",
                "description": "KV offloading only.",
                "depends_on": [],
                "main_files": [
                    {"path": "pkg/core/kv_offload/a.py", "role": "A."}
                ],
            }
        ],
        "folds": [],
    }
    tree = EnrichedTree.model_validate(data)
    cov = compute_coverage(tree, skel)
    # The required ancestor pkg/core is not covered by emitting its descendant.
    assert "pkg/core" in cov.missing


def test_fold_clears_missing_only_with_valid_evidence_in_main_files(
    tmp_path: Path,
) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    tree = EnrichedTree.model_validate(data)
    # Valid: util folded into core, evidence file present in core.main_files.
    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert "pkg/core/util" not in cov.missing
    assert "pkg/core/util" in cov.folded


def test_readme_or_nonsource_main_file_fails(tmp_path: Path) -> None:
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "README.md", "# hi\n")
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["main_files"] = [
        {"path": "pkg/core/README.md", "role": "Docs."},
        {"path": "pkg/core/engine.py", "role": "Engine."},
    ]
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError, match="not a real non-symlink source file"):
        validate_enriched_tree(tree, tmp_path, _repository(), skel)


def test_external_dependency_colliding_with_internal_qn_fails(tmp_path: Path) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    repo = Repository(
        name="r", summary="s", source_root="pkg", external_dependencies=["core"]
    )
    tree = EnrichedTree.model_validate(_full_tree())
    with pytest.raises(CrossArtifactError, match="collides"):
        validate_enriched_tree(tree, tmp_path, repo, skel)


def test_optional_fold_accepted_and_reported(tmp_path: Path) -> None:
    # Add an optional (non-required) single-file dir under core; fold it.
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "extra" / "x.py")
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["main_files"].append(
        {"path": "pkg/core/extra/x.py", "role": "Extra (folded)."}
    )
    data["folds"].append(
        {
            "path": "pkg/core/extra",
            "into": "pkg/core",
            "reason": "optional single-file dir",
            "evidence_files": ["pkg/core/extra/x.py"],
        }
    )
    tree = EnrichedTree.model_validate(data)
    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert "pkg/core/extra" in cov.optional_folded
    # It was optional, so folding it does not change the required set.
    assert "pkg/core/extra" not in cov.required
