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
    forced_repository_level_files,
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


def test_source_less_module_may_cite_nonsource_main_files(tmp_path: Path) -> None:
    # The vllm `docker/` regression: a directory with zero direct
    # source-extension files (only Dockerfiles + `.hcl` + `.json`) whose only
    # real source lives in a child `entrypoints/`. It has nothing source-typed
    # to cite, so it may cite its own real non-symlink files of any extension as
    # main_files. The source child is folded, and one of its `.sh` files is the
    # fold evidence in the module's main_files.
    _write(tmp_path / "docker" / "Dockerfile", "FROM scratch\n")
    _write(tmp_path / "docker" / "docker-bake.hcl", 'target "x" {}\n')
    _write(tmp_path / "docker" / "entrypoints" / "run.sh", "echo hi\n")
    _write(tmp_path / "docker" / "entrypoints" / "test_run.sh", "echo test\n")
    skel = build_skeleton(tmp_path, "")
    repo = Repository(name="r", summary="s", source_root="")
    data = {
        "modules": [
            {
                "name": "docker",
                "path": "docker",
                "description": "Container build definitions and entrypoints.",
                "depends_on": [],
                "main_files": [
                    {"path": "docker/Dockerfile", "role": "Primary build image."},
                    {"path": "docker/docker-bake.hcl", "role": "buildx bake targets."},
                    {"path": "docker/entrypoints/run.sh", "role": "Entrypoint (folded)."},
                ],
                "submodules": [],
            }
        ],
        "folds": [
            {
                "path": "docker/entrypoints",
                "into": "docker",
                "reason": "single cohesive entrypoint unit",
                "evidence_files": ["docker/entrypoints/run.sh"],
            }
        ],
    }
    tree = EnrichedTree.model_validate(data)
    validate_enriched_tree(tree, tmp_path, repo, skel)
    cov = compute_coverage(tree, skel)
    assert cov.ok
    assert "docker" in cov.emitted
    assert "docker/entrypoints" in cov.folded


def test_source_bearing_module_keeps_strict_main_file_gate(tmp_path: Path) -> None:
    # A module directory that DOES hold a direct source file keeps the strict
    # source-extension gate: a Dockerfile sitting next to real source cannot be
    # cited as a main_file.
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "Dockerfile", "FROM scratch\n")
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["main_files"] = [
        {"path": "pkg/core/Dockerfile", "role": "Build image."},
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


def test_all_single_child_parents_reported_together(tmp_path: Path) -> None:
    # Two clustered single-child parents in one tree. Rule 4 must name BOTH in a
    # single error so the one bounded repair pass can collapse them together;
    # reporting one-at-a-time exhausts the repair budget (the vllm quantization
    # regression: transform/inc/quark each wrapped a lone schemes child).
    _write(tmp_path / "pkg" / "core" / "engine.py")
    _write(tmp_path / "pkg" / "core" / "runner.py")
    _write(tmp_path / "pkg" / "core" / "left" / "l.py")
    _write(tmp_path / "pkg" / "core" / "left" / "only" / "a.py")
    _write(tmp_path / "pkg" / "core" / "right" / "r.py")
    _write(tmp_path / "pkg" / "core" / "right" / "solo" / "b.py")
    skel = build_skeleton(tmp_path, "pkg")
    data = {
        "modules": [
            {
                "name": "core",
                "path": "pkg/core",
                "description": "Core runtime.",
                "depends_on": [],
                "main_files": [{"path": "pkg/core/engine.py", "role": "Engine."}],
                "submodules": [
                    {
                        "name": "left",
                        "path": "pkg/core/left",
                        "description": "Left branch.",
                        "main_files": [{"path": "pkg/core/left/l.py", "role": "L."}],
                        "submodules": [
                            {
                                "name": "only",
                                "path": "pkg/core/left/only",
                                "description": "Only child.",
                                "main_files": [
                                    {"path": "pkg/core/left/only/a.py", "role": "A."}
                                ],
                            }
                        ],
                    },
                    {
                        "name": "right",
                        "path": "pkg/core/right",
                        "description": "Right branch.",
                        "main_files": [{"path": "pkg/core/right/r.py", "role": "R."}],
                        "submodules": [
                            {
                                "name": "solo",
                                "path": "pkg/core/right/solo",
                                "description": "Solo child.",
                                "main_files": [
                                    {"path": "pkg/core/right/solo/b.py", "role": "B."}
                                ],
                            }
                        ],
                    },
                ],
            }
        ],
        "folds": [],
    }
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError) as excinfo:
        validate_enriched_tree(tree, tmp_path, _repository(), skel)
    message = str(excinfo.value)
    assert "pkg/core/left" in message
    assert "pkg/core/right" in message


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


# ── forced_repository_level_files ──────────────────────────────────────────


def test_forced_files_at_repo_root() -> None:
    # source_root == "": only files directly at the repo root are forced; the
    # packaged tree under vllm/ is untouched. This is the vllm regression: a
    # stray root-level script (build_vllm_ppc64le.sh) that the model may miss.
    files = [
        "build_rust.sh",
        "build_vllm_ppc64le.sh",
        "setup.py",
        "use_existing_torch.py",
        "vllm/__init__.py",
        "vllm/core/engine.py",
    ]
    assert forced_repository_level_files("", files) == [
        "build_rust.sh",
        "build_vllm_ppc64le.sh",
        "setup.py",
        "use_existing_torch.py",
    ]


def test_forced_files_at_nonempty_root() -> None:
    # source_root == "src": only files directly at src/ are forced; nested
    # package files and files outside the root are not.
    files = ["src/pkg/__init__.py", "src/pkg/a.py", "src/top.py", "tools/x.py"]
    assert forced_repository_level_files("src", files) == ["src/top.py"]


def test_forced_files_none_when_all_nested() -> None:
    assert forced_repository_level_files("", ["vllm/core/x.py"]) == []
    assert forced_repository_level_files("src", ["src/pkg/a.py"]) == []
