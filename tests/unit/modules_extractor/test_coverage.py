"""Coverage-equation and cross-artifact validation regression tests.

Mirrors the plan's Testing/"Coverage regression" bullets. A synthetic repo
stands in for the vllm layout: ``pkg/core`` (required, two files) with a
required nested ``pkg/core/kv_offload`` and a single-file foldable
``pkg/core/util``. The coverage equation is ``missing = required - emitted -
folded`` with no ancestor inference and no implicit folds.
"""

from __future__ import annotations

import json
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
    # Try to "cover" kv_offload with a fold whose evidence names a file that
    # does not exist in the folded territory.
    data["folds"].append(
        {
            "path": "pkg/core/kv_offload",
            "into": "pkg/core",
            "reason": "hand-wave",
            "evidence_files": ["pkg/core/kv_offload/nope.py"],
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


def test_fold_clears_missing_with_valid_evidence(tmp_path: Path) -> None:
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    tree = EnrichedTree.model_validate(data)
    # Valid: util folded into core with a real evidence file under it. (This
    # tree also happens to cite the evidence in core.main_files — the
    # compatibility property: everything the old receipt-coupled validator
    # accepted, the Level-1 validator still accepts.)
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


def test_passthrough_fold_evidence_need_not_be_in_main_files(tmp_path: Path) -> None:
    # The vllm `fla/` regression: an organizational passthrough package whose
    # only direct file is `__init__.py`, with its real source in an emitted child
    # `fla/ops/`. Folding the passthrough into an emitted ancestor cites only its
    # `__init__.py` as evidence — nothing substantive of its own exists to place
    # in the target's main_files, so Rule 7's in-main_files requirement is waived.
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "fla" / "__init__.py", "# passthrough\n")
    _write(tmp_path / "pkg" / "core" / "fla" / "ops" / "chunk.py")
    _write(tmp_path / "pkg" / "core" / "fla" / "ops" / "index.py")
    skel = build_skeleton(tmp_path, "pkg")
    assert "pkg/core/fla" in skel.organizational_only

    data = _full_tree()
    # Emit fla/ops directly under core (the passthrough fla is not emitted).
    data["modules"][0]["submodules"].append(
        {
            "name": "ops",
            "path": "pkg/core/fla/ops",
            "description": "Fused linear-attention ops.",
            "main_files": [
                {"path": "pkg/core/fla/ops/chunk.py", "role": "Chunked op."}
            ],
        }
    )
    # Fold the passthrough into core; evidence is the __init__.py, NOT in
    # core.main_files.
    data["folds"].append(
        {
            "path": "pkg/core/fla",
            "into": "pkg/core",
            "reason": "organizational passthrough; real source emitted as ops child",
            "evidence_files": ["pkg/core/fla/__init__.py"],
        }
    )
    tree = EnrichedTree.model_validate(data)
    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert cov.ok
    assert "pkg/core/fla/ops" in cov.emitted
    assert "pkg/core/fla" in cov.folded


def _container_repo(tmp_path: Path) -> Path:
    """The llm-d-router `cmd/` layout: a directory holding only sub-directories."""
    _write(tmp_path / "cmd" / "epp" / "main.go")
    _write(tmp_path / "cmd" / "epp" / "runner" / "run.go")
    _write(tmp_path / "cmd" / "epp" / "runner" / "health.go")
    _write(tmp_path / "cmd" / "pd_sidecar" / "main.go")
    return tmp_path


def _container_tree() -> dict:
    return {
        "modules": [
            {
                "name": "cmd",
                "path": "cmd",
                "description": "Command entrypoints.",
                "main_files": [],
                "submodules": [
                    {
                        "name": "epp",
                        "path": "cmd/epp",
                        "description": "Endpoint-picker binary.",
                        "main_files": [
                            {"path": "cmd/epp/main.go", "role": "Entrypoint."},
                            {"path": "cmd/epp/runner/run.go", "role": "Runner (folded)."},
                        ],
                    },
                    {
                        "name": "pd_sidecar",
                        "path": "cmd/pd_sidecar",
                        "description": "Prefill/decode sidecar binary.",
                        "main_files": [
                            {"path": "cmd/pd_sidecar/main.go", "role": "Entrypoint."}
                        ],
                    },
                ],
            }
        ],
        "folds": [
            {
                "path": "cmd/epp/runner",
                "into": "cmd/epp",
                "reason": "single cohesive run unit",
                "evidence_files": ["cmd/epp/runner/run.go"],
            }
        ],
    }


def test_pure_container_module_may_cite_no_main_files(tmp_path: Path) -> None:
    # The llm-d-router `cmd/` regression. `cmd/` holds no file at all, only
    # `epp/` and `pd-sidecar/`. Once both children are emitted, every candidate
    # main_file belongs to an emitted descendant, so Rule 3 is unsatisfiable and
    # the only escapes were an illegal tree or folding both real binaries away.
    # A pure container may therefore cite nothing.
    _container_repo(tmp_path)
    skel = build_skeleton(tmp_path, "")
    repo = Repository(name="r", summary="s", source_root="")
    tree = EnrichedTree.model_validate(_container_tree())
    validate_enriched_tree(tree, tmp_path, repo, skel)
    cov = compute_coverage(tree, skel)
    assert cov.ok
    assert "cmd/epp" in cov.emitted
    assert "cmd/pd_sidecar" in cov.emitted
    assert "cmd/epp/runner" in cov.folded


def test_container_module_citing_a_childs_file_still_fails(tmp_path: Path) -> None:
    # The exemption is "cite nothing", not "cite a child's file": ownership of a
    # file stays with the deepest emitted module that contains it.
    _container_repo(tmp_path)
    skel = build_skeleton(tmp_path, "")
    repo = Repository(name="r", summary="s", source_root="")
    data = _container_tree()
    data["modules"][0]["main_files"] = [
        {"path": "cmd/epp/main.go", "role": "Entrypoint."}
    ]
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError, match="belongs to emitted descendant"):
        validate_enriched_tree(tree, tmp_path, repo, skel)


def test_module_owning_files_must_still_cite_main_files(tmp_path: Path) -> None:
    # A module whose directory does hold direct files may not cite none —
    # the pydantic bound moved to the cross-artifact validator, it did not go.
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["submodules"][0]["main_files"] = []
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError, match="cites no main_files"):
        validate_enriched_tree(tree, tmp_path, _repository(), skel)


def test_main_file_and_single_child_violations_reported_together(
    tmp_path: Path,
) -> None:
    # Rules 3 and 4 constrain the same choice, so one repair must see both: the
    # llm-d-router failure was a repair told only about Rule 3, which rearranged
    # the emitted set, tripped Rule 4, and had no attempt left to fix it.
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "deep" / "d1.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "deep" / "d2.py")
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    # Rule 3 offender: core cites a file owned by its emitted scheduler child.
    data["modules"][0]["main_files"].append(
        {"path": "pkg/core/scheduler/s2.py", "role": "S2."}
    )
    # Rule 4 offender: kv_offload emits exactly one child.
    data["modules"][0]["submodules"][0]["submodules"] = [
        {
            "name": "deep",
            "path": "pkg/core/kv_offload/deep",
            "description": "Deep offload.",
            "main_files": [{"path": "pkg/core/kv_offload/deep/d1.py", "role": "D1."}],
        }
    ]
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError) as excinfo:
        validate_enriched_tree(tree, tmp_path, _repository(), skel)
    message = str(excinfo.value)
    assert "belongs to emitted descendant" in message
    assert "single child" in message


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


# ── Level-1 receipt-budget regression (committed tiering fixture) ──────────

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "modules_extractor"


def test_six_folds_into_one_module_need_no_receipt_slots() -> None:
    # The vLLM `tiering` shape: one leaf module absorbing six child directories.
    # Under the old receipt rule this needed six evidence entries in the
    # target's five `main_files` slots — structurally impossible. Level 1
    # validates evidence in place, so the module passes with its two genuine
    # main files.
    repo = _FIXTURES / "tiering"
    skel = build_skeleton(repo, "pkg")
    tree = EnrichedTree.model_validate(
        json.loads(
            (_FIXTURES / "tiering_enriched_tree.json").read_text(encoding="utf-8")
        )
    )
    assert len(tree.folds) == 6
    assert len(tree.modules[0].main_files) <= 5

    validate_enriched_tree(
        tree, repo, Repository(name="tiering", summary="s", source_root="pkg"), skel
    )
    cov = compute_coverage(tree, skel)
    assert cov.ok
    assert len(cov.folded) == 6


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


# ── Fold evidence waiver for no-direct-source passthroughs ─────────────────


def test_fold_of_dir_without_direct_source_waives_mainfiles_evidence(
    tmp_path: Path,
) -> None:
    # A Java-style package chain: pkg/mod owns direct files; pkg/mod/java holds
    # no direct file at all, only two source-bearing children. Folding `java`
    # into `mod` cannot put a direct file of `java` in mod's main_files (none
    # exists), so the requirement is waived; the kafka batch runs failed
    # exactly here.
    _write(tmp_path / "pkg" / "mod" / "build.py")
    _write(tmp_path / "pkg" / "mod" / "conf.py")
    _write(tmp_path / "pkg" / "mod" / "java" / "a" / "a1.py")
    _write(tmp_path / "pkg" / "mod" / "java" / "a" / "a2.py")
    _write(tmp_path / "pkg" / "mod" / "java" / "b" / "b1.py")
    _write(tmp_path / "pkg" / "mod" / "java" / "b" / "b2.py")
    skel = build_skeleton(tmp_path, "pkg")

    tree = EnrichedTree.model_validate(
        {
            "modules": [
                {
                    "name": "mod",
                    "path": "pkg/mod",
                    "description": "Module.",
                    "main_files": [{"path": "pkg/mod/build.py", "role": "B."}],
                    "submodules": [
                        {
                            "name": "a",
                            "path": "pkg/mod/java/a",
                            "description": "A.",
                            "main_files": [
                                {"path": "pkg/mod/java/a/a1.py", "role": "A1."}
                            ],
                        },
                        {
                            "name": "b",
                            "path": "pkg/mod/java/b",
                            "description": "B.",
                            "main_files": [
                                {"path": "pkg/mod/java/b/b1.py", "role": "B1."}
                            ],
                        },
                    ],
                }
            ],
            "folds": [
                {
                    "path": "pkg/mod/java",
                    "into": "pkg/mod",
                    "reason": "package chain passthrough",
                    "evidence_files": ["pkg/mod/java/a/a1.py"],
                }
            ],
        }
    )

    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert cov.missing == []
    assert "pkg/mod/java" in cov.folded


def test_fold_evidence_need_not_be_in_target_main_files(tmp_path: Path) -> None:
    # Level 1: the fold's evidence file is validated against the filesystem in
    # place — removing it from the target's main_files no longer invalidates
    # the fold. (Under the old receipt rule this exact tree was rejected.)
    _repo(tmp_path)
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["modules"][0]["main_files"] = [
        {"path": "pkg/core/engine.py", "role": "Engine."}
    ]
    tree = EnrichedTree.model_validate(data)
    validate_enriched_tree(tree, tmp_path, _repository(), skel)
    cov = compute_coverage(tree, skel)
    assert cov.ok
    assert "pkg/core/util" in cov.folded


def test_fold_evidence_owned_by_emitted_descendant_is_invalid(
    tmp_path: Path,
) -> None:
    # Evidence must prove inspection of the *folded* territory. A file that the
    # nearest-owner rule assigns to a separately emitted descendant module
    # cannot serve: it belongs to that module, not to the folded remainder.
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "extra" / "x.py")
    _write(tmp_path / "pkg" / "core" / "extra" / "deep" / "d1.py")
    _write(tmp_path / "pkg" / "core" / "extra" / "deep" / "d2.py")
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    # Emit extra/deep as a module, then fold extra citing only deep's file.
    data["modules"][0]["submodules"].append(
        {
            "name": "deep",
            "path": "pkg/core/extra/deep",
            "description": "Deep.",
            "main_files": [{"path": "pkg/core/extra/deep/d1.py", "role": "D1."}],
        }
    )
    data["folds"].append(
        {
            "path": "pkg/core/extra",
            "into": "pkg/core",
            "reason": "thin wrapper",
            "evidence_files": ["pkg/core/extra/deep/d2.py"],
        }
    )
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError, match="no valid evidence file"):
        validate_enriched_tree(tree, tmp_path, _repository(), skel)


def test_symlink_fold_evidence_is_invalid(tmp_path: Path) -> None:
    _repo(tmp_path)
    _write(tmp_path / "pkg" / "core" / "extra" / "real_helper.py")
    (tmp_path / "pkg" / "core" / "extra" / "link.py").symlink_to(
        tmp_path / "pkg" / "core" / "engine.py"
    )
    skel = build_skeleton(tmp_path, "pkg")
    data = _full_tree()
    data["folds"].append(
        {
            "path": "pkg/core/extra",
            "into": "pkg/core",
            "reason": "helper",
            "evidence_files": ["pkg/core/extra/link.py"],
        }
    )
    tree = EnrichedTree.model_validate(data)
    with pytest.raises(CrossArtifactError, match="no valid evidence file"):
        validate_enriched_tree(tree, tmp_path, _repository(), skel)
