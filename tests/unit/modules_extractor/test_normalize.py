"""Deterministic normalization tests: single-child collapse.

The synthetic repo mirrors the observed vllm failure shape: ``pkg/worker`` with
direct files and exactly one child directory ``gpu``, which itself has two
substantial children. Attempt 1 of the real run emitted that chain and was
rejected only on Rule 4; the normalizer must accept it mechanically — collapse
``gpu`` into ``worker``, keep every required path covered, and leave the tree
valid under the unchanged ``validate_enriched_tree``.

(The old fold-evidence sync was deleted with the Level-1 receipt decoupling:
Rule 8 validates ``evidence_files`` directly against the filesystem, so there
is no `main_files` cross-reference left to repair.)
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.modules_extractor.coverage import (
    compute_coverage,
    validate_enriched_tree,
)
from spotlights_engine.modules_extractor.normalize import normalize_enriched_tree
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import EnrichedTree
from spotlights_engine.schemas.project import Repository


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _repository() -> Repository:
    return Repository(name="r", summary="s", source_root="pkg")


def _worker_repo(tmp_path: Path) -> Path:
    # pkg/worker: 2 direct files, single child gpu; gpu: 1 direct file and two
    # substantial children (mm, sample). worker/gpu is the Rule-4 offender when
    # gpu is emitted as worker's only child.
    _write(tmp_path / "pkg" / "worker" / "base.py")
    _write(tmp_path / "pkg" / "worker" / "utils.py")
    _write(tmp_path / "pkg" / "worker" / "gpu" / "gpu_worker.py")
    _write(tmp_path / "pkg" / "worker" / "gpu" / "mm" / "m1.py")
    _write(tmp_path / "pkg" / "worker" / "gpu" / "mm" / "m2.py")
    _write(tmp_path / "pkg" / "worker" / "gpu" / "sample" / "s1.py")
    _write(tmp_path / "pkg" / "worker" / "gpu" / "sample" / "s2.py")
    # A sibling so pkg/worker isn't the sole top-level branch under pkg.
    _write(tmp_path / "pkg" / "other" / "o1.py")
    _write(tmp_path / "pkg" / "other" / "o2.py")
    return tmp_path


def _single_child_tree() -> EnrichedTree:
    """worker -> gpu (single child) -> {mm, sample}: coverage-perfect, Rule-4
    invalid — the exact shape the vllm run's attempt 1 produced."""
    return EnrichedTree.model_validate(
        {
            "modules": [
                {
                    "name": "worker",
                    "path": "pkg/worker",
                    "description": "Worker runtime.",
                    "main_files": [
                        {"path": "pkg/worker/base.py", "role": "Base."},
                    ],
                    "submodules": [
                        {
                            "name": "gpu",
                            "path": "pkg/worker/gpu",
                            "description": "GPU worker.",
                            "main_files": [
                                {
                                    "path": "pkg/worker/gpu/gpu_worker.py",
                                    "role": "GPU worker.",
                                }
                            ],
                            "submodules": [
                                {
                                    "name": "mm",
                                    "path": "pkg/worker/gpu/mm",
                                    "description": "Multimodal.",
                                    "main_files": [
                                        {
                                            "path": "pkg/worker/gpu/mm/m1.py",
                                            "role": "M1.",
                                        }
                                    ],
                                },
                                {
                                    "name": "sample",
                                    "path": "pkg/worker/gpu/sample",
                                    "description": "Sampling.",
                                    "main_files": [
                                        {
                                            "path": "pkg/worker/gpu/sample/s1.py",
                                            "role": "S1.",
                                        }
                                    ],
                                },
                            ],
                        }
                    ],
                },
                {
                    "name": "other",
                    "path": "pkg/other",
                    "description": "Other things.",
                    "main_files": [{"path": "pkg/other/o1.py", "role": "O1."}],
                },
            ],
            "folds": [],
        }
    )


# ── Single-child collapse ─────────────────────────────────────────────────


def test_collapse_absorbs_single_child_and_stays_covered(tmp_path: Path) -> None:
    repo = _worker_repo(tmp_path)
    skel = build_skeleton(repo, "pkg")
    tree = _single_child_tree()

    normalized, report = normalize_enriched_tree(tree)

    kinds = [a.kind for a in report.actions]
    assert "collapse_single_child" in kinds
    worker = normalized.modules[0]
    assert {s.name for s in worker.submodules} == {"mm", "sample"}
    assert "Includes gpu:" in worker.description
    assert any(f.path == "pkg/worker/gpu" for f in normalized.folds)

    # The normalized tree passes the unchanged validator and full coverage —
    # this is the vllm attempt-1 salvage. The synthesized fold's evidence (the
    # child's own citations) satisfies Rule 8 without touching main_files.
    validate_enriched_tree(normalized, repo, _repository(), skel)
    cov = compute_coverage(normalized, skel)
    assert cov.missing == []
    assert "pkg/worker/gpu" in cov.folded


def test_collapse_does_not_touch_parent_main_files(tmp_path: Path) -> None:
    # Level 1: fold evidence is validated in place; the collapse must no longer
    # copy the absorbed child's files into the parent's main_files.
    repo = _worker_repo(tmp_path)
    skel = build_skeleton(repo, "pkg")

    normalized, _report = normalize_enriched_tree(_single_child_tree())

    worker = normalized.modules[0]
    assert [f.path for f in worker.main_files] == ["pkg/worker/base.py"]
    gpu_fold = next(f for f in normalized.folds if f.path == "pkg/worker/gpu")
    assert gpu_fold.evidence_files == ["pkg/worker/gpu/gpu_worker.py"]
    validate_enriched_tree(normalized, repo, _repository(), skel)


def test_collapse_walks_chains_to_fixpoint(tmp_path: Path) -> None:
    # pkg/a -> b -> c -> {x, y}: two chained single-child parents.
    _write(tmp_path / "pkg" / "a" / "a.py")
    _write(tmp_path / "pkg" / "a" / "b" / "b.py")
    _write(tmp_path / "pkg" / "a" / "b" / "c" / "c.py")
    _write(tmp_path / "pkg" / "a" / "b" / "c" / "x" / "x1.py")
    _write(tmp_path / "pkg" / "a" / "b" / "c" / "x" / "x2.py")
    _write(tmp_path / "pkg" / "a" / "b" / "c" / "y" / "y1.py")
    _write(tmp_path / "pkg" / "a" / "b" / "c" / "y" / "y2.py")
    _write(tmp_path / "pkg" / "other" / "o1.py")
    _write(tmp_path / "pkg" / "other" / "o2.py")
    skel = build_skeleton(tmp_path, "pkg")

    def _leaf(name: str) -> dict:
        return {
            "name": name,
            "path": f"pkg/a/b/c/{name}",
            "description": f"{name} leaf.",
            "main_files": [{"path": f"pkg/a/b/c/{name}/{name}1.py", "role": "F."}],
        }

    tree = EnrichedTree.model_validate(
        {
            "modules": [
                {
                    "name": "a",
                    "path": "pkg/a",
                    "description": "A.",
                    "main_files": [{"path": "pkg/a/a.py", "role": "A."}],
                    "submodules": [
                        {
                            "name": "b",
                            "path": "pkg/a/b",
                            "description": "B.",
                            "main_files": [{"path": "pkg/a/b/b.py", "role": "B."}],
                            "submodules": [
                                {
                                    "name": "c",
                                    "path": "pkg/a/b/c",
                                    "description": "C.",
                                    "main_files": [
                                        {"path": "pkg/a/b/c/c.py", "role": "C."}
                                    ],
                                    "submodules": [_leaf("x"), _leaf("y")],
                                }
                            ],
                        }
                    ],
                },
                {
                    "name": "other",
                    "path": "pkg/other",
                    "description": "Other.",
                    "main_files": [{"path": "pkg/other/o1.py", "role": "O."}],
                },
            ],
            "folds": [],
        }
    )

    normalized, report = normalize_enriched_tree(tree)

    a = normalized.modules[0]
    assert {s.name for s in a.submodules} == {"x", "y"}
    collapsed = {x.path for x in report.actions if x.kind == "collapse_single_child"}
    assert collapsed == {"pkg/a/b", "pkg/a/b/c"}
    validate_enriched_tree(normalized, tmp_path, _repository(), skel)
    assert compute_coverage(normalized, skel).missing == []


def test_protected_paths_are_never_collapsed(tmp_path: Path) -> None:
    _worker_repo(tmp_path)

    # As a spine's promotion parent, pkg/worker may legitimately show a single
    # local child — its other children return at merge.
    normalized, report = normalize_enriched_tree(
        _single_child_tree(), protected_paths={"pkg/worker"}
    )
    assert report.actions == []
    assert normalized.modules[0].submodules[0].name == "gpu"

    # As the absorbee, the protected path must stay emitted (merge grafts into
    # it by path), so the collapse is skipped too.
    normalized, report = normalize_enriched_tree(
        _single_child_tree(), protected_paths={"pkg/worker/gpu"}
    )
    assert all(a.kind != "collapse_single_child" for a in report.actions)
    assert normalized.modules[0].submodules[0].name == "gpu"


def test_collapse_retargets_folds_into_absorbed_child(tmp_path: Path) -> None:
    repo = _worker_repo(tmp_path)
    # An extra foldable leaf under gpu whose fold targets gpu itself.
    _write(repo / "pkg" / "worker" / "gpu" / "metrics" / "stats.py")
    skel = build_skeleton(repo, "pkg")

    data = _single_child_tree().model_dump(mode="json")
    data["folds"] = [
        {
            "path": "pkg/worker/gpu/metrics",
            "into": "pkg/worker/gpu",
            "reason": "single-file metrics helper",
            "evidence_files": ["pkg/worker/gpu/metrics/stats.py"],
        }
    ]
    tree = EnrichedTree.model_validate(data)

    normalized, _report = normalize_enriched_tree(tree)

    metrics_fold = next(
        f for f in normalized.folds if f.path == "pkg/worker/gpu/metrics"
    )
    assert metrics_fold.into == "pkg/worker"
    validate_enriched_tree(normalized, repo, _repository(), skel)
    assert compute_coverage(normalized, skel).missing == []


def test_description_merge_is_capped(tmp_path: Path) -> None:
    _worker_repo(tmp_path)
    data = _single_child_tree().model_dump(mode="json")
    data["modules"][0]["description"] = "W" * 1990
    data["modules"][0]["submodules"][0]["description"] = "G" * 1990
    tree = EnrichedTree.model_validate(data)

    normalized, _report = normalize_enriched_tree(tree)

    assert len(normalized.modules[0].description) <= 2000
    # The result still round-trips through the strict schema.
    EnrichedTree.model_validate(normalized.model_dump(mode="json"))


def test_clean_tree_is_untouched(tmp_path: Path) -> None:
    _worker_repo(tmp_path)
    normalized, _report = normalize_enriched_tree(_single_child_tree())
    again, report2 = normalize_enriched_tree(normalized)
    assert report2.actions == []
    assert again.model_dump(mode="json") == normalized.model_dump(mode="json")
