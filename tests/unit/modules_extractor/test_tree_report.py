"""Tests for the derived per-directory decision report (`tree_report.py`).

The report is a pure join over four already-persisted stage models, so almost
everything here is hand-built: no Claude, and (except the reconciliation test,
which builds a real `Skeleton` from a synthetic repo) no filesystem.

See `design/module_extractor_visualization.md`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from spotlights_engine.modules_extractor.coverage import (
    CoverageReport,
    compute_coverage,
)
from spotlights_engine.modules_extractor.skeleton import build_skeleton
from spotlights_engine.modules_extractor.stage_schemas import (
    EnrichedTree,
    Skeleton,
    SkeletonNode,
    SourceRootDecision,
)
from spotlights_engine.modules_extractor.tree_report import (
    build_tree_decision_report,
    load_report_from_run_dir,
    main,
    render_markdown,
    report_json_text,
)

# ── Fixtures ──────────────────────────────────────────────────────────────


def _node(
    path: str,
    *,
    required: bool = True,
    reasons: list[str] | None = None,
    direct: int = 2,
    children: list[SkeletonNode] | None = None,
) -> SkeletonNode:
    kids = children or []
    return SkeletonNode(
        path=path,
        direct_source_file_count=direct,
        source_child_count=len(kids),
        representative_files=[],
        required=required,
        required_reasons=reasons or (["two_or_more_direct_source_files"] if required else []),
        children=kids,
    )


def _skeleton() -> Skeleton:
    """`pkg/core` with one of every outcome beneath it.

    - `kv_offload`  → emitted leaf
    - `scheduler`   → emitted parent (owns `policies`)
    - `util`        → folded into `pkg/core`
    - `util/deep`   → folded into `pkg/core` (a grandparent fold)
    - `broken`      → required but neither emitted nor folded → missing
    - `legacy`      → optional and untouched → optional_unaccounted
    """
    return Skeleton(
        source_root="pkg",
        nodes=[
            _node(
                "pkg/core",
                reasons=["two_or_more_direct_source_files"],
                children=[
                    _node("pkg/core/broken"),
                    _node("pkg/core/kv_offload"),
                    _node("pkg/core/legacy", required=False, direct=1),
                    _node(
                        "pkg/core/scheduler",
                        children=[_node("pkg/core/scheduler/policies")],
                    ),
                    _node(
                        "pkg/core/util",
                        required=False,
                        direct=1,
                        children=[
                            _node("pkg/core/util/deep", required=False, direct=1)
                        ],
                    ),
                ],
            )
        ],
        ignored=["__pycache__", "node_modules"],
        excluded=["pkg/generated"],
        organizational_only=["pkg/core/util"],
        skipped_symlinks=["pkg/core/link"],
        inventory_fingerprint="fp-abc",
    )


def _module(name: str, path: str, *, subs: list[dict] | None = None) -> dict:
    return {
        "name": name,
        "path": path,
        "description": f"{name} module.",
        "main_files": [{"path": f"{path}/main.py", "role": "Entry."}],
        "submodules": subs or [],
    }


def _enriched() -> EnrichedTree:
    return EnrichedTree.model_validate(
        {
            "modules": [
                _module(
                    "core",
                    "pkg/core",
                    subs=[
                        _module("kv_offload", "pkg/core/kv_offload"),
                        _module(
                            "scheduler",
                            "pkg/core/scheduler",
                            subs=[
                                _module(
                                    "policies", "pkg/core/scheduler/policies"
                                )
                            ],
                        ),
                    ],
                )
            ],
            "folds": [
                {
                    "path": "pkg/core/util",
                    "into": "pkg/core",
                    "reason": "single helper file supporting core's public API",
                    "evidence_files": ["pkg/core/util/helper.py"],
                },
                {
                    # Grandparent fold: two levels up, skipping `util`.
                    "path": "pkg/core/util/deep",
                    "into": "pkg/core",
                    "reason": "one more helper, same owner",
                    "evidence_files": ["pkg/core/util/deep/more.py"],
                },
            ],
        }
    )


def _decision() -> SourceRootDecision:
    return SourceRootDecision.model_validate(
        {
            "repository": {
                "name": "demo",
                "summary": "A demo package.",
                "source_root": "pkg",
            },
            "excluded_source_paths": [
                {
                    "path": "tests",
                    "reason": "centralized_tests",
                    "explanation": "Repository-wide test suite.",
                },
                {
                    "path": "docs",
                    "reason": "docs",
                    "explanation": "Documentation only.",
                },
            ],
        }
    )


def _report(coverage: CoverageReport | None = None):
    skeleton = _skeleton()
    enriched = _enriched()
    cov = coverage if coverage is not None else compute_coverage(enriched, skeleton)
    return build_tree_decision_report(skeleton, enriched, cov, _decision())


def _by_path(report) -> dict:
    return {n.path: n for n in report.iter_nodes()}


# ── Classification ────────────────────────────────────────────────────────


def test_every_decision_value_is_produced() -> None:
    nodes = _by_path(_report())
    assert nodes["pkg/core"].decision == "emitted_parent"
    assert nodes["pkg/core/kv_offload"].decision == "emitted_leaf"
    assert nodes["pkg/core/scheduler"].decision == "emitted_parent"
    assert nodes["pkg/core/scheduler/policies"].decision == "emitted_leaf"
    assert nodes["pkg/core/util"].decision == "folded"
    assert nodes["pkg/core/util/deep"].decision == "folded"
    assert nodes["pkg/core/broken"].decision == "missing"
    assert nodes["pkg/core/legacy"].decision == "optional_unaccounted"


def test_emitted_parent_vs_leaf_tracks_submodules() -> None:
    nodes = _by_path(_report())
    # `scheduler` owns one submodule → parent; `policies` owns none → leaf.
    assert nodes["pkg/core/scheduler"].emitted_name == "scheduler"
    assert nodes["pkg/core/scheduler"].decision == "emitted_parent"
    assert nodes["pkg/core/scheduler/policies"].decision == "emitted_leaf"
    # Only `pkg/core` is a top-level module.
    assert nodes["pkg/core"].top_level_module is True
    assert nodes["pkg/core/scheduler"].top_level_module is False


def test_fold_metadata_and_grandparent_fold() -> None:
    nodes = _by_path(_report())
    util = nodes["pkg/core/util"]
    assert util.folded_into == "pkg/core"
    assert util.evidence_files == ["pkg/core/util/helper.py"]
    assert "single helper file" in (util.fold_reason or "")
    # Grandparent fold: `into` skips the intermediate (also folded) directory.
    deep = nodes["pkg/core/util/deep"]
    assert deep.folded_into == "pkg/core"
    assert deep.path.startswith("pkg/core/util/")
    # Skeleton nesting is preserved regardless of the fold target.
    assert [c.path for c in util.children] == ["pkg/core/util/deep"]


def test_node_attributes_are_carried_through() -> None:
    nodes = _by_path(_report())
    core = nodes["pkg/core"]
    assert core.required is True
    assert core.required_reasons == ["two_or_more_direct_source_files"]
    assert core.direct_source_file_count == 2
    assert core.source_child_count == 5
    assert nodes["pkg/core/util"].organizational_only is True
    assert nodes["pkg/core"].organizational_only is False


def test_invalid_fold_tags_the_node_not_the_decision() -> None:
    skeleton, enriched = _skeleton(), _enriched()
    coverage = compute_coverage(enriched, skeleton)
    coverage = coverage.model_copy(
        update={"invalid_folds": ["pkg/core/util"]}
    )
    report = build_tree_decision_report(skeleton, enriched, coverage, _decision())
    util = _by_path(report)["pkg/core/util"]
    assert util.decision == "folded"  # decision is unchanged …
    assert util.fold_invalid is True  # … the problem is an attribute
    assert report.summary.invalid_folds == 1


def test_excluded_paths_are_grafted_from_stage1() -> None:
    report = _report()
    assert [(e.path, e.reason) for e in report.excluded_paths] == [
        ("docs", "docs"),
        ("tests", "centralized_tests"),
    ]
    assert report.summary.excluded == 2
    # Excluded paths are not skeleton nodes and never enter `nodes`.
    assert "docs" not in _by_path(report)


def test_pruned_section_lists_walk_leftovers() -> None:
    report = _report()
    assert report.pruned.ignored_dir_names == ["__pycache__", "node_modules"]
    assert report.pruned.excluded_paths == ["pkg/generated"]
    assert report.pruned.skipped_symlinks == ["pkg/core/link"]


def test_report_without_source_root_decision() -> None:
    skeleton, enriched = _skeleton(), _enriched()
    report = build_tree_decision_report(
        skeleton, enriched, compute_coverage(enriched, skeleton), None
    )
    assert report.excluded_paths == []
    assert report.summary.excluded == 0


# ── Reconciliation with coverage.json ─────────────────────────────────────


def _write(path: Path, content: str = "x = 1\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _real_run(tmp_path: Path) -> tuple[Skeleton, EnrichedTree, CoverageReport]:
    """A real Stage-2 skeleton over a synthetic repo + a matching Stage-3 tree."""
    _write(tmp_path / "pkg" / "core" / "engine.py")
    _write(tmp_path / "pkg" / "core" / "runner.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "a.py")
    _write(tmp_path / "pkg" / "core" / "kv_offload" / "b.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s1.py")
    _write(tmp_path / "pkg" / "core" / "scheduler" / "s2.py")
    _write(tmp_path / "pkg" / "core" / "util" / "helper.py")
    skeleton = build_skeleton(
        tmp_path, "pkg", excluded_dirs=frozenset(), excluded_files=frozenset()
    )
    enriched = EnrichedTree.model_validate(
        {
            "modules": [
                {
                    "name": "core",
                    "path": "pkg/core",
                    "description": "Core runtime.",
                    "main_files": [
                        {"path": "pkg/core/engine.py", "role": "Engine."},
                        {
                            "path": "pkg/core/util/helper.py",
                            "role": "Helper (folded).",
                        },
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
    )
    return skeleton, enriched, compute_coverage(enriched, skeleton)


def test_summary_reconciles_with_coverage_sets(tmp_path: Path) -> None:
    skeleton, enriched, coverage = _real_run(tmp_path)
    report = build_tree_decision_report(skeleton, enriched, coverage, None)

    paths = [n.path for n in report.iter_nodes()]
    # Every skeleton node appears exactly once, and nothing else appears.
    assert sorted(paths) == sorted(skeleton.all_paths())
    assert len(paths) == len(set(paths))

    s = report.summary
    assert s.total_nodes == len(skeleton.all_paths())
    # The five decisions partition the inventory.
    assert (
        s.emitted_parent
        + s.emitted_leaf
        + s.folded
        + s.missing
        + s.optional_unaccounted
        == s.total_nodes
    )
    # …and each bucket matches `coverage.json`'s own sets.
    assert s.emitted_parent + s.emitted_leaf == len(coverage.emitted)
    assert s.folded == len(coverage.folded)
    assert s.missing == len(coverage.missing)
    assert s.optional_unaccounted == len(coverage.optional_unaccounted)


def test_summary_reconciles_when_every_bucket_is_non_empty() -> None:
    """The reconciliation above runs on a clean tree, where `missing` and
    `optional_unaccounted` are both 0 and so prove nothing. The hand-built
    fixture populates all five buckets; the same identities must hold."""
    skeleton, enriched = _skeleton(), _enriched()
    coverage = compute_coverage(enriched, skeleton)
    assert coverage.missing and coverage.optional_unaccounted  # non-trivial
    report = build_tree_decision_report(skeleton, enriched, coverage, _decision())

    s = report.summary
    assert sorted(n.path for n in report.iter_nodes()) == sorted(
        skeleton.all_paths()
    )
    assert s.emitted_parent + s.emitted_leaf == len(coverage.emitted)
    assert s.folded == len(coverage.folded)
    assert s.missing == len(coverage.missing)
    assert s.optional_unaccounted == len(coverage.optional_unaccounted)
    assert (
        s.emitted_parent
        + s.emitted_leaf
        + s.folded
        + s.missing
        + s.optional_unaccounted
        == s.total_nodes
        == len(skeleton.all_paths())
    )


def test_top_level_rows_sum_to_the_global_summary(tmp_path: Path) -> None:
    skeleton, enriched, coverage = _real_run(tmp_path)
    report = build_tree_decision_report(skeleton, enriched, coverage, None)
    assert sum(t.counts.total_nodes for t in report.top_level) == (
        report.summary.total_nodes
    )
    assert [t.path for t in report.top_level] == [n.path for n in report.nodes]


# ── Determinism ───────────────────────────────────────────────────────────


def test_json_and_markdown_are_byte_identical_across_builds() -> None:
    first = build_tree_decision_report(
        _skeleton(), _enriched(), compute_coverage(_enriched(), _skeleton()), _decision()
    )
    second = build_tree_decision_report(
        _skeleton(), _enriched(), compute_coverage(_enriched(), _skeleton()), _decision()
    )
    assert report_json_text(first) == report_json_text(second)
    assert render_markdown(first) == render_markdown(second)


def test_child_order_is_independent_of_input_order() -> None:
    skeleton = _skeleton()
    shuffled = skeleton.model_copy(deep=True)
    shuffled.nodes[0].children.reverse()
    enriched = _enriched()
    a = build_tree_decision_report(
        skeleton, enriched, compute_coverage(enriched, skeleton), _decision()
    )
    b = build_tree_decision_report(
        shuffled, enriched, compute_coverage(enriched, shuffled), _decision()
    )
    assert report_json_text(a) == report_json_text(b)


# ── Markdown ──────────────────────────────────────────────────────────────


def test_markdown_smoke_legend_and_sections() -> None:
    md = render_markdown(_report())
    assert "Legend:" in md
    for symbol in ("●", "◐", "○", "✕", "✗", "⊘"):
        assert symbol in md
    # Global summary table first, then one `##` section per top-level node.
    assert md.index("## Summary") < md.index("## `pkg/core`")
    assert "```text" in md
    # Box-drawing tree with one line per node.
    assert "├── " in md and "└── " in md
    for path in _by_path(_report()):
        assert path.rsplit("/", 1)[-1] in md
    # Rollup annotation on an emitted parent.
    assert "(subtree:" in md
    # Excluded + pruned ledgers.
    assert "Excluded by stage 1" in md
    assert "## Pruned (not nodes)" in md


def test_problems_section_present_when_missing_or_invalid_folds() -> None:
    md = render_markdown(_report())  # the fixture has one `missing` node
    assert "## Problems" in md
    assert md.index("## Problems") < md.index("## `pkg/core`")
    assert "Missing required paths (1)" in md
    assert "`pkg/core/broken`" in md


def test_problems_section_absent_on_a_clean_run(tmp_path: Path) -> None:
    skeleton, enriched, coverage = _real_run(tmp_path)
    assert not coverage.missing and not coverage.invalid_folds
    md = render_markdown(
        build_tree_decision_report(skeleton, enriched, coverage, None)
    )
    assert "## Problems" not in md
    assert "Legend:" in md


def test_problems_section_lists_invalid_folds() -> None:
    skeleton, enriched = _skeleton(), _enriched()
    coverage = compute_coverage(enriched, skeleton).model_copy(
        update={"invalid_folds": ["pkg/core/util"]}
    )
    md = render_markdown(
        build_tree_decision_report(skeleton, enriched, coverage, _decision())
    )
    assert "Invalid folds (1)" in md
    assert "!INVALID-FOLD" in md


def test_render_filters_under_and_max_depth() -> None:
    report = _report()
    under = render_markdown(report, under="pkg/core/scheduler")
    assert "## `pkg/core/scheduler`" in under
    assert "kv_offload" not in under
    assert "policies" in under

    shallow = render_markdown(report, under="pkg/core", max_depth=0)
    assert "deeper node(s) hidden" in shallow
    assert "kv_offload" not in shallow


# ── CLI ───────────────────────────────────────────────────────────────────


def _run_dir(tmp_path: Path) -> Path:
    run = tmp_path / "modules_extractor"
    skeleton, enriched, coverage = _real_run(tmp_path / "repo")
    (run / "02_skeleton").mkdir(parents=True)
    (run / "02_skeleton" / "skeleton.json").write_text(
        json.dumps(skeleton.model_dump(mode="json")), encoding="utf-8"
    )
    (run / "enriched_tree.json").write_text(
        json.dumps(enriched.model_dump(mode="json")), encoding="utf-8"
    )
    (run / "coverage.json").write_text(
        json.dumps(coverage.model_dump(mode="json")), encoding="utf-8"
    )
    (run / "source_root_decision.json").write_text(
        json.dumps(_decision().model_dump(mode="json")), encoding="utf-8"
    )
    return run


def test_cli_writes_both_files(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    assert main([str(run)]) == 0
    written = json.loads((run / "tree_decisions.json").read_text(encoding="utf-8"))
    assert written["schema_version"] == "tree_decisions.v1"
    md = (run / "tree_decisions.md").read_text(encoding="utf-8")
    assert "Legend:" in md
    # Byte-identical to a fresh in-process build.
    report = load_report_from_run_dir(run)
    assert (run / "tree_decisions.json").read_text(encoding="utf-8") == (
        report_json_text(report)
    )
    assert md == render_markdown(report)


def test_cli_accepts_the_parent_of_the_run_dir(tmp_path: Path) -> None:
    run = _run_dir(tmp_path)
    assert main([str(run.parent)]) == 0
    assert (run / "tree_decisions.json").exists()


def test_cli_stdout_does_not_write(tmp_path: Path, capsys) -> None:
    run = _run_dir(tmp_path)
    assert main([str(run), "--stdout"]) == 0
    assert "Legend:" in capsys.readouterr().out
    assert not (run / "tree_decisions.md").exists()


def test_cli_under_and_max_depth(tmp_path: Path, capsys) -> None:
    run = _run_dir(tmp_path)
    assert main([str(run), "--stdout", "--under", "pkg/core/scheduler"]) == 0
    out = capsys.readouterr().out
    assert "## `pkg/core/scheduler`" in out
    assert "kv_offload" not in out

    assert main([str(run), "--stdout", "--max-depth", "0"]) == 0
    assert "deeper node(s) hidden" in capsys.readouterr().out


def test_cli_refuses_to_write_a_filtered_view_over_the_artifacts(
    tmp_path: Path,
) -> None:
    """A view filter must never replace the run's full-depth `tree_decisions.md`
    (which would also leave it disagreeing with the always-full JSON beside it),
    so `--under`/`--max-depth` are stdout-only."""
    run = _run_dir(tmp_path)
    for argv in (
        [str(run), "--under", "pkg/core/scheduler"],
        [str(run), "--max-depth", "0"],
    ):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
    assert not (run / "tree_decisions.md").exists()
    assert not (run / "tree_decisions.json").exists()


def test_cli_reports_unknown_paths_and_missing_artifacts(
    tmp_path: Path, capsys
) -> None:
    run = _run_dir(tmp_path)
    assert main([str(run), "--stdout", "--under", "nope/nowhere"]) == 2
    assert "no node matches" in capsys.readouterr().err

    empty = tmp_path / "empty"
    empty.mkdir()
    assert main([str(empty)]) == 2
    assert "tree_report:" in capsys.readouterr().err


# ── Scale ─────────────────────────────────────────────────────────────────


def _synthetic_skeleton(total: int) -> tuple[Skeleton, EnrichedTree]:
    """A ~`total`-node skeleton, 6 levels deep, with a plausible emit/fold mix."""
    counter = 0

    def _make(path: str, depth: int) -> SkeletonNode | None:
        nonlocal counter
        if counter >= total:
            return None
        counter += 1
        children: list[SkeletonNode] = []
        if depth < 6:
            for i in range(4):
                child = _make(f"{path}/d{depth}_{i}", depth + 1)
                if child is None:
                    break
                children.append(child)
        return _node(path, children=children, required=depth % 3 != 0)

    root = _make("src/pkg", 1)
    assert root is not None
    skeleton = Skeleton(
        source_root="src",
        nodes=[root],
        ignored=["node_modules"],
        excluded=[],
        organizational_only=[],
        skipped_symlinks=[],
        inventory_fingerprint="fp-synthetic",
    )
    # Emit the root plus its direct children; fold everything else into the
    # nearest emitted ancestor.
    emitted = {root.path} | {c.path for c in root.children}
    tops = [
        _module(
            "pkg",
            root.path,
            subs=[
                _module(f"m{i}", c.path)
                for i, c in enumerate(root.children)
            ],
        )
    ]
    folds = []
    for child in root.children:
        for node in _walk(child):
            if node.path in emitted:
                continue
            folds.append(
                {
                    "path": node.path,
                    "into": child.path,
                    "reason": "synthetic fold",
                    "evidence_files": [f"{node.path}/f.py"],
                }
            )
    return (
        skeleton,
        EnrichedTree.model_validate({"modules": tops, "folds": folds}),
    )


def _walk(node: SkeletonNode):
    yield node
    for child in node.children:
        yield from _walk(child)


@pytest.mark.parametrize("total", [1000])
def test_thousand_node_skeleton_builds_and_renders_fast(total: int) -> None:
    skeleton, enriched = _synthetic_skeleton(total)
    assert len(skeleton.all_paths()) >= 900
    coverage = compute_coverage(enriched, skeleton)

    start = time.perf_counter()
    report = build_tree_decision_report(skeleton, enriched, coverage, None)
    text = report_json_text(report)
    markdown = render_markdown(report)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"build+render took {elapsed:.3f}s"

    assert report.summary.total_nodes == len(skeleton.all_paths())
    # Byte-deterministic at scale too.
    again = build_tree_decision_report(skeleton, enriched, coverage, None)
    assert report_json_text(again) == text
    assert render_markdown(again) == markdown
    # Nothing is omitted: one tree line per node (plus fences and headings).
    assert markdown.count("── ") >= report.summary.total_nodes - 1
