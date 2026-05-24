"""Aggregator tests: index rows + module page views."""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.results_renderer.aggregator import aggregate
from spotlights_engine.results_renderer.api import RendererConfig
from spotlights_engine.results_renderer.loader import load_run

from tests.unit.results_renderer._fixtures import make_full_run


def test_index_rows_relevant_findings_count(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)
    loaded = load_run(tmp_path)

    rows, views, skipped, warnings = aggregate(loaded, RendererConfig())

    by_qn = {r.module_qualified_name: r for r in rows}
    # kv_offload has two deep_research_proposals on cand 1 (find-0001 and
    # find-0099). Findings list contains find-0001 and find-0002. Intersection
    # is {find-0001} -> 1 relevant.
    assert by_qn["v1.kv_offload"].n_relevant_findings == 1
    # kernels has no proposals on its candidate -> 0.
    assert by_qn["kernels"].n_relevant_findings == 0
    assert by_qn["v1.kv_offload"].n_candidates == 3
    assert by_qn["v1.kv_offload"].n_high_impact_candidates == 1


def test_index_rows_sort_order(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)
    loaded = load_run(tmp_path)

    rows, _, _, _ = aggregate(loaded, RendererConfig())
    # kv_offload (3 cands, 1 high), kernels (1 cand, 1 high) — high is tied,
    # so n_candidates DESC sorts kv_offload first.
    assert [r.module_qualified_name for r in rows] == ["v1.kv_offload", "kernels"]


def test_module_page_view_candidate_sort(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)
    loaded = load_run(tmp_path)

    _, views, _, _ = aggregate(loaded, RendererConfig())
    kv = views["v1.kv_offload"]
    # cand-0001 high, cand-0002 medium, cand-0003 low.
    assert [c.id for c in kv.candidates_sorted] == [
        "cand-0001",
        "cand-0002",
        "cand-0003",
    ]


def test_aggregate_resolves_module_in_tree(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)
    loaded = load_run(tmp_path)
    _, views, _, _ = aggregate(loaded, RendererConfig())
    assert views["v1.kv_offload"].module is not None
    assert views["v1.kv_offload"].module.path == "src/v1/kv_offload"
    assert views["kernels"].module is not None


def test_skipped_modules_excluded_when_disabled(tmp_path: Path) -> None:
    paths, _ = make_full_run(tmp_path)
    # Pretend kernels is SKIPPED via manifest tweak.
    import json
    manifest = json.loads(paths.manifest_path.read_text())
    manifest["modules"]["kernels"]["status"] = "SKIPPED"
    paths.manifest_path.write_text(json.dumps(manifest))

    loaded = load_run(tmp_path)
    _, _, skipped, _ = aggregate(
        loaded, RendererConfig(include_skipped_modules=False)
    )
    assert "kernels" in skipped
