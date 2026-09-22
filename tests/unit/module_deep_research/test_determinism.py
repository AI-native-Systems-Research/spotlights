"""Determinism analysis over `--deep-research-repeat` sidecars.

The Claude matcher is never spawned here — a deterministic fake stands in, so
these tests assert the histogram + all-pairs confusion math, cluster-completion
safety, and the sidecar loader.
"""

from __future__ import annotations

import json
from pathlib import Path

from spotlights_engine.module_deep_research.determinism import (
    build_report,
    claude_matcher,
    load_repeat_runs,
)
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput


def _finding(n: int, title: str, url: str) -> Finding:
    return Finding(
        finding_id=f"find-seg-{n:04d}",
        title=title,
        url=url,
        source_type="paper",
        technique_summary="s",
    )


def _run(*papers: tuple[str, str]) -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(
        findings=[_finding(i + 1, t, u) for i, (t, u) in enumerate(papers)]
    )


def _exact_matcher(findings_by_run):
    """Cluster by exact (title, url) — deterministic stand-in for Claude."""
    groups: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, findings in enumerate(findings_by_run):
        for f, fnd in enumerate(findings):
            groups.setdefault((fnd.title, fnd.url), []).append((r, f))
    return list(groups.values())


def test_all_runs_agree_is_fully_deterministic() -> None:
    runs = [_run(("A", "u/a"), ("B", "u/b")) for _ in range(3)]
    rep = build_report(runs, matcher=_exact_matcher)

    assert rep.n_runs == 3
    assert rep.n_papers == 2
    # Both papers appear in all 3 runs.
    assert rep.histogram == {3: 2}
    assert rep.confusion.fp == 0
    assert rep.confusion.fn == 0
    assert rep.confusion.determinism == 1.0
    # 2 papers * ordered pairs both-present (3*2) = 12.
    assert rep.confusion.tp == 12


def test_flaky_paper_lowers_votes_and_adds_fp_fn() -> None:
    # Paper A in all 3 runs; paper B only in run 0.
    runs = [
        _run(("A", "u/a"), ("B", "u/b")),
        _run(("A", "u/a")),
        _run(("A", "u/a")),
    ]
    rep = build_report(runs, matcher=_exact_matcher)

    assert rep.n_papers == 2
    assert rep.histogram == {3: 1, 1: 1}  # A: 3 votes, B: 1 vote
    # B present in 1 of 3 runs -> ordered disagreements m*(n-m) = 1*2 = 2 each way.
    assert rep.confusion.fp == 2
    assert rep.confusion.fn == 2
    # A contributes tp 3*2=6; B contributes 1*0=0.
    assert rep.confusion.tp == 6
    # FP == FN by all-pairs symmetry.
    assert rep.confusion.fp == rep.confusion.fn
    # Flakiest-first ordering: B (1 vote) before A (3 votes).
    assert rep.papers[0].title == "B" and rep.papers[0].votes == 1


def test_fuzzy_matcher_collapses_reformatted_duplicates() -> None:
    # Same paper, different URL formatting per run. A fuzzy matcher (here: match
    # on title only) collapses them; an exact (title,url) diff would not.
    runs = [
        _run(("Muon", "arxiv.org/abs/1")),
        _run(("Muon", "arxiv.org/pdf/1")),
    ]

    def _title_matcher(findings_by_run):
        groups: dict[str, list[tuple[int, int]]] = {}
        for r, findings in enumerate(findings_by_run):
            for f, fnd in enumerate(findings):
                groups.setdefault(fnd.title, []).append((r, f))
        return list(groups.values())

    rep = build_report(runs, matcher=_title_matcher)
    assert rep.n_papers == 1
    assert rep.histogram == {2: 1}  # matched across both runs
    assert rep.confusion.determinism == 1.0


def test_unmatched_findings_become_singletons() -> None:
    # Matcher omits run 1's finding entirely; completion must add it back.
    runs = [_run(("A", "u/a")), _run(("A", "u/a"))]

    def _partial_matcher(findings_by_run):
        return [[(0, 0)]]  # forgets (1, 0)

    rep = build_report(runs, matcher=_partial_matcher)
    assert rep.n_papers == 2  # (0,0) + recovered singleton (1,0)
    assert any("unmatched" in i for i in rep.issues)


def test_out_of_range_ref_is_dropped_with_issue() -> None:
    runs = [_run(("A", "u/a"))]

    def _bad_matcher(findings_by_run):
        return [[(0, 0), (5, 9)]]  # (5,9) does not exist

    rep = build_report(runs, matcher=_bad_matcher)
    assert rep.n_papers == 1
    assert any("out-of-range" in i for i in rep.issues)


def test_load_repeat_runs_orders_canonical_then_indexed(tmp_path: Path) -> None:
    def _write(name: str, title: str) -> None:
        payload = {
            "output": _run((title, f"u/{title}")).model_dump(mode="json"),
            "duration_s": 1.0,
        }
        (tmp_path / name).write_text(json.dumps(payload), encoding="utf-8")

    _write("module_deep_research.json", "canonical")
    _write("module_deep_research.1.json", "one")
    _write("module_deep_research.2.json", "two")
    # Non-sidecar noise must be ignored.
    (tmp_path / "module_deep_research.last_message.md").write_text("x")

    runs = load_repeat_runs(tmp_path)
    assert len(runs) == 3
    assert runs[0].findings[0].title == "canonical"
    assert runs[1].findings[0].title == "one"
    assert runs[2].findings[0].title == "two"


def test_render_smoke() -> None:
    runs = [_run(("A", "u/a")), _run(("A", "u/a"), ("B", "u/b"))]
    out = build_report(runs, matcher=_exact_matcher).render()
    assert "Voting histogram" in out
    assert "confusion matrix" in out.lower()


def test_claude_matcher_parses_ids(monkeypatch) -> None:
    """`claude_matcher` shells out; here we stub subprocess to assert parsing of
    the `rR_fF` id scheme back into (run, finding) refs."""
    import spotlights_engine.module_deep_research.determinism as det

    result_event = {
        "type": "result",
        "structured_output": {
            "clusters": [
                {"members": ["r0_f0", "r1_f0"]},
                {"members": ["r1_f1"]},
            ]
        },
    }

    class _Completed:
        returncode = 0
        stdout = (json.dumps(result_event) + "\n").encode("utf-8")
        stderr = b""

    monkeypatch.setattr(det.subprocess, "run", lambda *a, **k: _Completed())
    runs = [_run(("A", "u/a")), _run(("A", "u/a"), ("B", "u/b"))]
    clusters = claude_matcher([list(r.findings) for r in runs])
    assert clusters == [[(0, 0), (1, 0)], [(1, 1)]]
