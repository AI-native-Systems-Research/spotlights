"""Unit tests for the K-run consensus helpers.

Covers `resolve_consensus_threshold`, `merge_run`, and `consensus_merge` —
the machinery that keeps only findings recurring across `>= threshold` of the
K merged search runs.
"""

from __future__ import annotations

import json

import pytest

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    RunResult,
    consensus_merge,
    merge_run,
    resolve_consensus_threshold,
)
from spotlights_engine.module_deep_research.validation import AgentFinding


def _finding(title: str, url: str) -> AgentFinding:
    return AgentFinding(
        finding_id="find-0001",
        title=title,
        url=url,
        source_type="paper",
        technique_summary="ts",
    )


def _run(*findings: AgentFinding) -> RunResult:
    return RunResult(findings=list(findings))


def _outcome(agent_name: str, findings: list[dict]) -> RunnerOutcome:
    payload = {"findings": findings, "issues": [], "search_queries": []}
    return RunnerOutcome(
        agent_name=agent_name,
        result=AgentExecResult(
            command=["agent"],
            returncode=0,
            stdout="",
            stderr="",
            final_message=json.dumps(payload),
        ),
    )


def _wire(title: str, url: str) -> dict:
    return {
        "finding_id": "find-0001",
        "title": title,
        "url": url,
        "source_type": "paper",
        "technique_summary": "ts",
    }


# resolve_consensus_threshold -------------------------------------------------


@pytest.mark.parametrize(
    ("k", "expected"),
    [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3)],
)
def test_threshold_defaults_to_ceil_half_k(k: int, expected: int) -> None:
    assert resolve_consensus_threshold(k, None) == expected


def test_threshold_explicit_is_clamped_to_range() -> None:
    assert resolve_consensus_threshold(3, 2) == 2
    assert resolve_consensus_threshold(3, 0) == 1  # clamp up to 1
    assert resolve_consensus_threshold(3, 5) == 3  # threshold > K -> K


# merge_run (within-run union + dedup) ----------------------------------------


def test_merge_run_unions_and_dedups_within_a_run() -> None:
    # Two runners in one run citing the same arxiv paper (abs vs pdf) collapse.
    outcomes = [
        _outcome("codex", [_wire("PagedAttention", "https://arxiv.org/abs/2309.06180")]),
        _outcome(
            "claude",
            [
                _wire("PagedAttention", "https://arxiv.org/pdf/2309.06180v2.pdf"),
                _wire("vAttention", "https://arxiv.org/abs/2405.04437"),
            ],
        ),
    ]

    run = merge_run(outcomes)

    assert [f.title for f in run.findings] == ["PagedAttention", "vAttention"]


def test_merge_run_records_runner_error_issue() -> None:
    run = merge_run([RunnerOutcome(agent_name="codex", error="boom")])

    assert run.findings == []
    assert run.issues and run.issues[0].recoverable is True


# consensus_merge -------------------------------------------------------------


def test_finding_at_or_above_threshold_kept_below_dropped() -> None:
    runs = [
        _run(_finding("Keeper", "https://x/keep")),
        _run(_finding("Keeper", "https://x/keep")),
        _run(_finding("Loner", "https://x/lone")),
    ]

    output = consensus_merge(runs, k=3, threshold=None, segment="mod")  # thr=2

    assert [f.title for f in output.findings] == ["Keeper"]


def test_within_run_duplicate_counts_once() -> None:
    # The same source appears twice in one run and once in a second run. Per-run
    # voting gives it 2 votes (not 3), so an explicit threshold of 3 drops it —
    # if within-run dups were counted separately it would have reached 3.
    runs = [
        _run(
            _finding("Dup", "https://x/dup"),
            _finding("Dup", "https://x/dup"),
        ),
        _run(_finding("Dup", "https://x/dup")),
        _run(_finding("Other", "https://x/other")),
    ]

    output = consensus_merge(runs, k=3, threshold=3, segment="mod")

    assert output.findings == []


def test_ordering_is_by_descending_votes_then_first_occurrence() -> None:
    runs = [
        _run(_finding("A", "https://x/a"), _finding("B", "https://x/b")),
        _run(_finding("B", "https://x/b")),
        _run(_finding("A", "https://x/a")),
    ]

    output = consensus_merge(runs, k=3, threshold=1, segment="mod")

    # A and B each appear in 2 runs; tie broken by first occurrence (A at run0
    # pos0 before B at run0 pos1).
    assert [f.title for f in output.findings] == ["A", "B"]


def test_arxiv_variants_across_runs_collapse_to_one_vote() -> None:
    runs = [
        _run(_finding("PagedAttention", "https://arxiv.org/abs/2309.06180")),
        _run(_finding("PagedAttention", "https://arxiv.org/pdf/2309.06180v2.pdf")),
    ]

    output = consensus_merge(runs, k=2, threshold=2, segment="mod")

    assert [f.title for f in output.findings] == ["PagedAttention"]


def test_k1_keeps_every_finding_uncapped() -> None:
    findings = [_finding(f"T{i}", f"https://x/{i}") for i in range(40)]
    runs = [_run(*findings)]

    output = consensus_merge(runs, k=1, threshold=None, segment="mod")  # thr=1

    assert len(output.findings) == 40


def test_empty_run_counts_toward_k_but_casts_no_votes() -> None:
    # One of three runs is empty (k_effective=2). The default threshold resolves
    # against k_effective -> ceil(2/2)=1, so a source seen in one live run
    # survives rather than being buried by the failed run raising the bar.
    runs = [
        _run(_finding("A", "https://x/a")),
        _run(),  # empty
        _run(_finding("B", "https://x/b")),
    ]

    lenient = consensus_merge(runs, k=3, threshold=None, segment="mod")
    assert sorted(f.title for f in lenient.findings) == ["A", "B"]

    # An explicit threshold is honored but clamped to [1, k_effective]: asking
    # for 3 (unanimous over nominal K) can never exceed the 2 live runs.
    clamped = consensus_merge(runs, k=3, threshold=3, segment="mod")
    assert clamped.findings == []  # A and B each have only 1 vote, need 2


def test_empty_run_records_a_step_issue() -> None:
    runs = [_run(_finding("A", "https://x/a")), _run()]

    output = consensus_merge(runs, k=2, threshold=None, segment="mod")

    assert any("produced no findings" in i.message for i in output.issues)


def test_findings_are_renumbered_with_the_segment() -> None:
    runs = [_run(_finding("A", "https://x/a"), _finding("B", "https://x/b"))]

    output = consensus_merge(runs, k=1, threshold=1, segment="kv_offload")

    assert [f.finding_id for f in output.findings] == [
        "find-kv_offload-0001",
        "find-kv_offload-0002",
    ]
