"""Unit tests for the step-3 single-paper finding filter (`--paper-link`)."""

from __future__ import annotations

import json

from spotlights_engine.module_deep_research.agent_exec import AgentExecResult
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    merge_outcomes,
    select_paper_finding,
)
from spotlights_engine.module_deep_research.validation import AgentFinding
from spotlights_engine.schemas.pipeline import PaperFilter


def _finding(finding_id: str, title: str, url: str) -> AgentFinding:
    return AgentFinding(
        finding_id=finding_id,
        title=title,
        url=url,
        source_type="paper",
        technique_summary="summary",
    )


def _findings() -> list[AgentFinding]:
    return [
        _finding("find-0001", "Flash attention", "https://example.com/flash"),
        _finding("find-0002", "Preble prefix caching", "https://arxiv.org/abs/2504.19874v2"),
        _finding("find-0003", "Speculative decoding", "https://example.com/spec"),
    ]


def test_select_by_url() -> None:
    matched, on = select_paper_finding(
        _findings(), url="https://example.com/spec", title=None
    )

    assert on == "url"
    assert [f.finding_id for f in matched] == ["find-0003"]


def test_select_by_title_fallback_when_url_differs() -> None:
    # The paper is cited at a different (conference PDF) URL than the finding's,
    # so only the title matches.
    matched, on = select_paper_finding(
        _findings(),
        url="https://proceedings.mlr.press/v999/preble.pdf",
        title="Preble prefix caching",
    )

    assert on == "title"
    assert [f.finding_id for f in matched] == ["find-0002"]


def test_arxiv_abs_pdf_and_version_normalize_equal() -> None:
    # Finding stored the versioned abs URL; the cited link is the unversioned pdf.
    matched, on = select_paper_finding(
        _findings(), url="https://arxiv.org/pdf/2504.19874", title=None
    )

    assert on == "url"
    assert [f.finding_id for f in matched] == ["find-0002"]


def test_no_match_returns_empty() -> None:
    matched, on = select_paper_finding(
        _findings(), url="https://nowhere.example/none", title="Nonexistent paper"
    )

    assert matched == []
    assert on is None


def test_url_preferred_over_title_when_both_present() -> None:
    # Give a URL that matches find-0003 and a title that matches find-0001.
    # URL must win.
    matched, on = select_paper_finding(
        _findings(),
        url="https://example.com/spec",
        title="Flash attention",
    )

    assert on == "url"
    assert [f.finding_id for f in matched] == ["find-0003"]


# ---------------------------------------------------------------------------
# merge_outcomes with paper_filter
# ---------------------------------------------------------------------------


def _outcome(findings: list[dict]) -> RunnerOutcome:
    payload = json.dumps({"findings": findings, "issues": []})
    return RunnerOutcome(
        agent_name="fake",
        result=AgentExecResult(
            command=["fake"],
            returncode=0,
            stdout="",
            stderr="",
            final_message=payload,
        ),
    )


def _wire(finding_id: str, title: str, url: str) -> dict:
    return {
        "finding_id": finding_id,
        "title": title,
        "url": url,
        "source_type": "paper",
        "technique_summary": "summary",
    }


def test_merge_outcomes_collapses_to_matching_finding() -> None:
    outcome = _outcome(
        [
            _wire("find-0001", "Flash attention", "https://example.com/flash"),
            _wire("find-0002", "Preble", "https://arxiv.org/abs/2504.19874v2"),
        ]
    )

    output = merge_outcomes(
        [outcome],
        max_findings_per_module=30,
        segment="mod",
        paper_filter=PaperFilter(url="https://arxiv.org/pdf/2504.19874"),
    )

    assert len(output.findings) == 1
    assert output.findings[0].finding_id == "find-mod-0001"
    assert output.findings[0].title == "Preble"
    assert output.issues == []


def test_merge_outcomes_no_match_yields_empty_plus_recoverable_issue() -> None:
    outcome = _outcome(
        [_wire("find-0001", "Flash attention", "https://example.com/flash")]
    )

    output = merge_outcomes(
        [outcome],
        max_findings_per_module=30,
        segment="mod",
        paper_filter=PaperFilter(url="https://nowhere.example/x", title="Nothing"),
    )

    assert output.findings == []
    assert len(output.issues) == 1
    assert output.issues[0].recoverable is True
    assert "matched no finding" in output.issues[0].message
