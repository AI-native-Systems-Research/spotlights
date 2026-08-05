"""Unit tests for the deep-research search-log markdown renderer."""

from __future__ import annotations

from spotlights_engine.module_deep_research.search_log import render_search_log_markdown
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.search import SearchQueryLog, SearchResult


def _output(**kwargs) -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(**kwargs)


def test_render_groups_by_candidate_then_agent_and_numbers_queries() -> None:
    output = _output(
        search_queries=[
            SearchQueryLog(
                agent="codex",
                query="paged kv cache",
                tool="web_search",
                candidate_id="cand-mod-0001",
                results=[
                    SearchResult(
                        title="vLLM", url="https://x", snippet="PagedAttention."
                    )
                ],
            ),
            SearchQueryLog(
                agent="codex",
                query="second query",
                candidate_id="cand-mod-0001",
                results=[],
            ),
            SearchQueryLog(
                agent="claude",
                query="claude query",
                candidate_id="cand-mod-0001",
                results=[],
            ),
        ],
    )

    md = render_search_log_markdown(output, qn="inference/attention")

    assert "# Deep-research search log — inference/attention" in md
    # The header now counts candidates (decision D7), not agents.
    assert "_3 queries across 1 candidate(s)._" in md
    assert "## Candidate: cand-mod-0001" in md
    assert "### Agent: codex" in md
    assert "### Agent: claude" in md
    assert "### 1. `paged kv cache`  (tool: web_search)" in md
    assert "### 2. `second query`" in md
    assert "- [vLLM](https://x) — PagedAttention." in md
    # claude's query is numbered from 1 within its own agent group.
    assert "### 1. `claude query`" in md


def test_render_separates_queries_by_candidate() -> None:
    """Decision D7: one module log, grouped by the candidate each survey ran
    for, in first-appearance order."""
    output = _output(
        search_queries=[
            SearchQueryLog(
                agent="codex", query="q-a", candidate_id="cand-mod-0001", results=[]
            ),
            SearchQueryLog(
                agent="codex", query="q-b", candidate_id="cand-mod-0002", results=[]
            ),
        ],
    )

    md = render_search_log_markdown(output, qn="mod")

    assert "_2 queries across 2 candidate(s)._" in md
    assert md.index("## Candidate: cand-mod-0001") < md.index("## Candidate: cand-mod-0002")


def test_render_labels_unattributed_queries() -> None:
    """Standalone/older sidecars leave candidate_id None; they group under a
    sentinel label rather than being dropped."""
    output = _output(
        search_queries=[SearchQueryLog(agent="codex", query="q", results=[])],
    )

    md = render_search_log_markdown(output, qn="mod")

    assert "## Candidate: (no candidate)" in md


def test_render_shows_no_results_marker_for_empty_query() -> None:
    output = _output(
        search_queries=[SearchQueryLog(agent="codex", query="q", results=[])],
    )

    md = render_search_log_markdown(output, qn="mod")

    assert "_(no results returned)_" in md


def test_render_is_byte_stable_for_equal_input() -> None:
    queries = [
        SearchQueryLog(agent="codex", query="a", candidate_id="cand-mod-0001", results=[]),
        SearchQueryLog(agent="claude", query="b", candidate_id="cand-mod-0001", results=[]),
    ]
    out1 = _output(search_queries=list(queries))
    out2 = _output(search_queries=list(queries))

    assert render_search_log_markdown(out1, qn="mod") == render_search_log_markdown(
        out2, qn="mod"
    )


def test_render_notes_crashed_runner_with_no_queries() -> None:
    output = _output(
        search_queries=[
            SearchQueryLog(
                agent="codex", query="q", candidate_id="cand-mod-0001", results=[]
            )
        ],
        issues=[
            StepIssue(
                step="module_deep_research",
                severity="error",
                message="module_deep_research claude execution failed: boom",
                recoverable=True,
            )
        ],
    )

    md = render_search_log_markdown(output, qn="mod")

    # Crashed runners that produced no queries are surfaced under their own
    # section so a missing agent isn't read as "ran zero searches".
    assert "## Failed runners" in md
    assert "claude — runner failed, no queries captured" in md


def test_render_collapses_repeated_runner_failures_with_a_count() -> None:
    """When the same agent crashes on several candidates (decision D7), it is
    reported once with a ×count suffix rather than one line per candidate."""
    output = _output(
        search_queries=[
            SearchQueryLog(
                agent="codex", query="q", candidate_id="cand-mod-0001", results=[]
            )
        ],
        issues=[
            StepIssue(
                step="module_deep_research",
                severity="error",
                message="module_deep_research claude execution failed: boom",
                recoverable=True,
            ),
            StepIssue(
                step="module_deep_research",
                severity="error",
                message="module_deep_research claude execution failed: boom again",
                recoverable=True,
            ),
        ],
    )

    md = render_search_log_markdown(output, qn="mod")

    assert "claude ×2 — runner failed, no queries captured" in md
