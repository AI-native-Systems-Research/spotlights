"""Unit tests for the deep-research search-log markdown renderer."""

from __future__ import annotations

from spotlights_engine.module_deep_research.search_log import render_search_log_markdown
from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.search import SearchQueryLog, SearchResult


def _output(**kwargs) -> ModuleDeepResearchOutput:
    return ModuleDeepResearchOutput(**kwargs)


def test_render_groups_by_agent_and_numbers_queries() -> None:
    output = _output(
        search_queries=[
            SearchQueryLog(
                agent="codex",
                query="paged kv cache",
                tool="web_search",
                results=[
                    SearchResult(
                        title="vLLM", url="https://x", snippet="PagedAttention."
                    )
                ],
            ),
            SearchQueryLog(agent="codex", query="second query", results=[]),
            SearchQueryLog(agent="claude", query="claude query", results=[]),
        ],
    )

    md = render_search_log_markdown(output, qn="inference/attention")

    assert "# Deep-research search log — inference/attention" in md
    assert "_3 queries across 2 agents._" in md
    assert "## Agent: codex" in md
    assert "## Agent: claude" in md
    assert "### 1. `paged kv cache`  (tool: web_search)" in md
    assert "### 2. `second query`" in md
    assert "- [vLLM](https://x) — PagedAttention." in md
    # claude's query is numbered from 1 within its own agent group.
    assert "### 1. `claude query`" in md


def test_render_shows_no_results_marker_for_empty_query() -> None:
    output = _output(
        search_queries=[SearchQueryLog(agent="codex", query="q", results=[])],
    )

    md = render_search_log_markdown(output, qn="mod")

    assert "_(no results returned)_" in md


def test_render_is_byte_stable_for_equal_input() -> None:
    queries = [
        SearchQueryLog(agent="codex", query="a", results=[]),
        SearchQueryLog(agent="claude", query="b", results=[]),
    ]
    out1 = _output(search_queries=list(queries))
    out2 = _output(search_queries=list(queries))

    assert render_search_log_markdown(out1, qn="mod") == render_search_log_markdown(
        out2, qn="mod"
    )


def test_render_notes_crashed_runner_with_no_queries() -> None:
    output = _output(
        search_queries=[SearchQueryLog(agent="codex", query="q", results=[])],
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

    assert "_1 queries across 2 agents._" in md
    assert "## Agent: claude" in md
    assert "_(runner failed — no queries captured)_" in md
