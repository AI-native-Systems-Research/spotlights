"""Unit tests for search-query collection in `merge_outcomes`."""

from __future__ import annotations

import json

from spotlights_engine.local_agent.base import AgentExecResult
from spotlights_engine.module_deep_research.orchestration import (
    RunnerOutcome,
    merge_outcomes,
)


def _result(payload: dict) -> AgentExecResult:
    return AgentExecResult(
        command=["agent"],
        returncode=0,
        stdout="",
        stderr="",
        final_message=json.dumps(payload),
    )


def _outcome(agent_name: str, queries: list[dict]) -> RunnerOutcome:
    return RunnerOutcome(
        agent_name=agent_name,
        result=_result({"findings": [], "issues": [], "search_queries": queries}),
    )


def test_merge_tags_queries_with_agent_and_preserves_order() -> None:
    outcomes = [
        _outcome(
            "codex",
            [
                {"query": "q-codex-1", "tool": "web_search", "results": []},
                {"query": "q-codex-2", "results": []},
            ],
        ),
        _outcome("claude", [{"query": "q-claude-1", "results": []}]),
    ]

    output = merge_outcomes(outcomes, max_findings_per_module=30, segment="mod")

    assert [(q.agent, q.query) for q in output.search_queries] == [
        ("codex", "q-codex-1"),
        ("codex", "q-codex-2"),
        ("claude", "q-claude-1"),
    ]


def test_merge_does_not_dedup_identical_queries_across_agents() -> None:
    outcomes = [
        _outcome("codex", [{"query": "same query", "results": []}]),
        _outcome("claude", [{"query": "same query", "results": []}]),
    ]

    output = merge_outcomes(outcomes, max_findings_per_module=30, segment="mod")

    assert len(output.search_queries) == 2
    assert output.search_queries[0].agent == "codex"
    assert output.search_queries[1].agent == "claude"
    assert output.search_queries[0].query == output.search_queries[1].query


def test_merge_carries_results_through() -> None:
    outcomes = [
        _outcome(
            "codex",
            [
                {
                    "query": "q",
                    "results": [
                        {
                            "title": "T",
                            "url": "https://x",
                            "snippet": "S",
                        }
                    ],
                }
            ],
        ),
    ]

    output = merge_outcomes(outcomes, max_findings_per_module=30, segment="mod")

    result = output.search_queries[0].results[0]
    assert (result.title, result.url, result.snippet) == ("T", "https://x", "S")
