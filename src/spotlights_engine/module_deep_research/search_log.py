"""Pure markdown renderer for the deep-research search-query log.

Renders `ModuleDeepResearchOutput.search_queries` into a human-diffable
markdown document, grouped by agent, with queries numbered in emission order
and each query's results as a bullet list. The output is **byte-stable for
equal input**: agents appear in first-appearance order and queries keep their
merge order, so `diff run1/…/search_log.md run2/…/search_log.md` highlights
exactly which queries/results changed between runs.

Kept free of any `spotlights_manager` import so persistence can call it without
an import cycle.
"""

from __future__ import annotations

import re

from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.search import SearchQueryLog

# Matches the issue message a crashed runner leaves behind (see
# `orchestration.merge_outcomes`): the runner produced no result, so it
# contributes no queries. We surface it so a missing agent isn't mistaken for
# "ran zero searches".
_RUNNER_FAILED_RE = re.compile(r"module_deep_research (\S+) execution failed")


def _failed_agents(output: ModuleDeepResearchOutput) -> list[str]:
    """Agent names of runners that crashed before producing any result.

    Derived from the recoverable issues the merge recorded; deduped while
    preserving first-appearance order for stable rendering."""
    failed: list[str] = []
    for issue in output.issues:
        match = _RUNNER_FAILED_RE.search(issue.message)
        if match:
            name = match.group(1)
            if name not in failed:
                failed.append(name)
    return failed


def _group_by_agent(
    queries: list[SearchQueryLog],
) -> list[tuple[str, list[SearchQueryLog]]]:
    """Group queries by agent, preserving first-appearance order of both the
    agents and the queries within each agent."""
    order: list[str] = []
    groups: dict[str, list[SearchQueryLog]] = {}
    for query in queries:
        if query.agent not in groups:
            groups[query.agent] = []
            order.append(query.agent)
        groups[query.agent].append(query)
    return [(agent, groups[agent]) for agent in order]


def _render_query(index: int, query: SearchQueryLog) -> list[str]:
    header = f"### {index}. `{query.query}`"
    if query.tool:
        header += f"  (tool: {query.tool})"
    lines = [header, ""]
    if not query.results:
        lines.append("- _(no results returned)_")
    else:
        for result in query.results:
            title = result.title or result.url or "(untitled)"
            if result.url:
                bullet = f"- [{title}]({result.url})"
            else:
                bullet = f"- {title}"
            if result.snippet:
                bullet += f" — {result.snippet}"
            lines.append(bullet)
    lines.append("")
    return lines


def render_search_log_markdown(
    output: ModuleDeepResearchOutput, *, qn: str
) -> str:
    """Render the search log for one module's deep-research output.

    `qn` is the module qualified name (threaded from the orchestrator, since
    neither `ModulePaths` nor `ModuleDeepResearchOutput` carries it)."""
    grouped = _group_by_agent(output.search_queries)
    reported = {agent for agent, _ in grouped}
    failed_agents = [agent for agent in _failed_agents(output) if agent not in reported]
    agent_count = len(grouped) + len(failed_agents)
    query_count = len(output.search_queries)

    lines: list[str] = [
        f"# Deep-research search log — {qn}",
        "",
        f"_{query_count} queries across {agent_count} agents._",
        "",
    ]

    for agent, queries in grouped:
        lines.append(f"## Agent: {agent}")
        lines.append("")
        for index, query in enumerate(queries, start=1):
            lines.extend(_render_query(index, query))

    # List runners that crashed with no captured queries so a missing agent
    # isn't read as "ran zero searches".
    for agent in failed_agents:
        lines.append(f"## Agent: {agent}")
        lines.append("")
        lines.append("_(runner failed — no queries captured)_")
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = ["render_search_log_markdown"]
