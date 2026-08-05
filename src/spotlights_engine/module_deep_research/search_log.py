"""Pure markdown renderer for the deep-research search-query log.

Renders `ModuleDeepResearchOutput.search_queries` into a human-diffable
markdown document, grouped by candidate (decision D7) then by agent, with
queries numbered in emission order and each query's results as a bullet list.
Step 3 now runs one survey per candidate inside a single module sidecar, so
queries carry the `candidate_id` they were issued for; grouping by candidate
keeps the one module-level log readable. The output is **byte-stable for equal
input**: candidates and agents appear in first-appearance order and queries keep
their merge order, so `diff run1/…/search_log.md run2/…/search_log.md`
highlights exactly which queries/results changed between runs.

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

# Label used when a query (or the whole log, on the standalone/resume path) has
# no candidate attribution — older sidecars and the standalone parse path leave
# `candidate_id` None.
_NO_CANDIDATE = "(no candidate)"


def _failed_agents(output: ModuleDeepResearchOutput) -> list[str]:
    """Agent names of runners that crashed before producing any result.

    Derived from the recoverable issues the merge recorded; deduped while
    preserving first-appearance order. With N surveys per module the same agent
    name can crash on several candidates, so this collapses them to one line and
    the caller reports the count separately (see `_failed_agent_counts`)."""
    failed: list[str] = []
    for issue in output.issues:
        match = _RUNNER_FAILED_RE.search(issue.message)
        if match:
            name = match.group(1)
            if name not in failed:
                failed.append(name)
    return failed


def _failed_agent_counts(output: ModuleDeepResearchOutput) -> dict[str, int]:
    """How many times each agent crashed across the module's candidate surveys.

    Keyed by agent name; a count > 1 means the runner failed on several
    candidates (decision D7 — otherwise one failure looks like all of them)."""
    counts: dict[str, int] = {}
    for issue in output.issues:
        match = _RUNNER_FAILED_RE.search(issue.message)
        if match:
            name = match.group(1)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _group_by_candidate(
    queries: list[SearchQueryLog],
) -> list[tuple[str, list[SearchQueryLog]]]:
    """Group queries by candidate id, preserving first-appearance order of both
    the candidates and the queries within each candidate. Queries with no
    `candidate_id` are collected under `_NO_CANDIDATE`."""
    order: list[str] = []
    groups: dict[str, list[SearchQueryLog]] = {}
    for query in queries:
        key = query.candidate_id or _NO_CANDIDATE
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(query)
    return [(key, groups[key]) for key in order]


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
    by_candidate = _group_by_candidate(output.search_queries)
    reported_agents = {q.agent for q in output.search_queries}
    candidate_count = len(by_candidate)
    query_count = len(output.search_queries)

    lines: list[str] = [
        f"# Deep-research search log — {qn}",
        "",
        f"_{query_count} queries across {candidate_count} candidate(s)._",
        "",
    ]

    for candidate_key, cand_queries in by_candidate:
        lines.append(f"## Candidate: {candidate_key}")
        lines.append("")
        for agent, agent_queries in _group_by_agent(cand_queries):
            lines.append(f"### Agent: {agent}")
            lines.append("")
            for index, query in enumerate(agent_queries, start=1):
                lines.extend(_render_query(index, query))

    # Report runners that crashed with no captured queries so a missing agent
    # isn't read as "ran zero searches". Because one agent can fail on several
    # candidates, report the crash count rather than one line per (candidate,
    # agent) — the per-candidate issue messages carry the detail.
    failed_counts = _failed_agent_counts(output)
    failed_only = {
        agent: count
        for agent, count in failed_counts.items()
        if agent not in reported_agents
    }
    if failed_only:
        lines.append("## Failed runners")
        lines.append("")
        for agent in _failed_agents(output):
            if agent not in failed_only:
                continue
            count = failed_only[agent]
            suffix = "" if count == 1 else f" ×{count}"
            lines.append(
                f"- {agent}{suffix} — runner failed, no queries captured"
            )
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


__all__ = ["render_search_log_markdown"]
