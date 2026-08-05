"""Validation helpers for module deep-research agent output.

The deep-research agents emit **bare** local finding ids (`find-NNNN`) — they
are unaware of module slugs (decision D3, option A). The persisted schema
`Finding`, by contrast, requires the module-prefixed form
`find-<segment>-NNNN`. To bridge that, the agent output is parsed into the
lenient wire types below (`AgentFinding` / `AgentModuleDeepResearchOutput`), and
`normalize_module_deep_research_output` renumbers + prefixes those into real
`Finding` objects with the manager-supplied segment, before the contract
`ModuleDeepResearchOutput` is returned (and before step 4 consumes findings, so
`Proposal.finding_ref_id` is born already-prefixed).
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding, FindingSourceType
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput
from spotlights_engine.schemas.search import SearchQueryLog, SearchResult
from spotlights_engine.utils.id_helpers import slug_for

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_AGENT_OUTPUT_KEYS = frozenset({"findings", "issues", "search_queries"})


class AgentFinding(BaseModel):
    """Wire shape the deep-research agents emit: a `Finding` with a **bare**
    local `find-NNNN` id. The id is advisory only — it is overwritten by the
    deterministic renumber+prefix in `normalize_module_deep_research_output`."""

    model_config = ConfigDict(extra="forbid")

    finding_id: str = Field(pattern=r"^find-\d{4}$")
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    source_type: FindingSourceType
    technique_summary: str = Field(min_length=1)
    supporting_evidence: str = ""


class AgentSearchResult(BaseModel):
    """Wire shape for one result a search query returned.

    Lenient by design (`extra="ignore"`, no `min_length`): the search log is
    advisory, so a stray extra field must not fail the whole payload and
    discard the runner's real findings — see the strictness rule in the debug
    design doc."""

    model_config = ConfigDict(extra="ignore")  # tolerate extra result fields

    title: str = ""
    url: str = ""
    snippet: str = ""  # short excerpt / description returned by the search


class AgentSearchQuery(BaseModel):
    """Wire shape for one search query an agent reports having issued.

    Lenient by design (`extra="ignore"`, no `min_length`) — an empty query
    must not fail the payload."""

    model_config = ConfigDict(extra="ignore")

    query: str = ""  # NO min_length — empty must not fail the payload
    tool: str = ""  # e.g. "web_search", "web_fetch" — optional, best-effort
    results: list[AgentSearchResult] = Field(default_factory=list)


class AgentModuleDeepResearchOutput(BaseModel):
    """Agent-facing wire output: `AgentFinding`s plus issues. Promoted to the
    persisted `ModuleDeepResearchOutput` (real `Finding`s with prefixed ids) by
    `normalize_module_deep_research_output`."""

    model_config = ConfigDict(extra="forbid")

    findings: list[AgentFinding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)
    search_queries: list[AgentSearchQuery] = Field(default_factory=list)


def _issue(message: str, *, recoverable: bool = True) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="error",
        message=message,
        recoverable=recoverable,
    )


def _iter_json_objects(text: str):
    """Yield every top-level JSON object decodable from `text`, in order.

    Uses `JSONDecoder.raw_decode` to consume one value at a time, skipping the
    whitespace/prose between values. Some runners concatenate the `text` parts of
    a multi-object NDJSON stream, so the payload can arrive as
    `{small preamble}\\n{real findings}` — the whole blob both starts with `{` and
    ends with `}`, which made a single `json.loads` raise `Extra data`. Iterating
    lets the caller pick the object that is actually the payload."""
    decoder = json.JSONDecoder()
    idx = 0
    n = len(text)
    while idx < n:
        next_brace = text.find("{", idx)
        if next_brace == -1:
            return
        try:
            obj, end = decoder.raw_decode(text, next_brace)
        except json.JSONDecodeError:
            # Not a valid object at this brace; advance past it and retry.
            idx = next_brace + 1
            continue
        yield obj
        idx = end


def _select_json_object(text: str, *, fallback_to_first: bool = True) -> str | None:
    first: str | None = None
    first_output_like: str | None = None
    for obj in _iter_json_objects(text):
        if not isinstance(obj, dict):
            continue
        candidate = json.dumps(obj)
        if first is None:
            first = candidate
        if "findings" in obj:
            return candidate
        if first_output_like is None and _AGENT_OUTPUT_KEYS.intersection(obj):
            first_output_like = candidate
    if first_output_like is not None:
        return first_output_like
    if fallback_to_first:
        return first
    return None


def _extract_json_object(text: str) -> str:
    """Return the JSON string that best matches the agent's expected payload.

    Prefers a fenced ```json block that looks like an agent output, then the
    first decodable object carrying a `findings` key (the real payload even when
    a smaller preamble object precedes it in a concatenated stream), then another
    output-shaped object such as issue-only output, then the first decodable
    object as a compatibility fallback."""
    for fenced in _JSON_FENCE.finditer(text):
        selected = _select_json_object(fenced.group(1).strip(), fallback_to_first=False)
        if selected is not None:
            return selected

    selected = _select_json_object(text)
    if selected is not None:
        return selected

    raise ValueError("no JSON object found in module_deep_research response")


_CLI_ENVELOPE_MARKERS = frozenset(
    {"session_id", "subtype", "is_error", "total_cost_usd", "num_turns"}
)


def _looks_like_cli_envelope(payload: Any) -> bool:
    """A `claude -p --output-format json` envelope (or similar CLI wrapper)
    that wasn't unwrapped by the runner. Three or more marker keys is a
    strong signal that this is a transport envelope, not the agent's
    `ModuleDeepResearchOutput` payload."""
    if not isinstance(payload, dict):
        return False
    return len(_CLI_ENVELOPE_MARKERS.intersection(payload)) >= 3


def _renumber_findings(
    findings: list[AgentFinding],
    max_findings: int,
    *,
    segment: str,
    candidate_id: str | None = None,
) -> list[Finding]:
    """Cap, renumber, and prefix wire findings into persisted `Finding`s.

    The agent-supplied (bare, advisory) id is discarded; each surviving finding
    gets a deterministic `find-<segment>-NNNN` id in output order and is stamped
    with `candidate_id` (decision D2). For the per-candidate path (D3) `segment`
    is the composite `<module_segment>-<candidate_counter>` string, so the id
    becomes `find-<module_segment>-<candidate_counter>-NNNN`."""
    return [
        Finding(
            finding_id=f"find-{segment}-{idx:04d}",
            title=finding.title,
            url=finding.url,
            source_type=finding.source_type,
            technique_summary=finding.technique_summary,
            supporting_evidence=finding.supporting_evidence,
            candidate_id=candidate_id,
        )
        for idx, finding in enumerate(findings[:max_findings], start=1)
    ]


def normalize_module_deep_research_output(
    output: AgentModuleDeepResearchOutput,
    *,
    max_findings_per_candidate: int,
    segment: str,
    candidate_id: str | None = None,
    search_queries: list[SearchQueryLog] | None = None,
) -> ModuleDeepResearchOutput:
    """Cap findings and assign deterministic prefixed finding IDs in output
    order, promoting the wire output to the persisted contract.

    `segment` is the per-candidate finding segment (D3); `candidate_id` is
    stamped on every promoted `Finding` (D2). `search_queries` is a passthrough:
    the wire model has no `agent` field (the agent label lives only on
    `RunnerOutcome.agent_name`), so the caller tags each query with its runner
    and forwards the already-built persisted `SearchQueryLog`s here verbatim."""
    return ModuleDeepResearchOutput(
        findings=_renumber_findings(
            output.findings,
            max_findings_per_candidate,
            segment=segment,
            candidate_id=candidate_id,
        ),
        issues=list(output.issues),
        search_queries=list(search_queries or []),
    )


def parse_agent_output(text: str) -> AgentModuleDeepResearchOutput:
    """Parse one agent response into the lenient wire output (bare ids).

    Invalid responses become a recoverable issue with empty findings so callers
    can mark the module degraded without crashing the full repository run. This
    is the per-runner parse; the segment-aware promotion to the persisted
    contract happens once, downstream, in `normalize_module_deep_research_output`
    (see `orchestration.merge_outcomes`)."""
    try:
        payload: Any = json.loads(_extract_json_object(text))
        if _looks_like_cli_envelope(payload):
            return AgentModuleDeepResearchOutput(
                findings=[],
                issues=[
                    _issue(
                        "module_deep_research agent returned a CLI envelope "
                        "without unwrapped content (no usable `result` / "
                        "`structured_output` / `message.content` field)"
                    )
                ],
            )
        return AgentModuleDeepResearchOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        return AgentModuleDeepResearchOutput(
            findings=[],
            issues=[_issue(f"could not parse module_deep_research output: {exc}")],
        )


def parse_module_deep_research_output(
    text: str,
    *,
    max_findings_per_candidate: int = 10,
    segment: str | None = None,
    candidate_id: str | None = None,
) -> ModuleDeepResearchOutput:
    """Parse and normalize one agent response into `ModuleDeepResearchOutput`.

    Convenience for direct/standalone callers: combines `parse_agent_output`
    with the segment-aware normalize. `segment` defaults to `slug_for("module")`
    when omitted (standalone use); the manager always supplies the real one.
    `candidate_id` is stamped on every promoted finding (D2) when supplied.
    """
    seg = segment if segment is not None else slug_for("module")
    parsed = parse_agent_output(text)
    # Promote the wire queries to persisted `SearchQueryLog`s tagged with a
    # synthetic agent label — this standalone path corresponds to a single
    # response, so there is no per-runner name to attach.
    search_queries = [
        SearchQueryLog(
            agent="agent",
            query=q.query,
            tool=q.tool,
            results=[
                SearchResult(title=r.title, url=r.url, snippet=r.snippet)
                for r in q.results
            ],
            candidate_id=candidate_id,
        )
        for q in parsed.search_queries
    ]
    return normalize_module_deep_research_output(
        parsed,
        max_findings_per_candidate=max_findings_per_candidate,
        segment=seg,
        candidate_id=candidate_id,
        search_queries=search_queries,
    )


__all__ = [
    "AgentFinding",
    "AgentModuleDeepResearchOutput",
    "AgentSearchQuery",
    "AgentSearchResult",
    "normalize_module_deep_research_output",
    "parse_agent_output",
    "parse_module_deep_research_output",
]
