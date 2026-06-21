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
from spotlights_engine.utils.id_helpers import slug_for

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


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


class AgentModuleDeepResearchOutput(BaseModel):
    """Agent-facing wire output: `AgentFinding`s plus issues. Promoted to the
    persisted `ModuleDeepResearchOutput` (real `Finding`s with prefixed ids) by
    `normalize_module_deep_research_output`."""

    model_config = ConfigDict(extra="forbid")

    findings: list[AgentFinding] = Field(default_factory=list)
    issues: list[StepIssue] = Field(default_factory=list)


def _issue(message: str, *, recoverable: bool = True) -> StepIssue:
    return StepIssue(
        step="module_deep_research",
        severity="error",
        message=message,
        recoverable=recoverable,
    )


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    fenced = _JSON_FENCE.search(text)
    if fenced:
        return fenced.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]

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
    findings: list[AgentFinding], max_findings: int, *, segment: str
) -> list[Finding]:
    """Cap, renumber, and prefix wire findings into persisted `Finding`s.

    The agent-supplied (bare, advisory) id is discarded; each surviving finding
    gets a deterministic `find-<segment>-NNNN` id in output order."""
    return [
        Finding(
            finding_id=f"find-{segment}-{idx:04d}",
            title=finding.title,
            url=finding.url,
            source_type=finding.source_type,
            technique_summary=finding.technique_summary,
            supporting_evidence=finding.supporting_evidence,
        )
        for idx, finding in enumerate(findings[:max_findings], start=1)
    ]


def normalize_module_deep_research_output(
    output: AgentModuleDeepResearchOutput,
    *,
    max_findings_per_module: int,
    segment: str,
) -> ModuleDeepResearchOutput:
    """Cap findings and assign deterministic module-prefixed finding IDs in
    output order, promoting the wire output to the persisted contract."""
    return ModuleDeepResearchOutput(
        findings=_renumber_findings(
            output.findings, max_findings_per_module, segment=segment
        ),
        issues=list(output.issues),
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
    max_findings_per_module: int = 30,
    segment: str | None = None,
) -> ModuleDeepResearchOutput:
    """Parse and normalize one agent response into `ModuleDeepResearchOutput`.

    Convenience for direct/standalone callers: combines `parse_agent_output`
    with the segment-aware normalize. `segment` defaults to `slug_for("module")`
    when omitted (standalone use); the manager always supplies the real one.
    """
    seg = segment if segment is not None else slug_for("module")
    return normalize_module_deep_research_output(
        parse_agent_output(text),
        max_findings_per_module=max_findings_per_module,
        segment=seg,
    )


__all__ = [
    "AgentFinding",
    "AgentModuleDeepResearchOutput",
    "normalize_module_deep_research_output",
    "parse_agent_output",
    "parse_module_deep_research_output",
]
