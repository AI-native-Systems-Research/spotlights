"""Validation helpers for module deep-research agent output."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from spotlights_engine.schemas.common import StepIssue
from spotlights_engine.schemas.finding import Finding
from spotlights_engine.schemas.pipeline import ModuleDeepResearchOutput

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


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


def _renumber_findings(findings: list[Finding], max_findings: int) -> list[Finding]:
    return [
        finding.model_copy(update={"finding_id": f"find-{idx:04d}"})
        for idx, finding in enumerate(findings[:max_findings], start=1)
    ]


def normalize_module_deep_research_output(
    output: ModuleDeepResearchOutput,
    *,
    max_findings_per_module: int,
) -> ModuleDeepResearchOutput:
    """Cap findings and assign deterministic finding IDs in output order."""
    return output.model_copy(
        update={
            "findings": _renumber_findings(output.findings, max_findings_per_module),
        }
    )


def parse_module_deep_research_output(
    text: str,
    *,
    max_findings_per_module: int = 10,
) -> ModuleDeepResearchOutput:
    """Parse and normalize one agent response into `ModuleDeepResearchOutput`.

    Invalid responses become a recoverable issue with empty findings so callers
    can mark the module degraded without crashing the full repository run.
    """
    try:
        payload: Any = json.loads(_extract_json_object(text))
        parsed = ModuleDeepResearchOutput.model_validate(payload)
    except (json.JSONDecodeError, TypeError, ValueError, ValidationError) as exc:
        return ModuleDeepResearchOutput(
            findings=[],
            issues=[_issue(f"could not parse module_deep_research output: {exc}")],
        )

    return normalize_module_deep_research_output(
        parsed,
        max_findings_per_module=max_findings_per_module,
    )


__all__ = ["normalize_module_deep_research_output", "parse_module_deep_research_output"]
