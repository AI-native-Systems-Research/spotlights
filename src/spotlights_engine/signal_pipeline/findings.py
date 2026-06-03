"""Roll up stage 03 candidates + stage 04 change specs into `findings.{json,md}`.

Called best-effort by the runner at the end of a pipeline invocation. Writes
two artifacts next to the run dir's stage outputs:

- `findings.json` — single JSON array, one entry per candidate, each
  shaped `{"candidate": <stage 03 candidate>, "change": <stage 04 spec>|None}`.
  Stable, machine-friendly join of the two stages.
- `findings.md` — same data, rendered as a summary table + per-candidate
  sections suitable for sharing with humans.

Idempotent: every call overwrites both files. No-ops if `03_candidates.json`
is missing (e.g. the run was scoped to stages 01-02). Stage 04 is optional
per candidate; missing change specs render as `_(stage 04 not run …)_`.

Failures are caught by the caller (`runner.run_pipeline`) and surfaced as
issues — they never abort the run.
"""

from __future__ import annotations

import json
from typing import Any

from spotlights_engine.signal_pipeline.layout import (
    RunDirLayout,
    atomic_write_json,
    atomic_write_text,
)


__all__ = ["emit_findings"]


def emit_findings(layout: RunDirLayout) -> None:
    """Write `findings.json` + `findings.md` joining 03 candidates with 04 changes.

    Idempotent — overwrites on every invocation. No-ops if 03_candidates.json
    is missing.
    """
    candidates_path = layout.stage_artifact("03", shape="single")
    if not candidates_path.exists():
        return

    raw = json.loads(candidates_path.read_text(encoding="utf-8"))
    candidates = (
        raw["candidates"] if isinstance(raw, dict) and "candidates" in raw else raw
    )

    changes_dir = layout.root / "04_changes"
    findings: list[dict[str, Any]] = []
    for c in candidates:
        cid = c.get("id")
        change_path = changes_dir / f"{cid}.json"
        change = (
            json.loads(change_path.read_text(encoding="utf-8"))
            if change_path.exists()
            else None
        )
        findings.append({"candidate": c, "change": change})

    atomic_write_json(layout.root / "findings.json", findings)
    atomic_write_text(layout.root / "findings.md", _render_markdown(layout, findings))


def _render_markdown(layout: RunDirLayout, findings: list[dict[str, Any]]) -> str:
    md: list[str] = []
    md.append(f"# Findings — `{layout.root.name}`\n")
    md.append(f"_{len(findings)} candidates_ from this pipeline run.\n")
    md.append("## Summary\n")
    md.append("| ID | Impact | File:Lines | Symbol | One-line |")
    md.append("|---|---|---|---|---|")
    for f in findings:
        c = f["candidate"]
        symbol = c.get("symbol", "?")
        kind = c.get("kind", "")
        loc = (
            f"`{c.get('file', '?')}:{c.get('line_start', '?')}"
            f"-{c.get('line_end', '?')}`"
        )
        impact = c.get("estimated_impact", "?")
        desc = (c.get("description") or "").split(".")[0][:120]
        md.append(
            f"| {c.get('id')} | {impact} | {loc} | "
            f"`{symbol}` ({kind}) | {desc} |"
        )
    md.append("")

    for f in findings:
        c = f["candidate"]
        chg = f["change"]
        md.append(f"## {c.get('id')} — `{c.get('symbol', '?')}`")
        md.append(
            f"- **Location:** `{c.get('file', '?')}:"
            f"{c.get('line_start', '?')}-{c.get('line_end', '?')}`"
        )
        md.append(f"- **Kind:** {c.get('kind', '?')}")
        md.append(f"- **Estimated impact:** {c.get('estimated_impact', '?')}")
        md.append("")
        md.append("### Description")
        md.append(c.get("description") or "_(none)_")
        md.append("")
        if c.get("current_approach"):
            md.append("### Current approach")
            md.append(c["current_approach"])
            md.append("")
        if c.get("estimated_impact_explanation"):
            md.append("### Why this matters")
            md.append(c["estimated_impact_explanation"])
            md.append("")
        if c.get("evolve_rationale"):
            md.append("### Rationale")
            md.append(c["evolve_rationale"])
            md.append("")
        if chg:
            md.append(f"### Proposed change ({chg.get('change_type', '?')})")
            md.append("**Mechanism:**")
            md.append(chg.get("mechanism", "_(none)_"))
            md.append("")
            md.append("**Required changes:**")
            md.append(chg.get("required_changes", "_(none)_"))
            md.append("")
            md.append("**Expected effect:**")
            md.append(chg.get("expected_effect", "_(none)_"))
            md.append("")
            md.append("**Evaluation:**")
            md.append(chg.get("evaluation_metric", "_(none)_"))
            md.append("")
        else:
            md.append("### Proposed change")
            md.append("_(stage 04 not run for this candidate)_")
            md.append("")
        md.append("---\n")

    return "\n".join(md)
