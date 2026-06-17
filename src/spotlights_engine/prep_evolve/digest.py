"""The shared findings/proposals digest (plan §6).

`render_digest(spec)` produces the compact Markdown block every adapter embeds
verbatim into its native context surface (skydiscover `prompt.system_message`,
CORAL `task.description`, Nous `target_system.description`). This is the one
place all findings land; proposals are cross-referenced to their source
finding by `finding_id`.
"""

from __future__ import annotations

from spotlights_engine.prep_evolve.spec import EvolveSpec, Target


def _target_line(t: Target) -> str:
    if t.scope_kind == "module_main_file":
        role = f" — {t.role}" if t.role else ""
        return f"- `{t.file}` (whole file{role})"
    loc = t.file
    if t.line_start is not None and t.line_end is not None:
        loc = f"{t.file}:{t.line_start}–{t.line_end}"
    impact = f" (impact: {t.estimated_impact})" if t.estimated_impact else ""
    sym = f"`{t.symbol}` in " if t.symbol else ""
    return f"- {sym}{loc}{impact}"


def render_digest(spec: EvolveSpec) -> str:
    """Render the optimization goal / target / findings / proposals digest."""
    lines: list[str] = []

    # --- Optimization goal ---
    lines.append("## Optimization goal")
    goal = spec.objective.goal
    hints = ", ".join(spec.objective.workload_hints)
    if hints:
        goal = f"{goal} — workload: {hints}"
    lines.append(f"{goal} (direction: {spec.objective.direction})")
    lines.append("")

    # --- Target ---
    lines.append("## Target")
    for t in spec.targets:
        lines.append(_target_line(t))
    # Richer detail for the candidate target(s).
    for t in spec.targets:
        if t.scope_kind != "candidate":
            continue
        if t.current_approach:
            lines.append(f"\nCurrent approach: {t.current_approach}")
        if t.evolve_rationale:
            lines.append(f"Why it's worth evolving: {t.evolve_rationale}")
        if t.oracles.correctness:
            lines.append(
                "Correctness oracle: " + ", ".join(t.oracles.correctness)
            )
        if t.oracles.performance:
            lines.append(f"Performance oracle: {t.oracles.performance}")
    lines.append("")

    # --- Research findings ---
    lines.append(f"## Research findings ({len(spec.findings)})")
    if spec.findings:
        for i, f in enumerate(spec.findings, start=1):
            header = f"{i}. {f.title} — {f.source_type}"
            if f.url:
                header += f" — {f.url}"
            lines.append(header)
            detail = f"   Technique: {f.technique_summary}"
            if f.supporting_evidence:
                detail += f' | Evidence: "{f.supporting_evidence}"'
            lines.append(detail)
    else:
        lines.append("_(none)_")
    lines.append("")

    # --- Existing proposals ---
    finding_index = {f.finding_id: i for i, f in enumerate(spec.findings, start=1)}
    lines.append(f"## Existing proposals ({len(spec.proposals)})")
    lines.append("← seeds/inspiration, the evolver may extend or discard")
    if spec.proposals:
        for p in spec.proposals:
            if p.finding_id and p.finding_id in finding_index:
                src = (
                    f" (from finding #{finding_index[p.finding_id]}: "
                    f"{spec.findings[finding_index[p.finding_id] - 1].title})"
                )
            elif p.origin == "agent":
                src = " (agent proposal, no source finding)"
            elif p.finding_id:
                src = f" (from finding {p.finding_id})"
            else:
                src = ""
            lines.append(f"- [{p.agent}] {p.title} — {p.rationale}{src}")
    else:
        lines.append("_(none)_")

    return "\n".join(lines).rstrip() + "\n"


__all__ = ["render_digest"]
