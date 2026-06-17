"""Iteration telemetry construction and structural-diff helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spotlights_engine.candidate_discovery.api import IterationTelemetry
from spotlights_engine.schemas.legacy.candidate import Candidates

if TYPE_CHECKING:  # pragma: no cover
    from spotlights_engine.candidate_discovery.agents import AgentInvocation
    from spotlights_engine.candidate_discovery.validation import DropCounters


class TelemetryBuilder:
    def build(
        self,
        n: int,
        agent: str,
        inv: AgentInvocation,
        survivors: Candidates,
        drops: DropCounters,
        prev: Candidates | None,
        schema_retries: int,
        iteration_duration_s: float,
    ) -> IterationTelemetry:
        added, removed, modified = _compute_diff(prev, survivors)
        return IterationTelemetry(
            n=n,
            agent=agent,  # type: ignore[arg-type]
            session_id=inv.session_id,
            duration_s=iteration_duration_s,
            cost_usd=inv.cost_usd,
            input_tokens=inv.input_tokens,
            output_tokens=inv.output_tokens,
            candidate_count=len(survivors.candidates),
            schema_retries=schema_retries,
            dropped_outside_module=drops.dropped_outside_module,
            dropped_missing_file=drops.dropped_missing_file,
            dropped_invalid_ranges=drops.dropped_invalid_ranges,
            added=added,
            removed=removed,
            modified=modified,
        )


def _compute_diff(
    prev: Candidates | None, current: Candidates
) -> tuple[list[str], list[str], list[str]]:
    current_by_id = {c.id: c for c in current.candidates}
    current_ids = set(current_by_id)

    if prev is None:
        return sorted(current_ids), [], []

    prev_by_id = {c.id: c for c in prev.candidates}
    prev_ids = set(prev_by_id)

    added = sorted(current_ids - prev_ids)
    removed = sorted(prev_ids - current_ids)
    modified = sorted(
        cid
        for cid in current_ids & prev_ids
        if _materially_differs(prev_by_id[cid], current_by_id[cid])
    )
    return added, removed, modified


def _materially_differs(a, b) -> bool:
    return (
        a.line_start != b.line_start
        or a.line_end != b.line_end
        or a.symbol.strip() != b.symbol.strip()
        or a.kind != b.kind
        or a.estimated_impact != b.estimated_impact
        or a.description.strip() != b.description.strip()
        or a.current_approach.strip() != b.current_approach.strip()
        or a.evolve_rationale.strip() != b.evolve_rationale.strip()
        or a.estimated_impact_explanation.strip()
        != b.estimated_impact_explanation.strip()
    )


def render_diff_markdown(prev: Candidates | None, current: Candidates) -> str:
    """Render `diff_from_prev.md` for a single iteration."""
    added, removed, modified = _compute_diff(prev, current)
    current_by_id = {c.id: c for c in current.candidates}
    prev_by_id = {c.id: c for c in (prev.candidates if prev else [])}

    def _added_line(cid: str) -> str:
        c = current_by_id[cid]
        return (
            f"- {cid} — {c.file}:{c.line_start}-{c.line_end} "
            f"[{c.kind}] {c.symbol} — {c.evolve_rationale}"
        )

    def _removed_line(cid: str) -> str:
        c = prev_by_id[cid]
        return (
            f"- {cid} — {c.file}:{c.line_start}-{c.line_end} "
            f"[{c.kind}] {c.symbol} — (was: {c.evolve_rationale})"
        )

    def _modified_line(cid: str) -> str:
        prev_c = prev_by_id[cid]
        cur_c = current_by_id[cid]
        return (
            f"- {cid} — {cur_c.file}:"
            f"{prev_c.line_start}-{prev_c.line_end} → "
            f"{cur_c.line_start}-{cur_c.line_end} "
            f"[{cur_c.kind}] {cur_c.symbol} — {cur_c.evolve_rationale}"
        )

    def _section(title: str, lines: list[str]) -> str:
        body = "\n".join(lines) if lines else "_(none)_"
        return f"## {title}\n{body}"

    return "\n\n".join(
        [
            _section("Added", [_added_line(cid) for cid in added]),
            _section("Removed", [_removed_line(cid) for cid in removed]),
            _section("Modified", [_modified_line(cid) for cid in modified]),
        ]
    ) + "\n"


__all__ = ["TelemetryBuilder", "render_diff_markdown"]
