"""Merge two `SpotlightReport`s into one with `RunInfo.pipeline="unified"`.

Field-by-field rules (see plan §"Merging two SpotlightReports"):

- `project_tree`: must be equal between the two reports (the unified runner
  prepopulates the same `ProjectTree` to both sub-pipelines, so by-construction
  equality holds).  Defensive `==` check — surfaces a divergence as a clear
  error rather than silently picking one side.
- `context`: must be equal (unified runner threads the same `SpotlightContext`
  to both pipelines).
- `candidates`, `findings`, `anomalies`: concatenated; `id` uniqueness is
  re-checked because the segmented id pattern doesn't by itself guarantee
  disjointness.  A DR module slug literally named `signal` would collide with
  the signal pipeline's hard-coded `signal` segment.  Collision raises
  `MergeIdCollisionError` rather than producing a report with non-unique ids.
- `issues`: concatenated.
- `run`: minted fresh (`pipeline="unified"`, run_id supplied by the caller).
  Cost handling: every contributor that supplied a `cost_usd` is summed; the
  result is `None` only when all contributors were `None`.
"""

from __future__ import annotations

from typing import Iterable, TypeVar

from spotlights_engine.schemas.pipeline import RunInfo, SpotlightReport
from spotlights_engine.unified_runner.errors import MergeIdCollisionError


T = TypeVar("T")


def _check_unique(items: list[T], *, key, kind: str) -> None:
    seen: dict[str, str] = {}
    for item in items:
        item_id = key(item)
        if item_id in seen:
            raise MergeIdCollisionError(
                f"duplicate {kind} id {item_id!r} after merge "
                f"(first seen in {seen[item_id]}, again here)"
            )
        seen[item_id] = kind


def _sum_costs(values: Iterable[float | None]) -> float | None:
    total = 0.0
    saw_any = False
    for v in values:
        if v is None:
            continue
        total += float(v)
        saw_any = True
    return round(total, 4) if saw_any else None


def _earliest(*values: str | None) -> str:
    """Pick the lexicographically smallest non-empty ISO timestamp.

    Both upstream pipelines emit ISO-8601 strings, so lexicographic ordering
    matches chronological ordering for same-zone timestamps.
    """
    candidates = [v for v in values if v]
    if not candidates:
        raise ValueError("merge_reports requires at least one started_at")
    return min(candidates)


def _latest(*values: str | None) -> str | None:
    candidates = [v for v in values if v]
    return max(candidates) if candidates else None


def merge_reports(
    *,
    signal: SpotlightReport | None,
    dr: SpotlightReport | None,
    run_id: str,
) -> SpotlightReport:
    """Merge two `SpotlightReport`s into one with `pipeline="unified"`.

    Either side may be `None` (`mode="signal"` skips DR; `mode="dr"` skips
    signal).  At least one must be non-None.

    Caller supplies the `run_id`; the unified runner derives it from a hash
    of the `UnifiedInput` so it's resume-stable.
    """
    if signal is None and dr is None:
        raise ValueError("merge_reports requires at least one report")

    # Pick the canonical project_tree + context + run-anchor side.
    primary = dr if dr is not None else signal
    other = signal if dr is not None else None
    assert primary is not None  # narrows the type for the checker

    if other is not None:
        if primary.project_tree != other.project_tree:
            raise ValueError(
                "merge_reports: project_tree differs between reports — extract-once "
                "should keep them equal"
            )
        if primary.context != other.context:
            raise ValueError(
                "merge_reports: context differs between reports — unified runner "
                "should plumb the same SpotlightContext to both pipelines"
            )

    candidates = list(primary.candidates) + (
        list(other.candidates) if other is not None else []
    )
    findings = list(primary.findings) + (
        list(other.findings) if other is not None else []
    )
    anomalies = list(primary.anomalies) + (
        list(other.anomalies) if other is not None else []
    )
    issues = list(primary.issues) + (list(other.issues) if other is not None else [])

    _check_unique(candidates, key=lambda c: c.id, kind="candidate")
    _check_unique(findings, key=lambda f: f.finding_id, kind="finding")
    _check_unique(anomalies, key=lambda a: a.anomaly_id, kind="anomaly")

    started_at = _earliest(
        *(r.run.started_at for r in (signal, dr) if r is not None)
    )
    finished_at = _latest(
        *(r.run.finished_at for r in (signal, dr) if r is not None)
    )
    cost = _sum_costs(r.run.cost_usd for r in (signal, dr) if r is not None)

    run_info = RunInfo(
        pipeline="unified",
        run_id=run_id,
        started_at=started_at,
        finished_at=finished_at,
        cost_usd=cost,
    )

    return SpotlightReport(
        project_tree=primary.project_tree,
        context=primary.context,
        candidates=candidates,
        findings=findings,
        anomalies=anomalies,
        run=run_info,
        issues=issues,
    )


__all__ = ["merge_reports"]
