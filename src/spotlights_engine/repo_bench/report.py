"""Build + render the end-to-end run report.

Two artifacts per run, side by side under `runs/repo_bench/<run_id>/`:
  - `report.json` (canonical)
  - `report.md`   (rendered)

The MD is generated from the JSON. Same idiom as bench_spec.py.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from spotlights_engine.repo_bench.schemas import (
    RunReport,
    SnapshotPin,
    StageReport,
)
from spotlights_engine.repo_bench.storage import (
    atomic_write_text,
    runs_root,
    write_json,
)

log = logging.getLogger(__name__)

_TEMPLATE_PATH = (
    Path(__file__).parent / "templates" / "run_report.md.template"
)


def make_run_id(window_id: str, view_id: str) -> str:
    """`bench-<window>__<view>__<UTC-timestamp>`.

    The `bench-` prefix makes run dirs self-identifying when browsing
    `runs/repo_bench/` — distinguishes them from other
    timestamped artifacts that might land near them.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"bench-{window_id}__{view_id}__{ts}"


def write_report(
    *,
    run_id: str,
    window_id: str,
    view_id: str,
    started_at: datetime,
    finished_at: datetime,
    stages: list[StageReport],
    snapshot: SnapshotPin | None,
    bench_spec_path: Path | None,
    next_steps: list[str] | None = None,
    out_root: Path | None = None,
) -> tuple[Path, Path]:
    """Build the RunReport, write JSON + MD, return (json_path, md_path)."""
    report = RunReport(
        run_id=run_id,
        window_id=window_id,
        view_id=view_id,
        started_at=started_at,
        finished_at=finished_at,
        stages=stages,
        snapshot=snapshot,
        bench_spec_path=str(bench_spec_path) if bench_spec_path else None,
        next_steps=next_steps or _default_next_steps(snapshot),
    )

    root = out_root or runs_root()
    out_dir = root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "run_report.json"
    write_json(json_path, report)

    md_path = out_dir / "run_report.md"
    atomic_write_text(md_path, render_md(report))

    log.info("report: wrote %s and %s", json_path, md_path)
    return json_path, md_path


def render_md(report: RunReport) -> str:
    """Render the run report MD from its canonical JSON form."""
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")

    duration_s = (report.finished_at - report.started_at).total_seconds()
    final_status = _final_status(report.stages)

    summary_extras = _summary_extras(report)
    stage_rows = "\n".join(
        f"| {i+1} | `{s.stage}` | {_status_marker(s.status)} {s.status} | {s.detail} |"
        for i, s in enumerate(report.stages)
    )

    snapshot_block = _snapshot_block(report.snapshot)
    bench_spec_block = (
        f"Spec rendered to: `{report.bench_spec_path}`\n\n"
        f"Hand this to the observability bench module."
        if report.bench_spec_path
        else "_(not produced — run did not reach the bench-spec stage)_"
    )
    next_steps_block = "\n".join(f"- {s}" for s in report.next_steps) or "_(none)_"

    return template.format(
        run_id=report.run_id,
        window_id=report.window_id,
        view_id=report.view_id,
        started_at=report.started_at.isoformat(),
        finished_at=report.finished_at.isoformat(),
        duration_s=duration_s,
        final_status=final_status,
        summary_table_extras=summary_extras,
        stage_rows=stage_rows,
        snapshot_block=snapshot_block,
        bench_spec_block=bench_spec_block,
        next_steps_block=next_steps_block,
    )


# ── Internals ─────────────────────────────────────────────────────────


def _final_status(stages: list[StageReport]) -> str:
    if any(s.status == "failed" for s in stages):
        return "FAILED"
    if all(s.status in ("done", "skipped") for s in stages):
        return "OK"
    return "PARTIAL"


def _status_marker(status: str) -> str:
    return {"done": "[OK]", "skipped": "[--]", "failed": "[FAIL]"}.get(status, "[?]")


def _summary_extras(report: RunReport) -> str:
    """Extra rows in the summary table — derived from stage counts."""
    rows: list[str] = []
    for s in report.stages:
        if s.stage == "filter" and "n_kept" in s.counts:
            rows.append(f"| Filtered PRs | {s.counts['n_kept']} kept "
                        f"of {s.counts.get('n_input', '?')} |")
        if s.stage == "snapshot" and report.snapshot is not None:
            short = report.snapshot.snapshot_sha[:12]
            rows.append(f"| Snapshot SHA | `{short}` |")
    return "\n".join(rows)


def _snapshot_block(pin: SnapshotPin | None) -> str:
    if pin is None:
        return "_(not produced — run did not reach the snapshot stage)_"
    return (
        f"- **SHA**: `{pin.snapshot_sha}`\n"
        f"- **PR**: #{pin.snapshot_pr_number} (merged {pin.snapshot_merged_at.isoformat()})\n"
        f"- **Anchor**: PR #{pin.earliest_in_view_pr_number} "
        f"(earliest in view, merged {pin.earliest_in_view_merged_at.isoformat()})\n"
        f"- **Buffer**: {pin.buffer_hours}h\n"
        f"- **View PRs anchored against**: {pin.n_view}\n\n"
        f"{pin.rationale}"
    )


def _default_next_steps(snapshot: SnapshotPin | None) -> list[str]:
    if snapshot is None:
        return [
            "Run did not produce a snapshot. Inspect the failed stage above and re-run.",
        ]
    return [
        "Hand the bench spec (`OBSERVABILITY_BENCH_SPEC.md`) to the "
        "observability bench module.",
        "When their bundle lands, run a discovery method against the "
        "target-repo source at the pinned SHA to produce a `findings.json`.",
        "Run `repo-bench match --findings <path-to-findings.json> "
        f"--window {snapshot.window_id} --view {snapshot.view_id} "
        "--experiment <name>` to grade.",
    ]


__all__ = ["make_run_id", "render_md", "write_report"]
