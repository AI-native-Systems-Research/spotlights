"""Single-call orchestrator over the per-step APIs.

Wraps `aggregation.scrape`, `filtering.derive`, `diff_fetcher.fetch_diffs`,
`snapshot.pick_snapshot`, `bench_spec.write_spec`, and
`report.write_report` so a caller can run the whole benchmark
end-to-end in one call.

The per-step APIs still exist and remain the documented contract — the
orchestrator is composition, not a replacement.

Each step's existing cache / existence check decides whether real work
happens. A second invocation with identical inputs is a no-op (every
step cache-hits or skips).

Default `through_step` is `bench-spec`. The full pipeline runs
aggregate → filter → fetch-diffs → snapshot → bench-spec, then writes
a `run_report.{json,md}`.

The `match` step is a separate command — it grades a `findings.json`
against the filtered view by direct match against PR diffs. It is NOT
part of `run` because it requires findings produced by an external
discovery method (signal pipeline, deep research, etc.).

Refresh semantics:

  - `refresh=True`        bypass cache for filter / fetch-diffs.
  - `refresh_aggregate`   also re-scrape from GitHub.

Partial runs:

  - `from_step` / `through_step` bound the run.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from spotlights_engine.repo_bench import (
    aggregation,
    bench_spec,
    diff_fetcher,
    filtering,
    report,
    snapshot as snapshot_mod,
    workload_commands,
    workloads,
)
from spotlights_engine.repo_bench.config import (
    RepoBenchConfig,
    compile_filter_patterns,
    compile_workload_patterns,
    load_config,
)
from spotlights_engine.repo_bench.filtering import heuristics as _heuristics
from spotlights_engine.repo_bench.aggregation import ScrapeHandle
from spotlights_engine.repo_bench.diff_fetcher import DiffFetchHandle
from spotlights_engine.repo_bench.filtering.derive import ViewHandle
from spotlights_engine.repo_bench.schemas import SnapshotPin, StageReport
from spotlights_engine.repo_bench.storage import (
    raw_dir,
    runs_root,
    window_id_for,
)
from spotlights_engine.repo_bench.workload_commands import WorkloadCommandsHandle
from spotlights_engine.repo_bench.workloads import WorkloadAnalysisHandle

log = logging.getLogger(__name__)

Step = Literal[
    "aggregate", "filter", "fetch-diffs", "snapshot", "workloads", "bench-spec",
]
_STEP_ORDER: tuple[Step, ...] = (
    "aggregate", "filter", "fetch-diffs", "snapshot", "workloads", "bench-spec",
)
_DEFAULT_THROUGH: Step = "bench-spec"


# ── Result type ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class BenchmarkHandle:
    """Whatever each step produced — None for any step that was skipped."""

    window_id: str
    view_id: str | None
    aggregate: ScrapeHandle | None
    filter: ViewHandle | None
    fetch_diffs: DiffFetchHandle | None
    snapshot: SnapshotPin | None
    workloads: WorkloadAnalysisHandle | None
    workload_commands: WorkloadCommandsHandle | None
    bench_spec_json: Path | None
    bench_spec_md: Path | None
    report_json: Path | None
    report_md: Path | None


# ── Public API ────────────────────────────────────────────────────────


def benchmark(
    *,
    window_start: date | datetime | str,
    window_end: date | datetime | str,
    rules: list[filtering.Rule],
    reference_bundle_name: str = "20260525T202105Z_util0.4_mem16_lru",
    bench_spec_config_notes: str = "",
    refresh: bool = False,
    refresh_aggregate: bool = False,
    from_step: Step = "aggregate",
    through_step: Step = _DEFAULT_THROUGH,
    github_token: str | None = None,
    github_repo: str = aggregation.DEFAULT_REPO,
    data_root_override: Path | None = None,
    runs_root_override: Path | None = None,
    snapshot_buffer_hours: int | None = None,
    workload_llm: bool = False,
    workload_llm_model: str = "sonnet",
    workload_llm_top_n: int = 5,
    config: str | RepoBenchConfig = "vllm",
) -> BenchmarkHandle:
    """Run the benchmark pipeline end-to-end.

    Default flow (through_step="bench-spec"):
        aggregate → filter → fetch-diffs → snapshot → bench-spec →
        write report. Stops there. The user hands the spec to the
        observability bench module.

    Snapshot, bench-spec, and report stages are deterministic (no LLM).

    `data_root_override` overrides where input cache (raw, views) is
    read from / written to. `runs_root_override` overrides where the
    per-run output dir lands. Both default to `data_root()` /
    `runs_root()` from `storage`.
    """
    _validate_step_range(from_step, through_step)
    started_at = datetime.now(timezone.utc)
    stages: list[StageReport] = []

    # Resolve config first; it drives filter + workload patterns.
    cfg = config if isinstance(config, RepoBenchConfig) else load_config(config)
    _heuristics.set_active_config(compile_filter_patterns(cfg))
    workloads.set_active_config(compile_workload_patterns(cfg))
    log.info(
        "benchmark: config=%s (%d models, %d features, %d hardware, %d file_cats)",
        cfg.name,
        len(cfg.workload.models), len(cfg.workload.features),
        len(cfg.workload.hardware), len(cfg.workload.file_categories),
    )

    window_id = window_id_for(window_start, window_end)
    view_id_pre = filtering.view_id_for(rules)
    run_id = report.make_run_id(window_id, view_id_pre)

    runs_base = runs_root_override or runs_root()
    run_dir = runs_base / run_id

    log.info(
        "benchmark: window=%s view=%s run_id=%s from=%s through=%s "
        "refresh=%s refresh_aggregate=%s",
        window_id, view_id_pre, run_id, from_step, through_step,
        refresh, refresh_aggregate,
    )

    handle = _Handle(window_id=window_id, view_id=view_id_pre)

    try:
        # ── 1. aggregate ──────────────────────────────────────────────
        if _step_in_range("aggregate", from_step, through_step):
            t0 = time.monotonic()
            handle.aggregate = _maybe_aggregate(
                window_start=window_start, window_end=window_end,
                github_token=github_token, github_repo=github_repo,
                refresh_aggregate=refresh_aggregate,
                data_root_override=data_root_override,
            )
            stages.append(_stage(
                "aggregate",
                "skipped" if handle.aggregate is None else "done",
                "raw scrape exists; skipped"
                if handle.aggregate is None
                else f"{handle.aggregate.total_prs} PRs scraped",
                {"total_prs": handle.aggregate.total_prs}
                if handle.aggregate is not None else {},
                t0,
            ))
        else:
            if not (raw_dir(window_id, root=data_root_override) / "prs.jsonl").exists():
                raise FileNotFoundError(
                    f"raw scrape missing for {window_id}; cannot start at "
                    f"`{from_step}` without aggregated data"
                )

        if through_step == "aggregate":
            return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

        # ── 2. filter ────────────────────────────────────────────────
        if _step_in_range("filter", from_step, through_step):
            t0 = time.monotonic()
            run_dir.mkdir(parents=True, exist_ok=True)
            handle.filter = filtering.derive(
                window_id=window_id, rules=rules, run_dir=run_dir,
                data_root_override=data_root_override,
            )
            stages.append(_stage(
                "filter", "done",
                f"{handle.filter.n_kept} kept of "
                f"{handle.filter.n_kept + handle.filter.n_dropped}",
                {
                    "n_input": handle.filter.n_kept + handle.filter.n_dropped,
                    "n_kept": handle.filter.n_kept,
                    "n_dropped": handle.filter.n_dropped,
                },
                t0,
            ))

        view_id = (
            handle.filter.view_id
            if handle.filter is not None
            else filtering.view_id_for(rules)
        )
        handle.view_id = view_id
        view_path = run_dir / "view" / "prs.jsonl"

        if through_step == "filter":
            return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

        # ── 3. fetch-diffs ──────────────────────────────────────────
        if _step_in_range("fetch-diffs", from_step, through_step):
            t0 = time.monotonic()
            token = github_token or os.environ.get("GITHUB_TOKEN")
            if not token:
                raise ValueError(
                    "fetch-diffs requires a GitHub token "
                    "(pass `github_token=` or set $GITHUB_TOKEN)"
                )
            handle.fetch_diffs = diff_fetcher.fetch_diffs(
                window_id=window_id, view_path=view_path, token=token,
                repo=github_repo, refresh=refresh,
                data_root_override=data_root_override,
            )
            stages.append(_stage(
                "fetch-diffs", "done",
                f"{handle.fetch_diffs.n_fetched} fetched, "
                f"{handle.fetch_diffs.n_skipped} skipped",
                {
                    "n_total": handle.fetch_diffs.n_total,
                    "n_fetched": handle.fetch_diffs.n_fetched,
                    "n_skipped": handle.fetch_diffs.n_skipped,
                },
                t0,
            ))

        if through_step == "fetch-diffs":
            return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

        # ── 4. snapshot ──────────────────────────────────────────────
        if _step_in_range("snapshot", from_step, through_step):
            t0 = time.monotonic()
            run_dir.mkdir(parents=True, exist_ok=True)
            handle.snapshot = snapshot_mod.pick_snapshot(
                window_id=window_id, view_id=view_id,
                view_path=view_path, run_dir=run_dir,
                data_root_override=data_root_override,
                buffer_hours=snapshot_buffer_hours,
            )
            stages.append(_stage(
                "snapshot", "done",
                f"pinned SHA {handle.snapshot.snapshot_sha[:12]} "
                f"(PR #{handle.snapshot.snapshot_pr_number})",
                {"n_view": handle.snapshot.n_view},
                t0,
            ))

        if through_step == "snapshot":
            return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

        # ── 5. workloads ─────────────────────────────────────────────
        if _step_in_range("workloads", from_step, through_step):
            t0 = time.monotonic()
            run_dir.mkdir(parents=True, exist_ok=True)
            handle.workloads = workloads.analyze(
                window_id=window_id, view_path=view_path, run_dir=run_dir,
                data_root_override=data_root_override,
            )
            stages.append(_stage(
                "workloads", "done",
                f"{handle.workloads.n_prs} PRs analyzed, "
                f"{handle.workloads.portfolio_size}-entry portfolio",
                {
                    "n_prs": handle.workloads.n_prs,
                    "portfolio_size": handle.workloads.portfolio_size,
                },
                t0,
            ))

        if through_step == "workloads":
            return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

        # ── 5b. workload_commands (opt-in, LLM) ──────────────────────
        if workload_llm and _step_in_range("workloads", from_step, through_step):
            t0 = time.monotonic()
            handle.workload_commands = workload_commands.extract(
                window_id=window_id,
                view_path=view_path,
                run_dir=run_dir,
                judge_model=workload_llm_model,
                data_root_override=data_root_override,
                top_n=workload_llm_top_n,
            )
            stages.append(_stage(
                "workloads", "done",
                f"LLM extracted commands from {handle.workload_commands.n_prs_extracted} PRs, "
                f"{handle.workload_commands.n_clusters} clusters",
                {
                    "n_prs_extracted": handle.workload_commands.n_prs_extracted,
                    "n_prs_with_commands": handle.workload_commands.n_prs_with_commands,
                    "n_clusters": handle.workload_commands.n_clusters,
                },
                t0,
            ))

        # ── 6. bench-spec ────────────────────────────────────────────
        if _step_in_range("bench-spec", from_step, through_step):
            if handle.snapshot is None:
                raise RuntimeError(
                    "bench-spec stage reached but no snapshot pin available"
                )
            t0 = time.monotonic()
            run_dir.mkdir(parents=True, exist_ok=True)
            workload_summary_md: str | None = None
            workload_portfolio_md: str | None = None
            if handle.workloads is not None and handle.workloads.md_path.exists():
                workload_summary_md = handle.workloads.md_path.read_text(
                    encoding="utf-8"
                )
                # Default §6 source: regex-extracted runnable portfolio.
                workload_portfolio_md = handle.workloads.runnable_portfolio_md
            # Opt-in: LLM-extracted portfolio overrides regex when present.
            if handle.workload_commands is not None:
                workload_portfolio_md = handle.workload_commands.portfolio_md
            handle.bench_spec_json, handle.bench_spec_md = bench_spec.write_spec(
                snapshot=handle.snapshot,
                reference_bundle_name=reference_bundle_name,
                run_dir=run_dir,
                config_notes=bench_spec_config_notes,
                workload_summary_md=workload_summary_md,
                workload_commands_md=workload_portfolio_md,
            )
            stages.append(_stage(
                "bench-spec", "done",
                f"spec at {handle.bench_spec_md.name}",
                {},
                t0,
            ))

        return _finalize(handle, started_at, stages, window_id, run_id, run_dir)

    except Exception as e:  # noqa: BLE001
        if not stages or stages[-1].status == "done":
            failed_stage = _next_stage_after(stages, through_step)
            if failed_stage is not None:
                stages.append(StageReport(
                    stage=failed_stage,  # type: ignore[arg-type]
                    status="failed",
                    detail=f"{type(e).__name__}: {str(e)[:120]}",
                ))
        try:
            _finalize(handle, started_at, stages, window_id, run_id, run_dir)
        except Exception as inner:  # noqa: BLE001
            log.warning("report write failed during error path: %s", inner)
        raise


# ── Internals ─────────────────────────────────────────────────────────


@dataclass
class _Handle:
    """Mutable accumulator; copied into the frozen BenchmarkHandle."""
    window_id: str
    view_id: str | None = None
    aggregate: ScrapeHandle | None = None
    filter: ViewHandle | None = None
    fetch_diffs: DiffFetchHandle | None = None
    snapshot: SnapshotPin | None = None
    workload_commands: WorkloadCommandsHandle | None = None
    workloads: WorkloadAnalysisHandle | None = None
    bench_spec_json: Path | None = None
    bench_spec_md: Path | None = None


def _stage(
    stage: str, status: str, detail: str,
    counts: dict, t0: float,
) -> StageReport:
    return StageReport(
        stage=stage,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        detail=detail,
        counts=counts,
        duration_s=time.monotonic() - t0,
    )


def _finalize(
    handle: _Handle,
    started_at: datetime,
    stages: list[StageReport],
    window_id: str,
    run_id: str,
    run_dir: Path,
) -> BenchmarkHandle:
    """Always write a run_report.{json,md} and return the frozen handle."""
    report_json: Path | None = None
    report_md: Path | None = None
    if stages:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            report_json, report_md = report.write_report(
                run_id=run_id,
                window_id=window_id,
                view_id=handle.view_id or "noview",
                started_at=started_at,
                finished_at=datetime.now(timezone.utc),
                stages=stages,
                snapshot=handle.snapshot,
                bench_spec_path=handle.bench_spec_md,
                out_root=run_dir.parent,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("could not write run report: %s", e)

    return BenchmarkHandle(
        window_id=window_id,
        view_id=handle.view_id,
        aggregate=handle.aggregate,
        filter=handle.filter,
        fetch_diffs=handle.fetch_diffs,
        snapshot=handle.snapshot,
        workloads=handle.workloads,
        workload_commands=handle.workload_commands,
        bench_spec_json=handle.bench_spec_json,
        bench_spec_md=handle.bench_spec_md,
        report_json=report_json,
        report_md=report_md,
    )


def _next_stage_after(stages: list[StageReport], through_step: Step) -> str | None:
    done_stage_names = {s.stage for s in stages}
    through_idx = _STEP_ORDER.index(through_step)
    for i, name in enumerate(_STEP_ORDER):
        if i > through_idx:
            return None
        if name not in done_stage_names:
            return name
    return None


def _validate_step_range(from_step: Step, through_step: Step) -> None:
    if from_step not in _STEP_ORDER:
        raise ValueError(f"unknown from_step: {from_step!r}")
    if through_step not in _STEP_ORDER:
        raise ValueError(f"unknown through_step: {through_step!r}")
    if _STEP_ORDER.index(from_step) > _STEP_ORDER.index(through_step):
        raise ValueError(
            f"from_step ({from_step!r}) must come before or equal "
            f"through_step ({through_step!r}) in the order: "
            f"{', '.join(_STEP_ORDER)}"
        )


def _step_in_range(step: Step, from_step: Step, through_step: Step) -> bool:
    i = _STEP_ORDER.index(step)
    return _STEP_ORDER.index(from_step) <= i <= _STEP_ORDER.index(through_step)


def _maybe_aggregate(
    *,
    window_start: date | datetime | str,
    window_end: date | datetime | str,
    github_token: str | None,
    github_repo: str,
    refresh_aggregate: bool,
    data_root_override: Path | None,
) -> ScrapeHandle | None:
    window_id = window_id_for(window_start, window_end)
    prs_path = raw_dir(window_id, root=data_root_override) / "prs.jsonl"

    if prs_path.exists() and not refresh_aggregate:
        log.info(
            "benchmark: aggregate skipped — raw exists at %s "
            "(use refresh_aggregate=True to re-scrape)",
            prs_path,
        )
        return None

    token = github_token or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "aggregate requires a GitHub token "
            "(pass `github_token=` or set $GITHUB_TOKEN)"
        )

    return aggregation.scrape(
        window_start=window_start, window_end=window_end,
        token=token, repo=github_repo, out_root=data_root_override,
    )


__all__ = ["BenchmarkHandle", "Step", "benchmark"]
