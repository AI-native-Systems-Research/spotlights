"""CLI entry point for spotlights-validation."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import click

from spotlights_validation.execution.runner import run_validation_plan
from spotlights_validation.schemas import ValidationPlan


@click.group()
def cli() -> None:
    """spotlights-validation — validate changes against a target system."""


@cli.command()
@click.option(
    "--plan",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to validation_plan.json produced by the planning step.",
)
@click.option(
    "--source-tree",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root of the target repo to run tests against.",
)
@click.option(
    "--change-ref",
    default=None,
    help="Optional reference identifying the change under test (stored in result).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Log commands without executing them.",
)
@click.option(
    "--out",
    default=None,
    type=click.Path(path_type=Path),
    help="Write ValidationResult JSON to this path.",
)
@click.option(
    "--timeout-multiplier",
    default=2.0,
    show_default=True,
    help="Multiply each entry's estimated_duration by this factor for the subprocess timeout.",
)
@click.option(
    "--logs-dir",
    default=None,
    type=click.Path(path_type=Path),
    help=(
        "Directory for per-entry stdout/stderr logs and pytest-json reports. "
        "Defaults to '<out-parent>/logs/<timestamp>' when --out is set, else "
        "'./logs/<timestamp>'."
    ),
)
@click.option(
    "--indexes",
    default=None,
    help=(
        "Comma-separated list of entry indexes to execute (e.g. '1,3,5'). "
        "Indexes match the [X/total] number shown in the runner output, "
        "which is entry.index when set, else position in priority order "
        "starting at 1. All other entries are dropped from the plan."
    ),
)
def run(
    plan: Path,
    source_tree: Path,
    change_ref: str | None,
    dry_run: bool,
    out: Path | None,
    timeout_multiplier: float,
    logs_dir: Path | None,
    indexes: str | None,
) -> None:
    """Execute a validation plan against a source tree."""
    vplan = ValidationPlan.from_json(plan)

    if indexes is not None:
        vplan = _filter_by_indexes(vplan, indexes)

    if change_ref:
        vplan = vplan.model_copy(update={"change_ref": change_ref})

    if logs_dir is None:
        stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        base = out.parent if out else Path.cwd()
        logs_dir = base / "logs" / stamp

    total = len(vplan.entries)
    halt_count = sum(1 for e in vplan.entries if e.halt_on_failure)
    print(
        f"\nValidation plan: {total} entries  "
        f"({halt_count} halt-on-failure)  "
        f"source-tree: {source_tree}\n"
    )

    result = run_validation_plan(
        vplan,
        source_tree,
        dry_run=dry_run,
        timeout_multiplier=timeout_multiplier,
        logs_dir=logs_dir,
    )

    _print_summary(result)

    if out:
        result.to_json(out)
        print(f"\nResult written to {out}")

    raise SystemExit(0 if result.verdict == "pass" else 1)


def _parse_indexes(raw: str) -> list[int]:
    try:
        values = [int(p.strip()) for p in raw.split(",") if p.strip()]
    except ValueError as exc:
        raise click.BadParameter(
            f"--indexes must be a comma-separated list of integers, got {raw!r}"
        ) from exc
    if not values:
        raise click.BadParameter("--indexes is empty")
    return values


def _filter_by_indexes(vplan: ValidationPlan, raw: str) -> ValidationPlan:
    wanted = set(_parse_indexes(raw))
    sorted_entries = sorted(vplan.entries, key=lambda e: e.priority)
    kept: list = []
    seen: set[int] = set()
    for fallback_index, entry in enumerate(sorted_entries, start=1):
        idx = entry.index if entry.index is not None else fallback_index
        if idx in wanted:
            kept.append(entry)
            seen.add(idx)

    missing = wanted - seen
    if missing:
        raise click.BadParameter(
            f"--indexes references entries not in the plan: {sorted(missing)}"
        )

    return vplan.model_copy(update={"entries": kept})


def _print_summary(result) -> None:  # type: ignore[no-untyped-def]
    print("\n" + "─" * 72)
    print(f"  Verdict : {result.verdict.upper()}")
    print(f"  Reason  : {result.verdict_reasoning}")
    if result.conditions:
        print("  Conditions:")
        for c in result.conditions:
            print(f"    • {c}")
    print("─" * 72)

    if result.notes:
        print(f"\n  Notes: {result.notes}")

    if result.test_results:
        print(f"\n  {'Script':<45} {'Kind':<12} {'P':>4} {'F':>4} {'S':>4}  {'Secs':>6}")
        print(f"  {'─'*45} {'─'*12} {'─':>4} {'─':>4} {'─':>4}  {'─':>6}")
        for r in result.test_results:
            status = "FAIL" if r.failed else "pass"
            print(
                f"  {r.script[:45]:<45} {r.kind:<12} {r.passed:>4} {r.failed:>4}"
                f" {r.skipped:>4}  {r.duration_seconds:>6.1f}  {status}"
            )

        failed_entries = [r for r in result.test_results if r.errors]
        if failed_entries:
            print("\n  Failures:")
            for r in failed_entries:
                print(f"    [{r.harness_id}] {r.script}")
                if r.log_path:
                    print(f"      log: {r.log_path}")
                if r.json_report_path:
                    print(f"      json-report: {r.json_report_path}")
                for err in r.errors:
                    location = (
                        f" {err.file}:{err.lineno}"
                        if err.file and err.lineno is not None
                        else ""
                    )
                    print(f"      ✗ {err.nodeid}{location}")
                    if err.message:
                        for line in err.message.splitlines()[:3]:
                            print(f"        {line}")

    if result.benchmark_results:
        print(f"\n  {'Script':<45} {'Workload':<20} {'Status'}")
        print(f"  {'─'*45} {'─'*20} {'─'*10}")
        for r in result.benchmark_results:
            status = "[dry-run]" if r.raw_output == "[dry-run]" else "captured"
            print(f"  {r.script[:45]:<45} {r.workload_id:<20} {status}")
            if r.optimization_target is not None:
                print(f"      target: {_fmt_value(r.optimization_target)}")
            if r.metrics:
                for m in r.metrics:
                    baseline = _fmt_value(m.baseline_value)
                    measured = _fmt_value(m.measured_value)
                    rel = (
                        f"{m.relative_change * 100:+.1f}%"
                        if m.relative_change is not None
                        else "—"
                    )
                    tag = "  REGRESSED" if m.regressed else ""
                    print(
                        f"      • {m.name}: baseline={baseline}"
                        f" measured={measured} Δ={rel}{tag}"
                    )
            else:
                print("      (no metrics reported)")

    print()


def _fmt_value(value: float | None) -> str:
    return f"{value:g}" if value is not None else "—"


if __name__ == "__main__":
    cli()
