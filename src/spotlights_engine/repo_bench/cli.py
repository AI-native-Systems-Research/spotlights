"""CLI entry point for the repo bench module.

Invoked as:

    python -m spotlights_engine.repo_bench <subcommand> [args]

Subcommands:
    aggregate     Scrape merged PRs from a repo's GitHub for a date window.
    filter        Apply rules; write the ranked view.
    fetch-diffs   Fetch per-PR unified diffs.
    snapshot      Pick the target-repo SHA + write snapshot.json.
    bench-spec    Render the bench spec for handoff to the observability bench.
    run           Run the full pipeline end-to-end (default through bench-spec).
    match         Grade a findings.json by direct match against PR diffs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from pathlib import Path

from spotlights_engine.repo_bench import (
    aggregation,
    diff_fetcher,
    filtering,
    matching,
    run as run_module,
)


# Registry: CLI rule names → constructors.
_RULES: dict[str, type[filtering.Rule]] = {
    "not-bot": filtering.NotBot,
    "not-revert": filtering.NotRevert,
    "not-chore": filtering.NotChore,
    "title-strict-perf-claim": filtering.TitleStrictPerfClaim,
    "body-strict-perf-claim": filtering.BodyStrictPerfClaim,
    "any-strict-perf-claim": filtering.AnyStrictPerfClaim,
    "any-loose-perf-claim": filtering.AnyLoosePerfClaim,
    "any-perf-signal": filtering.AnyPerfSignal,
    "rank-spec-mag": filtering.RankBySpecificityAndMagnitude,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repo-bench",
        description=(
            "Repo bench — pull merged PRs from a target repo, "
            "filter to perf-relevant ones, pin a snapshot SHA + workload "
            "spec, then grade findings against the filtered view by "
            "direct diff match. See docs/repo-bench/."
        ),
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    agg = sub.add_parser(
        "aggregate",
        help="Scrape merged PRs from GitHub for a date window.",
    )
    agg.add_argument("--start", help="Window start (ISO date). Default: 6mo before --end.")
    agg.add_argument("--end", help="Window end (ISO date). Default: today.")
    agg.add_argument("--repo", default=aggregation.DEFAULT_REPO,
                     help=f"GitHub repo (default: {aggregation.DEFAULT_REPO}).")
    agg.add_argument("--token", default=None, help="GitHub PAT. Default: $GITHUB_TOKEN.")
    agg.add_argument("-v", "--verbose", action="store_true")

    flt = sub.add_parser(
        "filter",
        help="Apply rules to raw/<window>/prs.jsonl; write the ranked view "
             "into <run-dir>/view/.",
    )
    flt.add_argument("--window", required=True)
    flt.add_argument("--rules", required=True,
                     help="Comma-separated rule names. Available: "
                     + ", ".join(sorted(_RULES.keys())) + ".")
    flt.add_argument("--run-dir", required=True,
                     help="Run dir to write view/ into.")

    fd = sub.add_parser(
        "fetch-diffs",
        help="Fetch per-PR unified diffs for a run dir's view.",
    )
    fd.add_argument("--window", required=True)
    fd.add_argument("--run-dir", required=True,
                    help="Run dir containing view/prs.jsonl.")
    fd.add_argument("--repo", default=aggregation.DEFAULT_REPO)
    fd.add_argument("--token", default=None)
    fd.add_argument("--refresh", action="store_true")

    rn = sub.add_parser(
        "run",
        help=(
            "Run the full benchmark end-to-end: aggregate -> filter -> "
            "fetch-diffs -> snapshot -> bench-spec. Each step skips work "
            "that's already cached. A second invocation with the same "
            "inputs is a no-op."
        ),
    )
    rn.add_argument("--start", help="Window start. Default: 6mo before --end.")
    rn.add_argument("--end", help="Window end. Default: today.")
    rn.add_argument("--rules", required=True,
                    help="Comma-separated rule names. Available: "
                    + ", ".join(sorted(_RULES.keys())) + ".")
    rn.add_argument("--reference-bundle",
                    default="20260525T202105Z_util0.4_mem16_lru",
                    help="Name of the prior observability bundle whose "
                    "config the bench module should match.")
    rn.add_argument("--bench-spec-config-notes", default="",
                    help="Free-form extra context rendered into the bench spec.")
    rn.add_argument("--snapshot-buffer-hours", type=int, default=None,
                    help="Override the default 24h buffer between snapshot "
                    "SHA and earliest filtered PR.")
    rn.add_argument("--repo", default=aggregation.DEFAULT_REPO)
    rn.add_argument("--token", default=None)
    rn.add_argument("--refresh", action="store_true",
                    help="Bypass cache for filter / fetch-diffs.")
    rn.add_argument("--refresh-aggregate", action="store_true",
                    help="Re-scrape from GitHub. Expensive.")
    rn.add_argument("--from-step", choices=list(run_module._STEP_ORDER),
                    default="aggregate")
    rn.add_argument("--through-step", choices=list(run_module._STEP_ORDER),
                    default=run_module._DEFAULT_THROUGH)

    mt = sub.add_parser(
        "match",
        help=(
            "Grade a findings.json by direct match against PR diffs in "
            "the filtered view. Per-finding LLM judge over file-overlap "
            "candidates."
        ),
    )
    mt.add_argument("--findings", required=True,
                    help="Path to findings.json (v3 shape).")
    mt.add_argument("--bench-run-dir", required=True,
                    help="A bench run dir (contains view/prs.jsonl + "
                    "snapshot.json).")
    mt.add_argument("--experiment", required=True,
                    help="Experiment id; output lands at "
                    "<bench-run-dir>/matching/<experiment_id>/.")
    mt.add_argument("--judge-model", default="sonnet")
    mt.add_argument("--repo-path", default=".")
    mt.add_argument("--wallclock-s", type=int, default=matching.DEFAULT_WALLCLOCK_S)
    mt.add_argument("--max-turns", type=int, default=matching.DEFAULT_MAX_TURNS)

    return p


def _resolve_window(start: str | None, end: str | None) -> tuple[date, date]:
    end_d = date.fromisoformat(end) if end else date.today()
    start_d = date.fromisoformat(start) if start else end_d - timedelta(days=183)
    return start_d, end_d


def _cmd_aggregate(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GitHub token required. Set $GITHUB_TOKEN or pass --token.",
              file=sys.stderr)
        return 2
    start, end = _resolve_window(args.start, args.end)
    handle = aggregation.scrape(
        window_start=start, window_end=end, token=token, repo=args.repo,
    )
    print(f"window_id: {handle.window_id}")
    print(f"prs:       {handle.total_prs}")
    print(f"out_dir:   {handle.out_dir}")
    return 0


def _cmd_filter(args: argparse.Namespace) -> int:
    rule_names = [n.strip() for n in args.rules.split(",") if n.strip()]
    if not rule_names:
        print("ERROR: --rules must list at least one rule.", file=sys.stderr)
        return 2
    unknown = [n for n in rule_names if n not in _RULES]
    if unknown:
        print(
            f"ERROR: unknown rule(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_RULES.keys()))}.",
            file=sys.stderr,
        )
        return 2
    rules = [_RULES[n]() for n in rule_names]
    handle = filtering.derive(
        window_id=args.window, rules=rules,
        run_dir=Path(args.run_dir).resolve(),
    )
    print(f"view_id:   {handle.view_id}")
    print(f"out_dir:   {handle.path}")
    print(f"n_kept:    {handle.n_kept}")
    print(f"n_dropped: {handle.n_dropped}")
    return 0


def _cmd_fetch_diffs(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("ERROR: GitHub token required.", file=sys.stderr)
        return 2
    view_path = Path(args.run_dir).resolve() / "view" / "prs.jsonl"
    handle = diff_fetcher.fetch_diffs(
        window_id=args.window, view_path=view_path, token=token,
        repo=args.repo, refresh=args.refresh,
    )
    print(f"diffs_dir: {handle.diffs_dir}")
    print(f"n_total:   {handle.n_total}")
    print(f"n_fetched: {handle.n_fetched}")
    print(f"n_skipped: {handle.n_skipped}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    rule_names = [n.strip() for n in args.rules.split(",") if n.strip()]
    if not rule_names:
        print("ERROR: --rules must list at least one rule.", file=sys.stderr)
        return 2
    unknown = [n for n in rule_names if n not in _RULES]
    if unknown:
        print(
            f"ERROR: unknown rule(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(_RULES.keys()))}.",
            file=sys.stderr,
        )
        return 2
    rules = [_RULES[n]() for n in rule_names]

    start, end = _resolve_window(args.start, args.end)
    token = args.token or os.environ.get("GITHUB_TOKEN")

    handle = run_module.benchmark(
        window_start=start, window_end=end, rules=rules,
        reference_bundle_name=args.reference_bundle,
        bench_spec_config_notes=args.bench_spec_config_notes,
        refresh=args.refresh, refresh_aggregate=args.refresh_aggregate,
        from_step=args.from_step, through_step=args.through_step,
        github_token=token, github_repo=args.repo,
        snapshot_buffer_hours=args.snapshot_buffer_hours,
    )
    print(f"window_id:      {handle.window_id}")
    print(f"view_id:        {handle.view_id}")
    if handle.aggregate is not None:
        print(f"aggregate:      {handle.aggregate.total_prs} PRs scraped")
    if handle.filter is not None:
        print(f"filter:         {handle.filter.n_kept} kept, "
              f"{handle.filter.n_dropped} dropped")
    if handle.fetch_diffs is not None:
        print(f"fetch_diffs:    {handle.fetch_diffs.n_fetched} fetched, "
              f"{handle.fetch_diffs.n_skipped} skipped")
    if handle.snapshot is not None:
        print(f"snapshot:       SHA {handle.snapshot.snapshot_sha[:12]} "
              f"(PR #{handle.snapshot.snapshot_pr_number}, "
              f"{handle.snapshot.n_view} view PRs anchored)")
    if handle.workloads is not None:
        print(f"workloads:      {handle.workloads.md_path}")
    if handle.bench_spec_md is not None:
        print(f"bench spec:     {handle.bench_spec_md}")
    if handle.report_md is not None:
        print(f"run report:     {handle.report_md}")
    return 0


def _cmd_match(args: argparse.Namespace) -> int:
    handle = matching.run_matching(
        findings_path=Path(args.findings).resolve(),
        bench_run_dir=Path(args.bench_run_dir).resolve(),
        experiment_id=args.experiment,
        judge_model=args.judge_model,
        repo_path=Path(args.repo_path).resolve(),
        wallclock_s=args.wallclock_s,
        max_turns=args.max_turns,
    )
    print(f"out_dir:        {handle.out_dir}")
    print(f"report:         {handle.report_path}")
    print(f"n_findings:     {handle.n_findings}")
    print(f"judge calls:    {handle.n_judged}")
    print(f"same_idea:      {handle.same_idea}")
    print(f"related:        {handle.related}")
    print(f"neighborhood:   {handle.neighborhood}")
    print(f"no_match:       {handle.no_match}")
    print(f"tier_2_yield:   {handle.tier_2_yield}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.subcommand == "aggregate":
        return _cmd_aggregate(args)
    if args.subcommand == "filter":
        return _cmd_filter(args)
    if args.subcommand == "fetch-diffs":
        return _cmd_fetch_diffs(args)
    if args.subcommand == "run":
        return _cmd_run(args)
    if args.subcommand == "match":
        return _cmd_match(args)
    print(f"unknown subcommand: {args.subcommand}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
