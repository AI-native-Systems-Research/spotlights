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
from datetime import date

from pathlib import Path

from spotlights_engine.repo_bench import (
    aggregation,
    diff_fetcher,
    filtering,
    matching,
    run as run_module,
)
from spotlights_engine.repo_bench.config import compile_filter_patterns, load_config
from spotlights_engine.repo_bench.filtering import heuristics as _heuristics


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
    "any-perf-signal-or-label": filtering.AnyPerfSignalOrLabel,
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
    agg.add_argument("--start", required=True, help="Window start (ISO date).")
    agg.add_argument("--end", required=True, help="Window end (ISO date).")
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
    rn.add_argument("--start", required=True, help="Window start (ISO date).")
    rn.add_argument("--end", required=True, help="Window end (ISO date).")
    rn.add_argument("--rules", required=True,
                    help="Comma-separated rule names. Available: "
                    + ", ".join(sorted(_RULES.keys())) + ".")
    rn.add_argument("--bench-spec-config-notes", default="",
                    help="Free-form extra context rendered into the bench spec.")
    rn.add_argument("--snapshot-buffer-hours", type=int, default=None,
                    help="Override the default 24h buffer between snapshot "
                    "SHA and earliest filtered PR.")
    rn.add_argument("--config", default="vllm",
                    help="Config name (e.g. 'vllm') or path to a TOML file. "
                    "Drives filter perf-noun and workload-pattern lists. "
                    "Default: 'vllm'.")
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
    rn.add_argument("--workload-llm", action="store_true",
                    help="Run LLM workload-command extraction; produces "
                    "concrete runnable workloads in §6 of the bench spec. "
                    "Costs ~$0.05-0.10 per filtered PR.")
    rn.add_argument("--workload-llm-model", default="sonnet",
                    help="Model for the workload-command extractor.")
    rn.add_argument("--workload-llm-top-n", type=int, default=5,
                    help="Top-N clusters to render in the workload portfolio.")

    fr = sub.add_parser(
        "filter-report",
        help="Evaluate each predicate independently and report per-rule "
             "drop/keep stats with overlap matrix.",
    )
    fr.add_argument("--window", required=True)
    fr.add_argument("--rules", required=True,
                    help="Comma-separated rule names (predicates only; rankers ignored).")
    fr.add_argument("--config", default="vllm",
                    help="Config name (e.g. 'vllm') or path to a TOML file. "
                    "Drives filter perf-noun and label lists. Default: 'vllm'.")

    mw = sub.add_parser(
        "merge-windows",
        help="Merge multiple raw windows into one, deduplicating by pr_number.",
    )
    mw.add_argument("--windows", nargs="+", required=True,
                    help="Window IDs to merge (e.g. 2025-12-02__2026-06-03 "
                    "2026-06-03__2026-06-10). Later windows win on duplicates.")
    mw.add_argument("--target", default=None,
                    help="Explicit target window ID. Default: auto-computed "
                    "from the combined date span.")

    cw = sub.add_parser(
        "characterize-workloads",
        help="Classify benchmark workloads in filtered PRs as synthetic, "
             "recorded_trace, or combination. Extracts generator params "
             "and trace sources.",
    )
    cw.add_argument("--window", required=True,
                    help="Window ID (e.g. 2025-12-02__2026-06-10).")
    cw.add_argument("--run-dir", required=True,
                    help="Run dir containing view/prs.jsonl.")
    cw.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <run-dir>/.")
    cw.add_argument("--include-kernel", action="store_true",
                    help="Include kernel micro-benchmarks (excluded by default).")
    cw.add_argument("--include-accuracy", action="store_true",
                    help="Include accuracy evals like lm_eval (excluded by default).")

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


def _resolve_window(start: str, end: str) -> tuple[date, date]:
    return date.fromisoformat(start), date.fromisoformat(end)


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
        bench_spec_config_notes=args.bench_spec_config_notes,
        refresh=args.refresh, refresh_aggregate=args.refresh_aggregate,
        from_step=args.from_step, through_step=args.through_step,
        github_token=token, github_repo=args.repo,
        snapshot_buffer_hours=args.snapshot_buffer_hours,
        workload_llm=args.workload_llm,
        workload_llm_model=args.workload_llm_model,
        workload_llm_top_n=args.workload_llm_top_n,
        config=args.config,
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
    if handle.workload_commands is not None:
        print(f"workload cmds:  {handle.workload_commands.md_path} "
              f"({handle.workload_commands.n_clusters} clusters)")
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
    n = handle.n_findings or 1
    pct_strict = 100 * handle.same_idea / n  # rough; finding-level rate is in MD
    bar_w = 30
    filled = max(0, min(bar_w, round(handle.weighted_score * bar_w)))
    bar = "#" * filled + "-" * (bar_w - filled)

    print()
    print("=" * 68)
    print(f"  Match report — {args.experiment}")
    print("=" * 68)
    print()
    print(f"  Weighted score   {handle.weighted_score:.3f}  [{bar}]  / 1.000")
    print(f"                   (strict={handle.same_idea}  related={handle.related}  "
          f"neighborhood={handle.neighborhood}  no_match={handle.no_match})")
    print()
    print(f"  Findings         {handle.n_findings}")
    print(f"  Judged           {handle.n_judged}  (skipped: no candidate PRs in view)")
    print(f"  Tier-2 yield     {handle.tier_2_yield}  "
          f"(findings whose only hit was Tier 2)")
    print()
    print(f"  Report (json)    {handle.report_path}")
    print(f"  Report (md)      {handle.report_md_path}")
    print("=" * 68)
    return 0


def _cmd_characterize_workloads(args: argparse.Namespace) -> int:
    from spotlights_engine.repo_bench import workload_characterization

    run_dir = Path(args.run_dir).resolve()
    view_path = run_dir / "view" / "prs.jsonl"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir

    handle = workload_characterization.characterize(
        window_id=args.window,
        view_path=view_path,
        out_dir=out_dir,
        include_kernel=args.include_kernel,
        include_accuracy=args.include_accuracy,
    )
    print(f"analyzed:   {handle.n_total} PRs")
    print(f"with bench: {handle.n_with_benchmarks}")
    for wtype, count in sorted(handle.type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {wtype}: {count}")
    print(f"json:       {handle.json_path}")
    print(f"markdown:   {handle.md_path}")
    print(f"csv:        {handle.csv_path}")
    return 0


def _cmd_filter_report(args: argparse.Namespace) -> int:
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
    cfg = load_config(args.config)
    _heuristics.set_active_config(compile_filter_patterns(cfg))
    rules = [_RULES[n]() for n in rule_names]
    report = filtering.generate_filter_report(window_id=args.window, rules=rules)
    formatted = filtering.format_report(report)
    print(formatted)
    from spotlights_engine.repo_bench.storage import raw_dir as _raw_dir
    import json
    out_dir = _raw_dir(args.window)
    md_path = out_dir / "filter_report.md"
    md_path.write_text(filtering.format_report_md(report) + "\n")
    json_path = out_dir / "filter_report.json"
    json_path.write_text(json.dumps(filtering.report_to_dict(report), indent=2) + "\n")
    print(f"\n  Written: {md_path}")
    print(f"  Written: {json_path}")
    return 0


def _cmd_merge_windows(args: argparse.Namespace) -> int:
    handle = aggregation.merge_windows(
        window_ids=args.windows,
        target_window_id=args.target,
    )
    print(f"window_id: {handle.window_id}")
    print(f"prs:       {handle.total_prs}")
    print(f"out_dir:   {handle.out_dir}")
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
    if args.subcommand == "characterize-workloads":
        return _cmd_characterize_workloads(args)
    if args.subcommand == "filter-report":
        return _cmd_filter_report(args)
    if args.subcommand == "merge-windows":
        return _cmd_merge_windows(args)
    if args.subcommand == "match":
        return _cmd_match(args)
    print(f"unknown subcommand: {args.subcommand}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
