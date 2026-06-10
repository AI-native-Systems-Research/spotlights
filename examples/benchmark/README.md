# Repo bench — example match reports

This directory holds curated `match_report.{json,md}` outputs from
`repo-bench match` runs, kept here so they can be linked / reviewed
without rerunning the bench.

## Layout

```
examples/benchmark/
  <experiment_id>/
    match_report.json    # canonical report (machine-readable)
    match_report.md      # human-readable view (same data)
```

Each subdirectory is one match experiment. The MD has the full
write-up: ground-truth PR counts, what Tier-1 / Tier-2 mean,
verdict definitions, the per-finding breakdown, and per-finding
candidate tables with citations. Open the MD first — the JSON is
there for tooling.

## Reproducing

These reports were produced by:

```
repo-bench run \
  --start <window_start> --end <window_end> \
  --rules not-bot,not-revert,not-chore,any-perf-signal-or-label,rank-spec-mag

repo-bench match \
  --findings <findings.json> \
  --bench-run-dir runs/repo_bench/bench-<window>__<view>__<UTC>/ \
  --experiment <experiment_id>
```

See [`docs/repo-bench/README.md`](../../docs/repo-bench/README.md) for
the full module surface.

## Current entries

| Experiment | Discovery method | Filter | Weighted score | Loose hit rate |
|---|---|---|---:|---:|
| [`signal-pipeline-otel-4b61266--label-relaxed`](signal-pipeline-otel-4b61266--label-relaxed/match_report.md) | Signal pipeline (OTel bundle, vLLM SHA `4b61266`) | `not-bot,not-revert,not-chore,any-perf-signal-or-label,rank-spec-mag` (624 PRs view) | 0.230 | 40% (2/5) |
