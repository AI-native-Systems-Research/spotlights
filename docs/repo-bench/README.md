# Repo bench

> A self-contained Python module that builds a reproducible answer
> key from a target repo's merged PRs, then grades discovery output
> against it by direct match against the PRs' actual diffs.

The benchmark answers: **how much of the perf work that human authors
actually merged would a discovery method (signal pipeline, deep
research, source-only LLM, ...) rediscover from the same starting
point?**

## Pipeline at a glance

![Repo bench pipeline](repo_bench_pipeline.svg)

A single `repo-bench run` invocation drives stages 1–6 and
produces a self-contained run dir: filtered view + cached diffs +
snapshot SHA + workload signals + bench-spec ready for handoff to the
observability bench module.

The `match` step runs separately, after a discovery method has
produced a `findings.json` against the pinned SHA. It grades those
findings by judging each one against PR diffs in the filtered view.

### What each step does, in one line

| Step | In one line |
|---|---|
| **aggregate** | Pull every merged PR from the target repo's GitHub for the date window and cache them as JSONL. |
| **filter** | Apply rules over the raw PRs to drop bots / reverts / chores and keep only those with quantifiable perf signal; rank by `% × 1/files`. |
| **fetch-diffs** | Download the unified `git diff` for each PR in the filtered view. |
| **snapshot** | Pick a target-repo commit before the earliest filtered PR. The discovery pipeline must be blind to changes after this point. |
| **workloads** | Regex over filtered PRs to surface dominant model / feature / hardware / file-category signals. Output is a portfolio readers use to choose what workloads to run for telemetry capture. |
| **bench-spec** | Render the cross-module contract — SHA + reference config + workload signals + output naming — for the observability bench module. |
| **match** *(separate command)* | LLM judge compares each finding in a `findings.json` against PR diffs in the filtered view; produces strict / related / neighborhood / no_match verdicts per (finding, PR) pair plus per-finding hit rates. |

## Module surface

Two equivalent entry points: a Python API and a CLI subcommand.

```python
from spotlights_engine.repo_bench import run, filtering

handle = run.benchmark(
    window_start="2025-12-02",
    window_end="2026-06-03",
    rules=[
        filtering.NotBot(),
        filtering.NotRevert(),
        filtering.NotChore(),
        filtering.AnyPerfSignal(),
        filtering.RankBySpecificityAndMagnitude(),
    ],
    reference_bundle_name="20260525T202105Z_util0.4_mem16_lru",
)
print(handle.bench_spec_md)  # OBSERVABILITY_BENCH_SPEC.md path
```

```
repo-bench run \
  --start 2025-12-02 --end 2026-06-03 \
  --rules not-bot,not-revert,not-chore,any-perf-signal,rank-spec-mag \
  --reference-bundle 20260525T202105Z_util0.4_mem16_lru
```

The default flow stops after `bench-spec`. Once a discovery method
produces a `findings.json`:

```
repo-bench match \
  --findings <findings.json> \
  --bench-run-dir runs/repo_bench/bench-<window>__<view>__<UTC>/ \
  --experiment <name>
```

`match` writes its report to `<bench-run-dir>/matching/<experiment>/match_report.json`.

## The six stages of `run`

| # | Stage | Role | Cost | Output |
|---|---|---|---|---|
| 1 | `aggregate` | Pull every merged PR in window from the target repo's GitHub | hour-ish first run; skip on re-run | `data/repo_bench/raw/<window>/prs.jsonl` |
| 2 | `filter` | Predicates + ranker over raw → ranked subset | <1s, deterministic | `<run_dir>/view/prs.jsonl` |
| 3 | `fetch-diffs` | Per-PR unified diffs from GitHub for the view | ~3 min for 200 PRs | `data/repo_bench/raw/<window>/diffs/<n>.diff` |
| 4 | `snapshot` | Deterministic: pick a target SHA ≥ buffer before earliest filtered PR | <1s | `<run_dir>/snapshot.json` |
| 5 | `workloads` | Regex extraction of model / feature / hardware / file-category signals across the view | <10s | `<run_dir>/workload_analysis.json` + `workload_summary.md` |
| 6 | `bench-spec` | Render the cross-module spec for the observability bench, with workload signals inlined | <1s | `<run_dir>/{bench_spec.json, OBSERVABILITY_BENCH_SPEC.md}` |

Every step's existing cache or skip-if-exists check decides whether
real work happens. A second invocation with identical inputs is a
no-op (zero LLM calls, all cache hits, byte-identical artifacts).

## The match command (separate step)

`match` grades a `findings.json` against the filtered view by direct
diff inspection. No curator-extracted answer key — the diffs *are* the
answer.

For each finding, the matcher:

1. Bucket candidate PRs by file overlap (PRs in the view whose diff
   touches the finding's file).
2. Sub-classify into **Tier 1** (diff hunk inside the finding's
   symbol) and **Tier 2** (same file, different function).
3. Slice each candidate's diff to only hunks in the finding's file
   (cuts noise from multi-file PRs).
4. One LLM call per finding: judge produces a verdict per candidate
   PR.

Verdicts:

| Verdict | Roll-up rule |
|---|---|
| `same_idea` | Diff makes essentially the same change as the finding |
| `related` | Diff touches the same code with a related but distinct change |
| `neighborhood` | Diff touches the same file but unrelated code |
| `no_match` | Diff has nothing to do with the finding |

Output: `match_report.json` with per-finding verdicts, top-line hit
rates (strict_share, loose_share), and `tier_2_yield` (findings that
matched only at Tier 2 — captures refactors where the finding's
symbol moved to a sibling function).

## Snapshot SHA — what it pins, why a buffer

The discovery pipeline operates on target-repo source at some commit.
The benchmark scores discovery against PRs that landed *after* that
commit, so they're real recall targets, not changes already merged
into the codebase.

The snapshot picker:

1. Loads the filtered view's PR numbers; joins against raw to get
   `merged_at`.
2. Finds the **earliest** `merged_at` across them.
3. Computes a cutoff = earliest − `buffer_hours` (default 24h,
   override via `--snapshot-buffer-hours`).
4. Picks the latest raw PR merged before that cutoff.
5. Pins its `merge_sha`. Every filtered PR is now ≥ buffer hours
   newer than the snapshot.

The buffer guards against the target merging two related PRs in
rapid succession; without it, the snapshot's parent could already
contain side effects of changes we're about to score against.

## Run-dir layout

Every `repo-bench run` invocation produces one
self-contained run directory under `runs/repo_bench/`:

```
runs/repo_bench/bench-<window>__<view>__<UTC>/
  view/
    prs.jsonl                    # filtered + ranked PRs
    manifest.json                # rule specs, counts, derived_at
  snapshot.json                  # SHA pin + rationale
  workload_analysis.json         # full per-PR signals
  workload_summary.md            # top-N tables (also inlined into bench-spec)
  bench_spec.json                # canonical cross-module contract
  OBSERVABILITY_BENCH_SPEC.md    # rendered spec, includes workload signals §6
  run_report.json                # per-stage status, counts, paths
  run_report.md                  # human-readable summary
  matching/                      # populated when `match` runs
    <experiment>/
      match_report.json
```

Cached input (raw scrape) lives separately under
`data/repo_bench/raw/<window>/` — distinct lifecycle. The
`bench-` prefix on the run-id makes the directory self-identifying.

The raw scrape (`prs.jsonl`) and per-PR diffs are gitignored — both
are GitHub-rebuildable via `aggregate` / `fetch-diffs`. Each fresh
clone of the repo re-fetches.

## Schemas (the contract)

| Type | What it is | Lives in |
|---|---|---|
| `RawPR` | One scraped PR; the input to filtering | `schemas.py` |
| `RuleSpec` | Filter rule's identity (name, version, kind, params) — drives `view_id` | `schemas.py` |
| `SnapshotPin` | The target-repo SHA + rationale + buffer + n_view | `schemas.py` |
| `BenchSpec` | The cross-module contract handed to the observability bench | `schemas.py` |
| `RunReport` | Per-run summary | `schemas.py` |
| `MatchOutput` | Match agent's per-finding verdicts | `matching.py` |

Hash fields are constrained to lowercase 64-char hex.
`SnapshotPin.snapshot_sha` is a 40-char hex pattern.

## Filter rules

| Rule | Effect |
|---|---|
| `NotBot` | Drop PRs whose author looks like a bot (dependabot, github-actions, `*-bot`, etc.) |
| `NotRevert` | Drop PRs whose title starts with `Revert` / `[Revert ...]` |
| `NotChore` | Drop PRs tagged as bugfix / CI / test / doc / refactor / chore — UNLESS body has a strict perf claim |
| `TitleStrictPerfClaim` | Keep PRs whose title carries a number bound to a perf noun (e.g. `5% throughput improvement`, `3x speedup`) |
| `BodyStrictPerfClaim` | Same regex over the body |
| `AnyStrictPerfClaim` | Title OR body |
| `AnyLoosePerfClaim` | Same shape, more tolerant of phrasing distance |
| `AnyPerfSignal` | Strict claim (title or body) OR `[Perf]` / `[Performance]` / `[Optimize]` author tag |
| `RankBySpecificityAndMagnitude` | Score = `claimed_pct × 1/files_changed`. Single ranker per view. |

The `view_id` is a 12-char hex hash of the canonical (sorted) rule
specs. Two rule lists with the same `view_id` produce the same
filtered view.

## Workload signals

The workloads stage reads filtered PRs' titles + bodies + diffs and
extracts:

- **Models** (regex over known model families: Qwen, DeepSeek, Llama, etc.)
- **Features** (FP8, MoE, expert-parallel, spec-decode, prefix-cache, ...)
- **Hardware** (Hopper, Blackwell, AMD, etc.)
- **File categories touched** (attention, scheduler, kernels, ...)
- **Parallelism axes** (tp, pp, dp, max_num_seqs)
- **Benchmark commands** (extracted `vllm serve` / `vllm bench`-style commands)

Aggregate counts + a greedy AND-cover portfolio (top model+feature
intersections) get rendered into `workload_summary.md` and inlined
into `OBSERVABILITY_BENCH_SPEC.md` as §6. Repo-agnostic patterns at
the regex level — repos that don't match the model/feature
vocabulary still get the file-category and benchmark-command tables;
just with sparser counts.

## Costs, roughly

The bench module itself does **zero** LLM calls during `run`. The
workload signals are regex; the snapshot is deterministic; the
bench-spec is template rendering. End-to-end `run` cost is GitHub
API (free with token) + ~5 min wallclock for ~200 diffs.

The match step is the only LLM cost on the repo-bench side:

| Call | Model | Input | Output | Per-call |
|---|---|---|---|---|
| Match (per finding) | Sonnet | ~5–50 KB (file-filtered diff slices for candidates) | ~1 KB | ~$0.05–0.20 |

For a 148-finding run against a 200-PR view, total match cost is
typically ~$1–5 (judge calls only fire when a finding has at least
one file-overlap candidate).

## Reproducibility, honestly

- **Cache replay is deterministic.** Re-running `run` with no input
  changes is a no-op: filter regenerates from cached raw, snapshot
  regenerates from cached raw + view, workload signals regenerate
  from cached diffs. Byte-identical output.
- **Match cache populates fresh per run.** Each `match` invocation
  judges the findings against current diffs. Matching is not yet
  cached (POC stance — prompt + judge model are stable across runs,
  but no on-disk cache).
- **`prompt_sha256` and `model_id` land in the manifest** so any
  future review can verify what produced the report.

## Contract with adjacent modules

```
                ┌──────────────────────────┐
                │ Repo bench      │
                │ (this module)            │
                │                          │
                │  produces:               │
                │  - filtered view +       │
                │    snapshot SHA          │
                │  - OBSERVABILITY_BENCH_  │
                │     SPEC.md   ───┐       │
                └──────────────────┼───────┘
                                   │
                                   ▼
            ┌──────────────────────────────────┐
            │ Observability bench module       │
            │                                  │
            │  reads spec, runs target repo at │
            │  pinned SHA against the workload │
            │  signals, produces:              │
            │  data/<timestamp>_<config>_      │
            │    repo-<short_sha>/             │
            └──────────┬───────────────────────┘
                       │
                       ▼
            ┌──────────────────────────────────┐
            │ Discovery method                 │
            │ (signal pipeline / deep research │
            │  / source-only LLM / ...)        │
            │                                  │
            │  reads RawTelemetry + source     │
            │  at the SHA, produces:           │
            │  findings.json                   │
            └──────────┬───────────────────────┘
                       │
                       ▼ (back into this module)
   repo-bench match → matching/<experiment>/match_report.json
```

The bench spec we hand to the observability bench module is the only
load-bearing communication outward. Everything else is one-way.

## Module layout

```
src/spotlights_engine/repo_bench/
  __init__.py
  cli.py                     # 5 subcommands: aggregate, filter,
                             #   fetch-diffs, run, match
  schemas.py                 # all pydantic models
  storage.py                 # path resolution, atomic writes
  aggregation.py             # GitHub scrape
  diff_fetcher.py            # per-PR unified diffs
  snapshot.py                # SHA picker (deterministic)
  workloads.py               # workload signal extraction
  bench_spec.py              # JSON + MD render for the spec
  matching.py                # findings → diffs LLM judge
  report.py                  # JSON + MD render for run summaries
  run.py                     # orchestrator (run.benchmark)
  templates/                 # MD templates (bench_spec, run_report)

  filtering/
    __init__.py
    heuristics.py            # bot/revert/perf-claim/perf-tag regex
    rules.py                 # all filter rules + RankBySpecificityAndMagnitude
    derive.py                # apply rules → <run_dir>/view/

tests/unit/repo_bench/
  test_aggregation.py        # GitHub scrape (no network in CI)
  test_filtering.py          # rules + derive + view_id determinism
  test_diff_fetcher.py       # diff fetcher with stub HTTP
  test_bench_spec.py         # spec JSON + MD render
  test_report.py             # run report JSON + MD render
  test_storage.py            # IO primitives
```

78 tests; the full suite runs in ~1 second. No live LLM calls in
tests — `match` accepts an injected runner protocol.

## Companion docs

- [`PROGRESS_UI_BRIEF.md`](PROGRESS_UI_BRIEF.md) — historical brief
  for adding a live multi-stage progress UI to `repo-bench
  run`.
