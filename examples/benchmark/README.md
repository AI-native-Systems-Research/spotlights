# Repo bench — example match reports

Curated `match_report.{json,md}` outputs from `repo-bench match` runs.
One subdir per experiment; both files are the same report (JSON for
tooling, MD for humans). Each MD has ground-truth counts, tier/verdict
definitions, per-finding tables with citations.

## Results

vLLM, window `2025-12-02 → 2026-06-03`, 624 filtered PRs, snapshot SHA
`a690fb5b`, judge model `sonnet`.

| Run | Scope | Findings | Weighted | `same_idea` | `related` | Report |
|---|---|---:|---:|---:|---:|---|
| Signal pipeline | OTel-derived (anomaly files across kv-offload, scheduler, kv-cache, spec-decode) | 5 | 0.23 | 0 | 2 | [match_report.md](signal-pipeline-otel-4b61266--label-relaxed/match_report.md) |
| spotlights-engine — `--mode code_only` (skip stages 3 + 4) | `v1.kv_offload` | 10 | 0.45 | 3 | 13 | [match_report.md](spotlights-code-only-skip3-4--a690fb5/match_report.md) |
| spotlights-engine — `--mode full` (all 5 stages, papers) | `v1.kv_offload` | 9 | **0.53** | **5** | 13 | [match_report.md](spotlights-full--a690fb5/match_report.md) |

## Column meaning

- **Findings** — number of candidate optimization sites the discovery
  method emitted for the target module.
- **Weighted score** — overall quality in `[0, 1]`. Per-finding score =
  best `(verdict, tier)` weight across its candidate matches; overall
  is the mean across findings (see formula below).
- **`same_idea`** — count of `(finding, candidate-PR)` pairs the LLM
  judge labeled as essentially the same change as the finding.
  Strongest evidence the discovery method anticipated what maintainers
  shipped.
- **`related`** — same code touched, related but distinct change.
  Counts the partial-credit hits.

## How the weighted score is computed

For every finding, `repo-bench match` collects candidate PRs in the
filtered view whose diff touches the finding's file, then asks an LLM
judge to label each `(finding, candidate-PR)` pair. Each candidate is
classified as:

- **Tier 1 (T1)** — the PR's diff lands inside the finding's exact
  symbol (function/method/class). Strongest match.
- **Tier 2 (T2)** — same file, different function. Same neighborhood.

Weights per `(verdict, tier)`:

| Verdict | T1 | T2 |
|---|---:|---:|
| `same_idea` | 1.00 | 0.70 |
| `related` | 0.60 | 0.40 |
| `neighborhood` | 0.05 | 0.05 |
| `no_match` | 0.00 | 0.00 |

Per-finding score = `max(weight)` over its matches. Overall
`weighted_score` = mean of per-finding scores. Always in `[0, 1]`.

## Reproducing

```
repo-bench run \
  --start 2025-12-02 --end 2026-06-03 \
  --rules not-bot,not-revert,not-chore,any-perf-signal-or-label,rank-spec-mag

repo-bench match \
  --findings <findings.json> \
  --bench-run-dir runs/repo_bench/bench-<window>__<view>__<UTC>/ \
  --experiment <experiment_id>
```

See [`docs/repo-bench/README.md`](../../docs/repo-bench/README.md) for
the full module surface and the deep-research / signal-pipeline
discovery sources.
