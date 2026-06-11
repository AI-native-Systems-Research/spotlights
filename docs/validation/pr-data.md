# PR Data Filtering Rules

The filtering pipeline used by `repo_bench` to surface performance-relevant PRs. The sample data throughout this document was fetched from the [vLLM repository](https://github.com/vllm-project/vllm).

## Overview

Rules are applied as: **predicate chain + ranker**. The four predicates are AND'd (a PR must pass all four), then the ranker scores and sorts survivors.

```
not-bot,not-revert,not-chore,any-perf-signal-or-label,rank-spec-mag
```

## Predicates

### `not-bot`

Drops PRs **created by** a bot account. The `author` field is the GitHub `user.login` of the account that opened the PR (from the GitHub API `detail["user"]["login"]`). It does not consider who merged, committed, or reviewed.

Detection is two-tier:

1. **Known-bot set** (exact match): `dependabot[bot]`, `pre-commit-ci[bot]`, `github-actions[bot]`, `renovate[bot]`, `mergify[bot]`, `copilot`, `vllm-bot`
2. **Suffix regex** (catches unlisted bots): `\[bot\]$`, `-bot$`, `^bot-`

### `not-revert`

Drops PRs whose title starts with `Revert` or `[Revert` (case-insensitive).

Pattern: `^\s*\[?revert\b`

### `not-chore`

Drops PRs tagged as routine maintenance via title prefixes like `[Bugfix]`, `fix:`, `[CI]`, `doc:`, `[Refactor]`, etc.

**Override**: keeps the PR if the title or body contains a concrete numeric perf claim (e.g. `[Bugfix] fix slow attention path, 12% throughput regression`).

### `any-perf-signal-or-label`

Keeps only PRs that demonstrate performance relevance through at least one of:

1. **Strict numeric perf claim** in title or body (e.g. `5% throughput improvement`, `3x speedup`, `latency decreased by 10%`)
2. **Perf tag** in title (`[Perf]`, `[Performance]`, `[Optimize]`, or `perf:`/`performance:` prefix)
3. **Label match** against the repo config's `perf_labels` whitelist

Perf nouns recognized: throughput, latency, speedup, performance, memory, tput, tps, qps, rps, ttft, tpot, itl, vram, footprint.

## Ranker

### `rank-spec-mag`

Scores each surviving PR:

```
score = claimed_pct × (1 / files_changed)
```

- `claimed_pct`: largest percentage extracted from the title (preferred) or title+body
- `files_changed`: number of files modified (minimum 1)

This favors specific, high-magnitude optimizations over sprawling PRs with small claims.

**Tie-breaking**: score desc → pct desc → files asc → pr_number asc.

**Examples**:
- 1-file PR claiming 10% → score 10.0
- 8-file PR claiming 10% → score 1.25
- PR with no extractable percentage → score 0 (kept but ranked last)

## Filter Report

The `filter-report` command evaluates each predicate **independently** against all raw PRs and produces a breakdown showing how many PRs fall into each category. Unlike the standard filtering (which short-circuits on the first failing rule), this tests every PR against every rule separately — so a bot-authored revert counts under both "Created by a bot" and "Revert PRs".

```
python -m spotlights_engine.repo_bench.cli filter-report \
  --window <window_id> \
  --rules not-bot,not-revert,not-chore,any-perf-signal-or-label \
  --config vllm
```

The report is saved to `data/repo_bench/raw/<window_id>/filter_report.md` (and a `.json` sibling with structured data).

### Sample output

Window `2025-12-02__2026-06-10` (vllm-project/vllm):

| Category | Count | % of total |
|----------|------:|-----------:|
| Created by a bot | 5 | 0.1% |
| Revert PRs | 47 | 0.8% |
| Chore (bugfix/CI/test/doc/refactor) | 2074 | 37.4% |
| Not related to performance optimization | 4709 | 84.9% |

**After all filters**: 653 PRs kept (11.8% of 5549 total)

### Reading the results

- **Created by a bot** (5): PRs opened by automated accounts (dependabot, renovate, etc.). Negligible in this repo.
- **Revert PRs** (47): PRs that undo a previous merge. These carry no new optimization signal.
- **Chore** (2074): PRs tagged as bugfix, CI, test, doc, refactor, etc. — routine maintenance unlikely to contain performance work. This is the second-largest exclusion category.
- **Not related to performance optimization** (4709): PRs with no numeric perf claim, no perf tag, and no perf-related label. This is the dominant filter — most PRs in a large project simply aren't performance work.

The overlap matrix shows that most chore PRs (1893 of 2074) are also caught by the perf-signal filter, but 181 chore PRs *would* have slipped through without the dedicated `not-chore` rule (they happen to mention a percentage in a non-performance context that the strict regex matches).

## Workload Characterization

The `characterize-workloads` command classifies each filtered PR's benchmark workload by type (synthetic traffic, recorded trace, or combination) and extracts the generator parameters and trace sources.

### How it works

The tool reads PR bodies from the filtered view and uses two-tier extraction:

**Tier 1 — Regex** (handles ~98% of cases): Finds benchmark commands in the PR body (`vllm bench serve`, `vllm bench throughput`, `benchmark_serving.py`, `python benchmarks/*.py`, `lm_eval`) and classifies based on flags:

- `--dataset-name random` / `--random-input-len` / `--random-output-len` → **synthetic**
- `--dataset-name sharegpt|hf|timed_trace|burstgpt|custom` → **recorded_trace**
- `--input-len` / `--output-len` without explicit dataset (vllm bench defaults to random) → **synthetic**
- Both synthetic and trace datasets in the same PR → **combination**
- Commands with `--model`/`--num-prompts` but no dataset flags (vllm bench default = random) → **synthetic**

**Tier 2 — LLM fallback** (opt-in, for the remaining unknowns): Called when a PR has benchmark results but no parseable command. Uses Claude to extract workload type from prose descriptions, with a hallucination guard that verifies any cited commands are verbatim substrings of the PR body.

Benchmark scope is categorized as `serving`, `kernel`, or `accuracy`. Kernel micro-benchmarks and accuracy evals (lm_eval) are excluded by default.

### Usage

```
python -m spotlights_engine.repo_bench characterize-workloads \
  --window <window_id> \
  --run-dir <path-to-run-dir> \
  [--include-kernel]       # include kernel micro-benchmarks
  [--include-accuracy]     # include lm_eval / accuracy evals
  [--llm-fallback]         # use LLM for unknowns (opt-in)
```

Output is saved to `<run-dir>/workload_characterization.json`, `<run-dir>/workload_characterization.md`, and `<run-dir>/workload_characterization.csv`. The CSV has one row per benchmark entry with columns for PR number, title, workload type, scope, tool, params_specified (default/explicit), trace source, flattened generator parameters, and the source command.

A sample CSV is available at [assets/workload_characterization_sample.csv](assets/workload_characterization_sample.csv).

### Sample output

Window `2025-12-02__2026-06-10`, 653 filtered PRs, with `--include-kernel`:

**138** PRs contain recognizable benchmark commands.

| Type | Count | % of benchmarked PRs |
|------|------:|---------------------:|
| synthetic | 105 | 76.1% |
| recorded_trace | 27 | 19.6% |
| combination | 4 | 2.9% |
| unknown | 2 | 1.4% |

_Accuracy evals (lm_eval): 68 entries excluded._

#### Trace sources (recorded + combination)

| Source | Count | Example PRs |
|--------|------:|-------------|
| sharegpt//tmp/ShareGPT_V3_unfiltered_cleaned_split.json | 12 | #31781, #32619, #33568, #34206, #34974 +7 more |
| sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json | 3 | #30528, #31246, #35220 |
| sharegpt/./ShareGPT_V3_unfiltered_cleaned_split.json | 2 | #35442, #40172 |
| speed_bench/benchmarks/speed/ | 1 | #36029 |
| hf/philschmid/mt-bench | 1 | #24322 |
| hf/likaixin/InstructCoder | 1 | #24322 |
| sharegpt | 1 | #29600 |
| hf/facebook/voxpopuli | 1 | #32300 |
| spec_bench/question.jsonl | 1 | #32951 |
| timed_trace/conversation_trace.jsonl | 1 | #39795 |
| hf/gorilla-llm/Berkeley-Function-Calling-Leaderboard | 1 | #42457 |

#### Synthetic workload parameters

**43** synthetic entries use default parameters (no explicit input/output length, concurrency, or rate specified in the command). When `vllm bench serve` is run without these flags, it uses:

| Parameter | Default value |
|-----------|:--------------|
| `--dataset-name` | `random` |
| `--random-input-len` | `1024` |
| `--random-output-len` | `128` |
| `--num-prompts` | `1000` |
| `--request-rate` | `inf` (send as fast as possible) |
| `--max-concurrency` | unlimited |

**89** synthetic entries specify explicit parameters. Top configurations:

| Input len | Output len | Num prompts | Request rate | Concurrency | Count |
|----------:|----------:|------------:|:-------------|:------------|------:|
| — | — | 1000 | — | — | 5 |
| 2 | 128 | 128 | inf | — | 3 |
| 2048 | — | 2000 | inf | 64 | 3 |
| — | — | — | — | 64 | 3 |
| 2 | 256 | 1024 | inf | — | 2 |
| 1024 | 128 | — | — | — | 2 |
| 100 | 100 | 8 | — | — | 2 |
| 100 | 100 | 512 | — | — | 2 |
| 2 | 512 | 128 | — | — | 2 |
| 2 | 512 | 128 | inf | — | 2 |

#### Benchmark tools used

| Tool | Count |
|------|------:|
| vllm bench serve | 121 |
| vllm bench throughput | 15 |
| vllm bench latency | 11 |
| benchmarks/kernels/benchmark_moe.py | 4 |
| benchmarks/kernels/benchmark_moe_permute_unpermute.py | 3 |
| benchmarks/benchmark_prefix_block_hash.py | 2 |
| benchmarks/attention_benchmarks/benchmark.py | 2 |
| benchmarks/kernels/benchmark_router_gemm.py | 2 |
| benchmarks/kernels/benchmark_vit_fp8_attn.py | 2 |
| vllm bench sweep | 1 |
| benchmarks/benchmark_prefix_caching.py | 1 |

### Reading the results

- **Synthetic (76%)**: The dominant workload type. PRs use randomly generated token sequences with configurable lengths. The vllm bench tool defaults to `random` when no dataset is specified.
- **Recorded trace (20%)**: PRs benchmarked against real conversation datasets — primarily ShareGPT (a shared corpus of ChatGPT conversations) and HuggingFace datasets like MT-Bench.
- **Combination (3%)**: PRs that tested with both synthetic and real workloads.
- **Unknown (1%)**: Commands pointing to remote endpoints or with insufficient flags to classify. Candidates for the LLM fallback tier.
