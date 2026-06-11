# PR Data Filtering Rules

The filtering pipeline used by `repo_bench` to surface performance-relevant PRs.

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
