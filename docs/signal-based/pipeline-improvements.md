# Signal-pipeline improvements — backlog

Future improvements to the `signal_pipeline` package, deferred from the
initial implementation. **This is a working backlog: when an item is
implemented, remove it from this file.** When all items are implemented,
delete this file.

Numbers below are grounded in the live smoke run on
`data/20260525T202105Z_util0.4_mem16_lru/` against the `vllm` checkout
(commit `889a0dd9` on `signal-based/pipeline-impl`):

| Stage | Wall | Cost | Turns |
|---|---|---|---|
| Signal extraction (stage 01) | 5m 04s | $0.87 | 29 |
| ProjectTree extraction (stage 02) | 2m 47s | $0.62 | 25 |
| Candidate generation (stage 03) | ~5 min | ~$0.5–1 | n/a |
| **Total** | **~13 min** | **~$2** | |

---

## 1. Parallelize stages 01 and 02

**What.** Run signal extraction (stage 01) and ProjectTree extraction
(stage 02) concurrently. They have disjoint declared upstreams
(`s01.upstream=()`, `s02.upstream=()`) — neither needs the other's
output. Candidate generation (stage 03) waits for both.

**How.** [runner.py:run_pipeline](../../src/spotlights_engine/signal_pipeline/runner.py)
walks `selected()` in declared order today. Build a small dependency
graph from `STAGES[s].upstream`, find stages whose upstreams are all
complete, launch in parallel via `concurrent.futures.ThreadPoolExecutor`.
Guard `status.json` writes with a lock (or have each stage own its
slot atomically).

**Saves.** ~3 min off every full pipeline run. Bounded by
`min(s01_time, s02_time)`. **Zero cost saving** — same tokens billed.

**Effort.** ~80 LoC plus one test asserting wall-clock ≈ `max(s01, s02)`.

**Risk.** Low. Two `claude -p` subprocesses concurrent on the same API
key; CLI is rate-limit-aware. Log dirs already separate.

---

## 2. Sonnet (instead of Opus) for stages 02 and 04

**What.** ProjectTree extraction (stage 02) and change generation
(stage 04) are transformation/structuring tasks, not reasoning-heavy.
Sonnet is roughly 3× faster and ~1/5 the cost on equivalent prompts,
with negligible quality drop on these tasks. Keep Opus on signal
extraction (stage 01), candidate generation (stage 03), and execution
backend (stage 05) — those need deeper reasoning.

**How.** Add `model: str | None = None` to
[claude_subprocess.py:run_claude](../../src/spotlights_engine/signal_pipeline/claude_subprocess.py),
pass `--model <id>` to `claude -p` when set. Each stage decides its
own model. Stages 02 and 04 pass `model="claude-sonnet-4-6"`; rest
inherit the user's CLI default.

`modules_extractor` (used by stage 02) lives on `origin/main` and
doesn't accept a model arg today. Either: (a) shell out to `claude`
ourselves bypassing modules_extractor, (b) PR to main adding
`ExtractorConfig.model`, or (c) set `ANTHROPIC_MODEL` env var before
invoking modules_extractor (if claude CLI honors it).

**Saves.**
- Stage 02: 2m 47s → ~50–70 s. Cost $0.62 → ~$0.13.
- Stage 04: ~30s/candidate → ~10s/candidate; with 5 candidates ~1.5
  min off. Cost roughly proportional.
- Total: ~3 min wall-clock, ~$1 cost (~50% reduction).

**Effort.** Half day, including a quality A/B against the existing
LRU smoke baseline.

**Risk.** Medium. ProjectTree `description`/`role` fields might get
terser on Sonnet — likely fine but worth checking. Change generation
quality matters more for stage 05; if `Change.mechanism` weakens,
stage 05 produces worse edits. A/B before committing.

---

## 3. Cross-run ProjectTree cache

**What.** Within a run, `--resume` skips stage 02 if
`02_projecttree.json` exists. Across runs (different `--artifacts-dir`), it
re-extracts every time even when the subject repo's HEAD hasn't
moved. Adding a cache keyed by `(subject_root, git rev-parse HEAD)`
makes ProjectTree extraction free on every subsequent run against the
same subject checkout.

**How.** Cache at `~/.cache/spotlights-engine/projecttree/<sha>.json`
or in-repo `.projecttree-cache/<sha>.json` (`runs/` is already
gitignored; this would live alongside).
[s02_projecttree.py:run](../../src/spotlights_engine/signal_pipeline/stages/s02_projecttree.py)
checks the cache before calling `_extract_project_tree`. Hit →
return cached `ProjectTree`. Miss → extract, write, return.

Cache invalidates automatically by SHA change. Manual `rm` to force
re-extract. Optionally include a hash of `git status --porcelain`
output in the key so uncommitted edits also invalidate.

**Saves.** ~3 min and ~$0.62 off every run-after-the-first against
the same subject checkout.

**Effort.** Half day, plus one test asserting cache hits return
byte-identical ProjectTree.

**Risk.** Low. Stale cache only if subject working tree changes
locally without HEAD moving — mitigated by hashing `git status` into
the key. Schema drift over time mitigated by including a schema
version in the cache filename.

---

## 4. Tighten signal-extraction prompt with concrete file hints

**What.** [prompts/signal_extraction.md](../../src/spotlights_engine/signal_pipeline/prompts/signal_extraction.md)
lists likely files (`traces.jsonl`, `metrics.jsonl`, `vllm_server.log`,
`otelcol.log`) as guidance, explicitly noted as "not a contract."
That's the right design for telemetry-format flexibility, but it cost
us 29 turns on the LRU smoke because the agent re-explored from
scratch (ls → head → wc → jq → repeat per file). Adding a
**recommended exploration order with concrete `jq`/`awk` invocations**
for known patterns lets the agent skip ahead.

**How.** Edit the markdown, no code change. Add a section like:

> If `traces.jsonl` is present with OTel `resourceSpans` shape, your
> first pass should be: count spans by name with this exact `jq`
> invocation, then for each span name compute `count` and duration
> percentiles in ms. If the file's structure differs, fall back to
> exploration.

**Saves.** ~2–3 min off signal extraction (29 turns → ~12–15).
Roughly proportional cost saving (~$0.30–$0.40).

**Effort.** ~15 min.

**Risk.** Medium-low. The flexibility goal slips a bit — if the next
telemetry capture's `traces.jsonl` has a different shape, the
suggested `jq` will fail. The agent should fall back to exploration on
error, but might confidently follow stale guidance. Mitigation: phrase
recommendations as "if these files exist with this structure, here's
the fast path; otherwise explore freely."

**Recommendation.** Hold off until telemetry-capture format stabilizes
across team captures. Right now it's still in flux per the project
state — better to keep the prompt flexible at the cost of extra turns
than risk it going stale on the next capture.

---

## Combined picture

| Lever | Wall saved | Cost saved | Effort | Risk |
|---|---|---|---|---|
| Parallelize 01+02 | ~3 min | $0 | 90 min | low |
| Sonnet for 02/04 | ~3 min | ~$1 | half day | medium |
| Cross-run PT cache | ~3 min on 2nd+ runs | ~$0.62 on 2nd+ runs | half day | low |
| Tighten BA prompt | ~2–3 min | ~$0.35 | 15 min | medium-low |
| **All four (first run)** | **~6–8 min off 13 min** | **~$1.4 off $2** | ~1.5 days | |
| **All four (subsequent runs against same subject)** | **~9–10 min off 13 min** | **~$2 off $2** (PT cached, only 01+03+04+05 fire) | | |

Cheapest, highest-impact pair to start with: **parallelize 01+02** + **tighten signal-extraction prompt**. ~2 hours total, ~5 min off every run, both low-risk. Sonnet routing and the ProjectTree cache pay off later when iterating frequently.
