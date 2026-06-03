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

## 1. Sonnet (instead of Opus) for stages 02 and 04

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

---

## Combined picture

| Lever | Wall saved | Cost saved | Effort | Risk |
|---|---|---|---|---|
| Sonnet for 02/04 | ~3 min | ~$1 | half day | medium |
