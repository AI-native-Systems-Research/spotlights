# Brief — live progress UI for `repo-bench run`

You are adding a **live multi-stage terminal progress UI** to the
repo-bench orchestrator. The pattern + reference implementation
you'll follow is in [live-terminal-progress-ui-guide.md](https://drive.local/live-terminal-progress-ui-guide.md)
(or wherever the project keeps it). Read that guide first; this brief
describes what's specific to this pipeline.

This is a **standalone session** — the existing pipeline tests must stay
green throughout. Touch only what's needed.

## Where the UI attaches

Two surfaces:

- **`spotlights_engine.repo_bench.run.benchmark(...)`** — the
  Python orchestrator API. Add a `progress: ProgressSink | None = None`
  parameter that's `None` by default (so library callers and tests don't
  see the UI).
- **`repo-bench run` CLI** (in
  `src/spotlights_engine/repo_bench/cli.py`). On `run` only,
  instantiate a `LiveProgressSink` and pass it to `benchmark(...)`.
  The other six subcommands (`aggregate`, `filter`, `fetch-diffs`,
  `curate`, `audit`, `score`) keep their plain logging — those are for
  iteration, not for the headline experience.

If `not sys.stdout.isatty()` (CI, piped output), fall back to plain
log lines — section 9 of the guide. **Do not** rip out the existing
`logging` calls; the live UI is additive.

## Stages

The benchmark has six stages, in order:

| # | Stage | Typical wall-clock | Substage progress |
|---|---|---|---|
| 1 | `aggregate` | 5–10 min if uncached; <1 s if `prs.jsonl` exists | total PRs scraped vs. discovered |
| 2 | `filter` | <1 s | none — atomic |
| 3 | `fetch-diffs` | 30 s for 24 PRs uncached; <1 s if cached | per-PR fetched/skipped |
| 4 | `curate` | 30 s × N PRs uncached; <1 s if all cached | per-PR done, with cache-hit count |
| 5 | `audit` | 30 s × N PRs uncached; <1 s if all cached | per-PR done, with cache-hit count |
| 6 | `score` | 5–10 min for tens of judge calls; <1 s if all cached | per-judge-pair done, with cache-hit count |

All times assume the 24-PR `2025-12-02__2026-06-03/127aad13f3d0` view.
Don't hard-code `24` or any view size — pipe the actual count from the
view file or from each stage's own counters.

## What each stage's progress bar should show

This is where this brief diverges most from the generic guide. Each
LLM-driven stage has two numbers worth surfacing:

- **`done / total`** — units processed. Drives the bar fraction.
- **`(cache hits / fresh calls)`** — split of where time is going.

Example detail strings:

```
4. Curate                  RUNNING   ███████░░░░░  58%  14/24  (3 cache, 11 fresh)  ETA 5m
5. Audit                   PENDING   ░░░░░░░░░░░░   0%
6. Score                   PENDING   ░░░░░░░░░░░░   0%
```

A cache-only run looks like:

```
4. Curate                  DONE      ████████████ 100%  24/24  (24 cache, 0 fresh)
```

That distinction is the headline value of having the UI in this
project: a 30-minute first run vs a 0.2-second cached re-run is the
*same numbers* but a wildly different experience, and the user should
see immediately which one is happening.

## ProgressSink protocol

Define a `ProgressSink` Protocol in
`src/spotlights_engine/repo_bench/progress.py`. The
orchestrator and the per-step internals call methods on it; the UI
implements them. A no-op default keeps tests and library use silent.

```python
from typing import Protocol, Literal

Stage = Literal["aggregate", "filter", "fetch-diffs", "curate", "audit", "score"]

class ProgressSink(Protocol):
    def stage_start(self, stage: Stage, *, total: int | None = None) -> None: ...
    def stage_step(
        self,
        stage: Stage,
        *,
        done: int,
        total: int,
        cache_hits: int = 0,
        fresh: int = 0,
        detail: str = "",
    ) -> None: ...
    def stage_done(self, stage: Stage, *, detail: str = "") -> None: ...
    def stage_failed(self, stage: Stage, *, error: str) -> None: ...
```

Rules:
- The orchestrator calls `stage_start` / `stage_done` for every stage.
- Stages with substage progress (curate, audit, score, fetch-diffs)
  call `stage_step` once per unit processed.
- Stages without substages (filter, sometimes aggregate when cached)
  emit `stage_start` then `stage_done` immediately.
- Failure path always emits `stage_failed` before re-raising.

## Where to plumb the sink in

Each per-step API needs a `progress: ProgressSink | None = None`
parameter, passed through from `run.benchmark`. **Do not** introduce a
module-level singleton — keeping it as an explicit kwarg means tests
can inject a stub and library callers can stay silent.

The four LLM-stage composers (`auto_curate`, `audit`, `score`, and
`fetch_diffs`) already loop over PRs / pairs internally. Inside each
loop, after the per-row work completes (cache hit OR fresh call),
emit `progress.stage_step(stage="curate", done=i+1, total=N, cache_hits=..., fresh=..., detail=f"PR #{pr_n}")`.

`aggregation.scrape` already prints progress every 25 PRs to its
logger — replace that with `progress.stage_step` and let the UI
choose the redraw cadence. (For a cached aggregate-skip, the
orchestrator emits one `stage_done` with `detail="cached"` and never
calls `stage_step`.)

## Render cadence

Don't redraw on every `stage_step` call — at 24 PRs over 30s each,
that's fine, but at 5,329 PRs in `aggregate` it'd be wasteful. Use the
guide's pattern: render whenever state changes, but cap to ≥250 ms
between full-screen redraws. Track `last_render_time` in the sink.

When a stage finishes, render unconditionally so the bar lands at 100%
visibly before the next stage takes over.

## CLI integration

In `cli.py::_cmd_run`, before calling `run.benchmark(...)`:

```python
from spotlights_engine.repo_bench.progress import LiveProgressSink, NullProgressSink

sink = LiveProgressSink() if sys.stdout.isatty() else NullProgressSink()
handle = run.benchmark(..., progress=sink)
```

The existing `print(...)` block at the end of `_cmd_run` (the summary
of the handle) stays. The live UI clears the screen during execution;
the summary is what the user sees after.

## What NOT to do

- **Don't add a dependency on `rich`, `tqdm`, `progressbar2`,** etc.
  Stick to stdlib + ANSI per the guide.
- **Don't background-thread the renderer.** The synchronous "mutate
  state then render" model the guide uses works fine here.
- **Don't print per-PR log lines while the live UI is on.** The
  existing `log.info(...)` calls in the composers print to stderr;
  redirect or suppress them while the UI is active (see "Log
  suppression" below).
- **Don't try to render a "global ETA"** that sums across stages. The
  stage-by-stage ETA the guide shows in §7 (per-stage rate × remaining
  units) is honest. A global ETA mixing aggregate (mostly cached) and
  curate (LLM-bound) gives misleading numbers.
- **Don't change the verbosity of per-step CLIs.** Only `run` gets the
  UI. `curate` / `audit` / `score` / etc. keep printing log lines —
  they're how the user iterates on a single stage.

## Log suppression while the UI is on

`LiveProgressSink` should, on construction:
1. Capture the current root logger level.
2. Set the level for `spotlights_engine.repo_bench.*` and
   `spotlights_engine.agent_proposals.*` loggers to `WARNING` (so
   info-level pipeline chatter doesn't garble the table).
3. Restore the original levels in a `close()` method (or `__exit__`).

The orchestrator's wrapping context manager calls `sink.close()` in a
`finally:` block.

## Failure handling

If any stage raises, the sink:
1. Marks that stage `failed` with the exception's `str()` (truncated to
   60 chars for the detail column).
2. Does a final render so the user sees the failed row in red.
3. Re-raises — the orchestrator's exception handling is unchanged.

Pending stages stay `pending` (gray) — that's the right visual: "we
got this far before failing."

## Testing

Add `tests/unit/repo_bench/test_progress.py`:

1. `NullProgressSink` accepts every method call without error.
2. A stub sink can be injected into `run.benchmark` (with all-stub
   agent runners — see existing `test_integration.py` for the
   pattern) and receives the expected `stage_start` / `stage_step` /
   `stage_done` sequence.
3. `LiveProgressSink` rendering is the harder one to test. Don't try
   to assert on terminal output. Instead:
   - Patch `clear_screen` to a no-op.
   - Capture `sys.stdout` to a buffer.
   - Run the demo flow and assert that final stdout contains:
     `"DONE"`, the final progress count, and no ANSI escape leaks
     past the last `RESET`.

The UI is additive, so no existing test should change. If a test
breaks, you've over-reached.

## Out of scope for this session

- Adding the same UI to per-step CLIs.
- A web dashboard / server-sent-events progress feed.
- Restoring scrolled output after the UI exits (the screen-clear
  approach the guide uses doesn't preserve prior terminal contents;
  that's a known tradeoff).
- Making `aggregate`'s 5,329-PR progress prettier than `done/total +
  ETA`. The single-stage view from the guide is fine.

## Suggested implementation order

1. `progress.py`: define `ProgressSink` Protocol + `NullProgressSink` +
   `LiveProgressSink`. Don't wire it in yet — just confirm
   `LiveProgressSink` renders correctly when driven by a hand-written
   demo (a copy of the guide's demo loop, but with the six real
   stages).
2. Plumb `progress: ProgressSink | None` through every per-step API
   (default `NullProgressSink`). Confirm the existing test suite still
   passes.
3. Wire emit-points inside the per-step composers (`auto_curate`,
   `audit`, `score`, `fetch_diffs`, `aggregation`). Check that running
   a per-step CLI without a sink is byte-identical to the prior
   behavior.
4. CLI: in `_cmd_run`, instantiate `LiveProgressSink` (or
   `NullProgressSink` when not a TTY) and pass it to
   `run.benchmark`. Test by running the cached `repo-bench
   run` (no LLM calls — every step a hit) and verifying the UI flips
   through six stages cleanly with all 100% green bars.
5. Add the `tests/unit/repo_bench/test_progress.py` tests.
6. Run the full repo-bench suite — it must still pass green.

Ship it as a single commit on a new branch
`yevgeny/progress-ui` (don't pile it onto the repo-bench
branch — it's an independent enhancement worth reviewing on its own).
