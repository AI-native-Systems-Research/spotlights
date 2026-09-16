# Rate-limit failures on the IOCR run — cause, fixes, and what's still missing

Branch: `yevgeny/spotlights-errors` (5 commits, `5f31334`..`147afae`)
Reference run: `page_latency-2026-08-19-ophir-new-module-extractor` — 30 modules, $123.60,
3 SUCCEEDED / 14 DEGRADED / 8 FAILED / 5 SKIPPED

## The cause

**API rate limiting**, and an engine that misreported every instance of it.

- **Step 2 (candidate discovery)** — all 8 module failures were 429s. Every one was reported
  as `DiscoveryValidationError: schema parse failed twice`.
- **Step 5 (agent proposals)** — 44 candidate failures, reported as `claude timed out after
  600.0s` (30) or `exit=1: stderr=''` (14). Replaying the archived streams shows ~33 of them
  name a rate limit or a server error. The 600 seconds went to the CLI's *own* retry loop.

The evidence was on disk the whole time. Both agent CLIs emit newline-delimited JSON events
and both record the terminal failure there — `<candidate>.<agent>.stdout` exists for exactly
the failed candidates, all 44 of them. Nothing needed to be collected; the failure path
simply never read the stream, and described each death by whatever the *caller* happened to
know.

Two consequences, and the second is the expensive one:

1. Every rate limit was labelled as something else, so the run looked like it had a schema
   bug and a latency problem.
2. Step 2 treated a failure in *any* iteration as invalidating the whole module, discarding
   88 of 214 candidates across the run (41%) — including 11 valid ones from a module whose
   only fault was a 429 in its final refinement pass.

## What was delivered

| # | Commit | Fix | Why | How | Outcome | Runtime / candidate impact |
|---|---|---|---|---|---|---|
| 1 | `5f31334` | Step 2 blamed the schema for rate limits and dumped all candidates on any failure | 8/8 module failures were 429s reported as "schema parse failed twice"; 41% of candidates thrown away | Read the agent's own stream for the terminal event; keep what completed iterations found and record a `DiscoveryTruncation` instead of raising | Rate limits are named as rate limits; a late 429 costs one refinement pass, not the module | **Yes, both.** +88 of 214 candidates survive and flow into steps 4–5 → more cost and runtime on a run that hits trouble; status shifts FAILED → DEGRADED. Clean runs unchanged |
| 2 | `99626cc` | Fix 1 salvaged *every* discovery failure, including ones that deserved to fail | Two failures look identical at that point: the CLI **cut off mid-flight** (429/timeout — salvage is right) vs. the CLI **finished and returned the wrong shape** (a real, repeatable bug). Salvaging the second would report a shape bug as "truncated by a rate limit, kept 7 candidates" — one lie swapped for another | Tag each parse failure: no/broken JSON → environmental → salvage; JSON that parses but violates the schema → contract → module FAILS | The salvage covers only what it was meant to. A wrong payload still fails loudly | **Yes — narrows #1.** Contract breaks fail again (0 candidates), as before #1 |
| 3 | `2aceb1c` | Step 5's 44 failures were all mislabelled as timeouts or `exit=1` | ~33 were rate limits or server errors; the 600s was the CLI's own retry loop | New `utils/agent_stream.py` (`describe` / `retry_class`) — classification only, no subprocess, concurrency or retry code, so it cannot change how a call is made. Step 5 appends the cause and keeps today's text as the prefix | The real cause reaches `status.json` and `result.json`; anything grepping `timed out` or `exit=` still matches | **No.** Error text only |
| 4 | `92f114a` | The retry verdict read the whole retry history rather than the failure | Found by Codex review of #3. `retry_statuses` is cumulative, so a 429 the stream *survived* could vouch for the 401 that actually killed it — i.e. retry an auth failure forever, the exact outcome the fatal rule exists to prevent | Verdict reads the final retry event alone; the cumulative list stays for the human-readable text | A survived 429 no longer rescues a trailing 401, and an earlier 401 no longer poisons a trailing 429 | **No.** `retry_class` has no caller yet — it exists for the retry stage below |
| 5 | `147afae` | Wrapper modules paid a full discovery pass to analyze an `__init__.py` | Some folders in the tree hold no real code: their only own file is a short `__init__.py` re-exporting the sub-folders beneath them, and the code lives in those sub-folders, which are analyzed as targets in their own right. A pass on the wrapper *cannot* find anything — a candidate must live in the module's own files, and `Validator` drops anything inside a submodule. 4 of the 30 IOCR modules, ~$3/run | `--skip-container-modules`, off by default. Marks the module SKIPPED before step 2 — and before the session semaphore, so it never takes a slot — with the reason on the checkpoint and in the log. Persists an empty `candidates.json` so a resume treats it as settled instead of re-running discovery | Opt-in saving that says *why* a module was never discovered, with nothing lost | **Yes, when enabled.** −4 modules, −4 discovery passes; candidate count drops by whatever those wrappers would have produced (likely 0, unverified). **Off by default = no effect** |

### Why fix 5 is off by default

I could not confirm from the archived run that those four wrappers produce nothing — a
lazy-import candidate in an `__init__.py` is plausible under a latency objective, and the
archived run's SKIPPED list is no longer on the machine. A single ~$0.76 module run would
settle it. Until then the flag is opt-in.

The 50-non-blank-line bound in the predicate is a guard, not the discriminator: wrappers are
selected on the *shape* of their own files (all `__init__.py`). Measured on the IOCR tree the
four wrappers hold 9, 9, 9 and 18 non-blank lines and the smallest non-wrapper module holds
96, so any value in that gap selects the same four.

### Verification

- 13 new unit tests for fix 5; full `tests/unit` at the known baseline of 7 pre-existing
  failures.
- `ruff` clean on every touched file. `mypy` shows the same 3 pre-existing `orchestrator.py`
  errors, confirmed by stashing the changes and re-running at HEAD.
- Fixes 1–4 were reviewed by `codex exec review`, which found the defect that became `92f114a`.
- The real evidence for fix 3 is the **replay** of the 44 archived step-5 streams, not a live
  run: fixes 1–4 change only text on the failure path, so a healthy run proves no regression
  and nothing more.

## Was it our own burst? Measured against the archive: no

**This section replaces an earlier claim in this document that the engine's own
unpaced concurrency produced the 429s. That claim was not supported by the run's
artifacts, and the artifacts were sufficient to test it.**

The test became possible because claude's stream events carry a wall-clock
`"timestamp"`. That is the only per-call clock in the archive — `run_manifest.json`
holds aggregates only, this run predates `a32fb4a` so its `run_config` is empty,
`manifest.json` carries no times, and all 30 module `status.json` files share a
single plan-time `started_at` (a module window is queue wait *plus* work, and
reading it as concurrency overstates it by 5.5x). Timestamping each stream instead
gives a real interval for all 77 claude sessions and the exact moment of every
rate limit. Reproduce with:

```
python scripts/analyze_stream_timeline.py <artifacts>/spotlights_manager
```

### What the streams prove

Rate limiting was pervasive, and worse than the failure counts suggested:

| Measurement | Value |
|---|---|
| claude `429 rate_limit` retry events | **149** |
| codex streams that gave up at `429` | **22** |
| `502 server_error` retries | 8 |
| claude calls that hit at least one 429 | **43 of 77 = 56%** |

### What they disprove

The engine's real pacing was **mean 2.9 concurrent claude sessions, median 3,
max 6** — not a burst. And four independent tests all fail to link our own
concurrency to the rate limits:

| Test | Result |
|---|---|
| corr(calls in flight, 429 count) per time bucket | **+0.20** at 2-min and 5-min resolution |
| Peers in flight when a call launched | 429-hit **2.60** vs clean **2.74** — *wrong direction*, p=0.71 |
| Other calls launched within ±30/60/120/300s | null at every window (p=0.36–0.91) |
| Dose-response, 429s per call by concurrency | **flat and non-monotonic**: 1.90 at 2–3 peers, 2.29 at 3–4, 1.60 at 4–5, 1.86 at 5+ |

Two further observations point the same way. The per-call rate limit rate is a
roughly constant tax of 0.9–1.9 per in-flight call in *every* 40-minute window of
the run, rather than something concentrated at peaks. And the busiest window of
the whole run — 8–9 calls in flight around +165 min — took **zero** rate limits,
while each of the five biggest 429 spikes struck **3–5 different modules
simultaneously**. Many modules throttled at the same instant, independent of how
many calls we had running, is the signature of a shared ceiling we were not
setting.

**The honest limit of this result.** It shows no *marginal* effect anywhere in the
1–6 concurrent-session range we actually operated in. It cannot show what would
happen at 1, because this run never went there. So the claim that is dead is
"pacing our own launches would have prevented these 429s"; a cap set anywhere in
the range we already occupied, and a launch stagger, would have changed nothing
here.

## What the evidence does support: recover from the 429s, do not try to out-pace them

The actionable gap is not how many calls we launch — it is that neither CLI waits
long enough to survive a rate-limit window:

| Measurement | Value |
|---|---|
| Retry events across the whole run | 169 |
| Median backoff between attempts | **601 ms** |
| Longest single backoff observed | 32.7 s |
| **Total time every CLI spent backing off, whole 4.3-hour run** | **316 s** |
| Deepest retry attempt reached | 7 (of `max_retries` 10) |

Against that, 30 step-5 candidates each burned the full 600 s wall — **5 hours
lost to timeouts against 316 seconds spent waiting on purpose, a factor of 57.**
The CLIs retry fast, shallow, and then hand back a stream that says `rate_limit`
while the caller reports a timeout. That is the whole failure mode.

### The proposal: engine-level backoff

**Long jittered backoff, driven by `retry_class` from fix 4.** Opt-in
(`agent_retry_attempts=1`), with `.attempt<N>` stream preservation so a retried
call keeps the evidence of why it was retried, and timeout-retry only where the
stream carries rate-limit evidence — a genuinely slow call must not be relaunched
just for being slow.

The sizing comes from the numbers above rather than from guesswork: to be worth
anything a retry has to wait far longer than the ~0.6 s the CLI already tried, and
the rate-limit spikes on this run persisted for minutes. Recorded by value in
`run_manifest.json`'s `run_config`, as the existing knobs are, so every run states
the policy that produced it.

This is the only change proposed. A global concurrency cap and a launch stagger
were the earlier proposal and are **not** proposed: the section above is the
measurement that withdrew them.

### Trade-offs

- **Wall-clock.** Backoff makes a bad run *longer* — that is the trade, and the
  reason it ships opt-in and gets measured before any default changes.
- **A shared ceiling.** The flat per-call tax and the simultaneous multi-module
  spikes both suggest the limit was not ours alone to spend. Waiting longer rides
  that out; it does not raise the ceiling.
- **Quota exhaustion is immune to it.** No backoff inside a run can fix an account
  out of budget, which is why fix 4 classifies it `fatal` rather than `retryable`.

Designed, not built.
