# Rate-limit failures on the IOCR run — cause, fixes, and what's still missing

Branch: `yevgeny/spotlights-errors` (11 commits, `5f31334`..`01ce381`)
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

**These two sections replace an earlier claim in this document that the engine's
own unpaced concurrency produced the 429s. That claim was not supported by the
run's artifacts, and the artifacts were sufficient to test it — first by call
count, then, when call count turned out to be the wrong unit, by token
throughput. Both come back null.**

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

## The retest that mattered: tokens, not calls

Everything above counts *calls*. That is the wrong unit twice over, and both
faults had to be fixed before the result could be trusted:

1. **Rate limits are priced in tokens.** This run moved ~53.7M tokens in 4.3
   hours, ~208k/min sustained. A call count says nothing about that.
2. **Codex streams carry no `"timestamp"`**, so 80 codex sessions — averaging
   189k input tokens each — were invisible to the session-count timeline
   entirely. Half the run's traffic was never in the measurement.

`candidate_discovery/iterations.jsonl` closes both gaps: it records every
iteration's agent, `duration_s` and full token breakdown, and iterations run
strictly in sequence inside a module. So the claude iterations, which do carry
timestamps, anchor the codex ones — place the anchored iteration on the clock,
then walk the sequence forwards and backwards by `duration_s`. That puts **100
iterations across 26 modules on the wall clock, 48 of them codex sessions the
earlier analysis could not see**, against the 128 timestamped 429s. Reproduce
with:

```
python scripts/analyze_token_timeline.py <artifacts>/spotlights_manager
```

The token hypothesis fails in the same direction as the call-count one:

| Bucket | corr(all tokens, 429s) | corr(fresh tokens, 429s) | corr(call count, 429s) |
|---|---|---|---|
| 1 min | −0.087 | −0.165 | −0.084 |
| 2 min | −0.105 | −0.180 | −0.084 |
| 5 min | −0.178 | −0.295 | −0.209 |
| 10 min | −0.127 | −0.339 | −0.238 |

Dose-response on tokens is as flat as it was on calls — 0.61, 0.81, 0.49, 0.62,
0.50 rate limits per minute across quintiles running from 98k to 400k tok/min.
Minutes that contained a 429 were *lighter* than minutes that did not:
**209,545 vs 244,189 tok/min, a ratio of 0.86x.** And the four heaviest
token-minutes of the entire run — up to **570,042 tok/min** — took **zero** rate
limits between them. A ceiling anywhere near our own load would have bitten
exactly there.

One more figure points the same way. Of that ~236k tokens/min, only **~29k/min
was fresh input and output**; the rest was discounted cache reads (codex alone
served 82% of its 11.9M input tokens from cache). A sustained 29k/min of fresh
tokens is not a load that troubles any plausible limit.

A live check of the gateway agrees, and it went ten times higher than the
archived run ever did. `scripts/ratelimit_probe.py` ramped
1 → 2 → 4 → 8 → 16 → 32 → 64 genuinely concurrent requests at 14:48 IDT on a
Thursday: **127 calls, 127 successes, zero refusals, and not one rate-limit
header on any response, for $0.0063.** The archived run peaked at 6 concurrent.
Request concurrency is not the constraint at any level we can reach from here.
`retry-after`,
`x-ratelimit-*` and `anthropic-ratelimit-*` are absent everywhere — the gateway
never states a limit or a wait, so any backoff we build has to be sized by
measurement rather than read off the wire. The key's budget headroom also rules
quota exhaustion out, and the key itself is confined to `llm_api_routes`, so
`/key/info` returns 403 and no administrative answer is available to us at all.

**The honest limits of this result.** Three, and none of them rescues the burst
claim:

- **Reverse causality.** Being throttled makes you push *fewer* tokens, because
  you are waiting. That partly explains the negative sign, and it could mask a
  real positive effect. A clean test needs *offered* load; the artifacts record
  only *delivered* load.
- **Placement is unverified.** Every module had exactly one timestamped anchor,
  so there was no second anchor to check the sequence assumption against. Codex
  placement could drift by minutes.
- **Tokens are spread evenly across each iteration's `duration_s`**, which
  smooths real peaks — tokens actually move during `api_time_s` only. So any
  correlation found here is a floor, not a ceiling.

What is dead is the claim that pacing our own launches would have prevented
these 429s. It is dead on both yardsticks — call count and token throughput —
and now across both CLIs. A cap set anywhere in the 1–6 concurrent range we
already occupied, and a launch stagger, would have changed nothing here. What
the run cannot show is what happens at a concurrency well above 6 or a token
rate well above 570k/min, because it never went there.

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

### What was built for it: engine-level backoff (`01ce381`)

**Long jittered backoff, driven by `retry_class` from fix 4.** Opt-in
(`--agent-retry-attempts 1`), with `.attempt<N>` stream preservation so a retried
call keeps the evidence of why it was retried, and timeout-retry only where the
stream carries rate-limit evidence — a genuinely slow call must not be relaunched
just for being slow.

| Piece | Where |
|---|---|
| Policy: attempts, `base_s` 30, `max_s` 300, `delay_s()` = `uniform(0, min(max_s, base_s · 2^(n−2)))` | **new** `utils/agent_retry.py` |
| Driver: `_launch_with_retry`, wrapping both step-5 passes | `agent_proposals/api.py` |
| Knobs: `--agent-retry-attempts` / `--agent-retry-base-s` / `--agent-retry-max-s` | `cli.py` → `AgentProposalsConfig` |
| Recorded by value, so a run states the policy that produced it | `costing/manifest.py`, `orchestrator.py` |
| Three keys kept **out** of the resume fingerprint while attempts == 1 | `spotlights_manager/persistence.py` |

Two choices worth defending. **Full jitter, not a fixed delay**: parallel
candidates throttled by the same event would otherwise all come back at the same
instant and rebuild the burst. **The fingerprint exclusion**: `attempts=1` means
"behave exactly as before", so it has to *hash* exactly as before, or three new
keys at their defaults would make every half-finished run directory already on
disk fail `--resume` with `ResumeMismatchError` for a feature it never used. That
trap has been sprung here before; the precedent followed is the codebase's own
`_OPTIONAL_MODEL_FIELDS`. Turning the retry **on** does move the hash, which is
correct — it changes how an agent call is made.

Verified by 28 new unit tests (18 policy, 5 driver, 5 CLI/fingerprint), the
driver ones exercising the real backoff path including jitter at millisecond
delays rather than patching it out. Full `tests/unit` at the known baseline of 7
pre-existing failures; `ruff` clean on every touched file. The strongest of them
asserts that three attempts against a permanently rate-limited stream make
exactly three calls and leave `.attempt1`, `.attempt2` and the final stream all
on disk.

The sizing comes from the numbers above rather than from guesswork: to be worth
anything a retry has to wait far longer than the ~0.6 s the CLI already tried, and
the rate-limit spikes on this run persisted for minutes. 30 s is ~50x the 601 ms
median the CLIs already tried and failed with; the cap is 300 s because the spikes
lasted minutes, not hours. Recorded by value in `run_manifest.json`, as the
existing knobs are, so every run states the policy that produced it.

This is the only behavioural change made. A global concurrency cap and a launch
stagger were the earlier proposal and are **not** built: the two sections above
are the measurement that withdrew them, and the 64-concurrent probe closed the
one gap they might still have been hiding in.

### Trade-offs

- **Wall-clock.** Backoff makes a bad run *longer* — that is the trade, and the
  reason it ships opt-in and gets measured before any default changes.
- **A shared ceiling.** The flat per-call tax, the simultaneous multi-module
  spikes, and the 570k-token minute that took no refusals at all point the same
  way: the limit was not ours alone to spend. The whole team shares this gateway,
  each with their own key, and a ceiling shared with other people moves as they
  start and stop working — which is exactly what a load-independent 429 rate
  looks like. Waiting longer rides that out; it does not raise the ceiling.
- **Quota exhaustion is immune to it.** No backoff inside a run can fix an account
  out of budget, which is why fix 4 classifies it `fatal` rather than `retryable`.

### What would still settle it

Two things the archive cannot show, both cheap, and both about finding the ceiling
rather than explaining the run:

1. **Where the wall is.** *Half done.* One ramp to 64 concurrent has been run
   (14:48 IDT Thursday) and found no wall at all, so there is no knee yet to
   compare against. The comparison is still open: run
   `scripts/ratelimit_probe.py --label <when>` at several times of day. A knee
   that sits at the same place every time is a limit attached to our key, and
   pacing would fix it; a knee that wanders is a ceiling shared with the rest of
   the team, and only waiting helps. A full ramp costs a couple of cents — 127
   calls measured $0.0063.
2. **Whether backoff actually recovers.** That needs one real 429 from any source,
   not a reproduction of 19 Aug. Push hard enough to draw refusals, then show a
   long jittered wait gets through where the CLI's own ~0.6 s does not. This is
   the test that validates the retry above, and it is the only one that can — the
   unit tests prove the retry does what it is told, not that waiting works. 127
   concurrent-ramped calls drew no refusal at all, so provoking one is harder than
   it sounds, which is itself consistent with the archived 429s having needed a
   neighbour on the shared gateway to be busy at the same time.

Neither needs administrative access, which is just as well: our key is confined to
`llm_api_routes`, so there is no path to the limit value, the retry-after policy,
or the gateway's traffic log for 19 Aug from where we stand.

### Also still open

- Retry at the **step-2** discovery launch site. Fix 1 salvages a truncated
  discovery pass; it does not retry one, and all 8 module failures were 429s there.
- Folding `modules_extractor::api_failure_reason` into the shared classifier, so
  one table describes every agent-CLI failure in the engine rather than two.
- Turning the retry on by default, which needs the recovery evidence above plus a
  measured wall-clock cost on a run that actually hits trouble.

Built and tested, opt-in, unproven against a live refusal.
