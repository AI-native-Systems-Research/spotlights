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

## What is still missing: the burst

**None of the five fixes reduces the load we put on the API.** They make failures honest and
far less destructive; the number of agent CLI sessions the engine fires, and when, is
unchanged. Fix 1 slightly *increases* load, because candidates that used to be discarded now
carry on into steps 4 and 5.

Today each step limits its own parallelism — `max_parallel_sessions`,
`max_parallel_pairs`, `max_parallel_candidates` — but there is **no cap on the total number
of agent CLI sessions in flight** across the six launch sites, and **no stagger**: whatever
is scheduled together hits the API in the same instant. That is what produces the 429s.

### Proposal

**Stage B — stop creating the burst.** One global concurrency gate across all six agent-CLI
launch sites, plus a launch stagger, both shipping opt-in
(`max_concurrent_agent_calls=None`, `agent_launch_stagger_s=0.0`) so nothing changes until
we ask for it. Recorded by value in `run_manifest.json`'s `run_config`, as the existing
parallelism knobs are, so every run states the policy that produced it.

**Stage C — recover from the 429s that still land.** Long jittered backoff, driven by
`retry_class` from fix 4. This is needed because both CLIs already retry *fast* (claude backs
off ~562 ms) and *shallow* (10 attempts) before reporting failure — far too short for a real
rate-limit window, which is why a `retryable` verdict is only actionable by a caller prepared
to wait much longer. Also opt-in (`agent_retry_attempts=1`), with `.attempt<N>` stream
preservation, and timeout-retry only where there is rate-limit evidence in the stream.

### Trade-offs to decide before making either default

- **Wall-clock.** Lower concurrency means longer runs. That is the whole trade, and the
  reason both stages ship opt-in and get measured first.
- **A shared ceiling.** If the rate limit is account-wide and something else is consuming it,
  pacing our own run only helps so far.
- **Quota exhaustion is immune to both.** No backoff inside a run can fix an account that is
  out of budget, which is why fix 4 classifies it `fatal` rather than `retryable`.

Both stages are designed and specced. Neither is built.
