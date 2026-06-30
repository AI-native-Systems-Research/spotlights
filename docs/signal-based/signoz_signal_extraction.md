# Stage 01 — SigNoz-backed signal extraction (design)

How stage 01 (signal extraction) gains a second telemetry source: a run's
OpenTelemetry data read **live from SigNoz/ClickHouse by `run_id`**, alongside
the existing file-based path. This document is the design contract; it does not
describe code that exists yet.

Related:
- Current stage behaviour: [`src/spotlights_engine/signal_pipeline/README.md`](../../src/spotlights_engine/signal_pipeline/README.md)
  and [`stages/s01_signal_extraction.py`](../../src/spotlights_engine/signal_pipeline/stages/s01_signal_extraction.py).
- Producer/consumer data contract (source of truth for the SigNoz schema):
  `discovery_observability/docs/signoz-data-model.md` (sibling repo).
- Output schema: `Signals` in [`schemas.py`](../../src/spotlights_engine/signal_pipeline/schemas.py).

---

## 1. Goal & non-goals

**Goal.** Run the existing signal-extraction stage against a run that lives in
SigNoz, selected by `run_id`, without changing anything downstream. The stage
still emits the same `Signals` (`{workload, traces, anomalies}`) object.

**Non-goals.**
- No change to the `Signals` schema or to stages 02–05, the report rollup,
  resume, or `--inject`.
- No telemetry parser in Python. Telemetry interpretation stays in the agent +
  prompt, exactly as the file-based path does today.
- No ingestion/replay work — that is the sibling `discovery_observability`
  project. We are a read-only consumer.

## 2. Decisions (settled)

1. **Additive, flag-selected.** Keep `--telemetry-from` (file/dir) working
   as-is; add a SigNoz source alongside it (mutually exclusive). `--signoz`
   is the source selector; `--run-id` refines *within* it (see §6):
   - `--signoz` alone — auto-select the run: one run in the DB → use it (no id
     to type); more than one → **use the latest and emit a notice** telling the
     user the count and how to pick another.
   - `--signoz --run-id <id>` — analyze that specific run. `--run-id` is only
     valid together with `--signoz` (error otherwise).
   Fixtures and offline runs keep working unchanged.
2. **Agent-driven, like today.** The stage spawns one `claude -p` session that
   *decides for itself which queries to run* — the SQL analogue of today's
   "agent decides which `jq` passes to make." The prompt is the iteration
   surface; telemetry shape can evolve without code changes here.
3. **Everything via SQL.** All telemetry access is ClickHouse SQL through
   SigNoz's `clickhouse_sql` query API. No JSONL is materialised and re-parsed.
4. **Read-only, defence in depth.** The agent authors arbitrary SQL, so the
   path is read-only at two layers (CLAUDE.md §5):
   - **Hard control — a Viewer-scoped SigNoz key.** Server-enforced; an agent
     cannot talk its way past it. This is the real guarantee. The current
     `.env` key is already Viewer-scoped; `.env` now warns that any
     replacement key must be Viewer-scoped too.
   - **Soft layer — a defensive prompt instruction** that only read-only
     `SELECT` queries are permitted. This is *not* a security control on its
     own (the agent could ignore it); it just avoids wasted turns attempting
     writes and makes intent explicit.

## 3. Why agent-authored SQL (and where the guardrails are)

Giving the agent full SQL — rather than a fixed menu of pre-baked queries —
preserves the property that makes the file path robust: the model adapts to
what is actually present (new metric families, renamed series, extra signals)
without a code change. The new DCGM GPU metrics (`DCGM_FI_PROF_SM_ACTIVE`,
`SM_OCCUPANCY`, `PIPE_TENSOR_ACTIVE`, …) are picked up "for free" the same way
a new file would be today.

The risk of free-form SQL is that the agent re-derives — and re-breaks — the
same awkward bits every run. The line we draw: **run selection and run scoping
are canonical (tool-owned, identical every run); only the analytics are the
agent's.** Anything involving `run_id` is plumbing that never varies, so the
agent never hand-writes it — it is the exact surface that breaks (wrong join,
forgotten filter, `!= ''` instead of `= '<id>'`).

| Awkward bit | Where it's absorbed |
|---|---|
| Split response shape (`labels` strings + `values[].value` numerics) | `signoz-sql` merges rows; the agent always sees clean dicts. |
| Metrics fingerprint-join; per-signal `run_id` placement (resource attr vs JSON label); self-telemetry (`run_id=''`) exclusion | **Canonical scoping CTEs injected by `signoz-sql --run <id>`** (§4.1). `run_id` appears exactly once, in tool-owned SQL. The agent queries stable aliases (`spans` / `logs` / `metric_samples`), never raw tables. |
| Selecting a run / time windows | `run_id` does the selecting (time-irrelevant per the producer doc); the agent does no time handling. |

Everything analytical (quantiles, group-bys, correlations, the GPU-busy
question) is the agent's call, server-side, per run — but always *on top of* the
canonical run-scoped CTEs, never by composing its own `run_id` filter.

## 4. Components

```
signal_pipeline/
├── signoz_tool.py     # NEW: `signoz-sql` — the agent's one read-only SQL tool
├── schemas.py         # SignalPipelineInput gains `run_id: str | None`
├── cli.py             # NEW --signoz / --run-id (mut. excl. with --telemetry-from)
├── stages/
│   └── s01_signal_extraction.py   # NEW branch: run_id -> _extract_signals_via_signoz
└── prompts/
    └── signal_extraction_signoz.md  # NEW: SQL-driven prompt + schema cheat-sheet
```

### 4.1 `signoz_tool.py` — the one new primitive

This is the SQL analogue of the `jq` the file path already relies on: the agent
needs *some* command that can run a query against SigNoz and hand back usable
rows. No such command exists on the system, so we provide exactly one — a small
(~60-line) module, not a library/CLI split.

**Why a tool and not raw `curl`** (the agent could in principle hand-build the
request itself; we don't let it, for three correctness/safety reasons):
1. **Secret hygiene.** The pipeline logs raw stdout/stderr *and the prompt* to
   `_logs/`. A `curl -H "SIGNOZ-API-KEY: …"` would leak the gitignored key into
   those logs (CLAUDE.md §5). The tool reads the key from `.env` itself, so it
   never appears in the agent's command line or logs.
2. **Split response shape.** SigNoz returns `labels` (strings) and
   `values[].value` (numerics) in separate arrays; the tool merges them once,
   instead of the agent re-deriving the reshape (and getting it wrong) per call.
3. **Envelope boilerplate.** Each query is JSON-escaped SQL nested inside the
   `compositeQuery.chQueries` body — fragile to hand-assemble every call.

Interface (invoked via `Bash`, read-only):
- `signoz-sql --list-runs` → distinct non-empty `run_id` across traces (canonical).
- `signoz-sql --run <id> "<SQL>"` → run the agent's analytical SQL **on top of
  canonical run-scoped CTEs**; returns clean JSON rows. This is the only way the
  agent queries run data — it references the injected aliases, never raw tables.

The `--run <id>` mode prepends a fixed CTE block (the `<id>` substituted exactly
once) and appends the agent's SQL:

```sql
WITH
  spans AS (SELECT * FROM signoz_traces.signoz_index_v3
            WHERE resources_string['run_id'] = '<id>'),
  logs  AS (SELECT * FROM signoz_logs.logs_v2
            WHERE resources_string['run_id'] = '<id>'),
  run_fp AS (SELECT fingerprint FROM signoz_metrics.time_series_v4
             WHERE JSONExtractString(labels,'run_id') = '<id>'),
  metric_samples AS (SELECT s.metric_name, s.unix_milli, s.value
                     FROM signoz_metrics.samples_v4 s
                     INNER JOIN run_fp t ON s.fingerprint = t.fingerprint)
<agent SQL over spans / logs / metric_samples>
```

So the fingerprint-join, the per-signal `run_id` placement, and the
self-telemetry exclusion are all canonical and unguessable; the agent only
authors analytics over the three stable aliases. (Time windows are irrelevant —
`run_id` does the selecting — so there is no `--window` query in the agent's
surface; the run window, if ever needed for display, is derived separately.)

Internals (all testable functions in the one module): substitute `<id>` into the
canonical CTE template, build the `clickhouse_sql` envelope with `start`/`end` at
a wide retention bound, POST to `/api/v4/query_range` with the read-only key from
`.env`, merge the split response into row dicts. `list_runs()` and the CTE
template are the only fixed SQL here — **no analytical SQL lives in the tool**.
Registered as a console script, or invoked as
`python -m spotlights_engine.signal_pipeline.signoz_tool`, so it is callable from
the agent's working directory.

### 4.3 Stage wiring
`SignalPipelineInput` gains `signoz: bool = False` and `run_id: str | None`
(new). `cli.py` adds `--signoz` (source selector) and `--run-id` (refiner).
Validation at parse time: `--signoz`/`--run-id` are rejected if combined with
`--telemetry-from`, and **`--run-id` requires `--signoz`**.

Stage 01 resolves the concrete run before extraction:

```
if ctx.signal_input.signoz:                  # SigNoz source
    if ctx.signal_input.run_id:              #   refined to a specific run
        run_id = ctx.signal_input.run_id
    else:                                    #   auto-select
        runs = list_runs()                   #   via signoz-sql --list-runs
        if not runs:
            raise SignalExtractionError("no runs found in SigNoz")
        run_id = max(runs)                   #   latest: run_id sorts chronologically
        if len(runs) > 1:
            warn_loudly(run_id, runs)        #   prominent multi-line warning (below)
    return _extract_signals_via_signoz(run_id, ctx.log_dir, ctx.on_event, ctx.model)
# else: existing pre-cooked / agent-on-dir / synthetic branches, unchanged
```

When more than one run exists the auto-pick is **non-blocking but loud**: a
prominent multi-line warning (banner-style, not a single buried line) goes to
the live progress stream (`ctx.on_event`) and the stage log, naming the chosen
`run_id`, the total count, a capped list of the other runs, and how to override
(`--signoz --run-id <id>`). No prompt, no `--yes` flag, identical in CI and
interactively. `max(runs)` is a plain string max — the `YYYYMMDDTHHMMSSZ` UTC id
format sorts chronologically, so "latest" needs no timestamp query.

`_extract_signals_via_signoz` mirrors `_extract_signals_via_claude`: one
`claude -p` session, tools `Read` + `Bash`, `permission_mode="plan"`, the same
`{workload, traces, anomalies}` output schema, the new prompt. `cwd` is a
scratch/log dir (there is no telemetry directory on this path).

### 4.4 The prompt (`signal_extraction_signoz.md`)
A SQL-driven rewrite of `signal_extraction.md`. The agent only writes correct
SQL if it understands the SigNoz/ClickHouse layout, so the prompt **must teach
that layout explicitly**, sourced from the proven producer contract — the
telemetry-creation project's `discovery_observability/docs/signoz-data-model.md`
(the authoritative, live-verified description of tables, `run_id` placement,
the metrics fingerprint-join, families, and units). That document is the source
of truth; the prompt carries a compact cheat-sheet **distilled from it and kept
in sync with it** (if the producer changes the schema, update that doc first,
then this prompt). The cheat-sheet covers:
- the **canonical scoped aliases** the agent queries — `spans`, `logs`,
  `metric_samples` (§4.1) — and their columns. The prompt is explicit that the
  agent **never writes `run_id` filters, the fingerprint-join, or raw table
  names**; scoping is the tool's job. (This also means the agent never has to
  worry about the ~440 untagged self-telemetry records — they're outside the
  scoped CTEs.)
- **units**: `gen_ai.latency.*` are seconds, `durationNano` is nanoseconds;
- stable span names (`llm_request`, `POST /v1/chat/completions`, …) and metric
  families (`vllm:*`, `DCGM_FI_*`, `system.*`, `http_*`);
- worked example queries **over the aliases** (e2e-latency quantiles from
  `spans`, span-name counts, token distributions, `vllm:*` series from
  `metric_samples`, DCGM GPU-utilisation) — none of them mention `run_id`;
- the same anomaly guidance as today **plus** the GPU-from-DCGM family that the
  file path could never reach;
- a **defensive read-only instruction**: only `SELECT` queries are permitted;
  never attempt `INSERT`/`ALTER`/`DROP`/DDL (the Viewer key would reject them
  anyway — this just sets expectations and saves turns);
- emit the same `Signals` JSON.

## 5. The SigNoz data model we rely on (verified live)

From `discovery_observability/docs/signoz-data-model.md`, re-verified against
run `20260615T073440Z` on the shared instance:

| Signal | Table | `run_id` access | Select-a-run filter |
|---|---|---|---|
| Traces | `signoz_traces.signoz_index_v3` | `resources_string['run_id']` | `resources_string['run_id'] = '…'` |
| Logs | `signoz_logs.logs_v2` | `resources_string['run_id']` | `resources_string['run_id'] = '…'` |
| Metrics | `signoz_metrics.samples_v4` + `time_series_v4` | `JSONExtractString(labels,'run_id')` | fingerprint join (below) |

```sql
-- A run's metric samples — time window irrelevant, run_id does the selecting
SELECT s.metric_name, s.unix_milli, s.value
FROM signoz_metrics.samples_v4 s
INNER JOIN (
    SELECT fingerprint FROM signoz_metrics.time_series_v4
    WHERE JSONExtractString(labels,'run_id') = '<RUN_ID>'
) t ON s.fingerprint = t.fingerprint
```

Verified for `20260615T073440Z`: 43,491 spans; **350,160 metric samples across
196 metrics** in families `system.*` (19), `vllm:*` (81), `DCGM_FI_*` (65),
`http_*`/other (~15); 505 tagged log records **plus 440 untagged**
self-telemetry that must be excluded.

Known gaps inherited from the producer: `code.*` span attributes are **not
emitted** (so stage-03 drill-down keeps relying on `Read` over the repo, as it
already does); there is **no root run span** yet (window comes from trace
min/max).

## 6. Backward compatibility & source selection

| Flags | Source | Behaviour |
|---|---|---|
| `--telemetry-from <path>` | file | existing pre-cooked / agent-on-dir, unchanged |
| `--signoz` | SigNoz | auto-select: 0 runs → error; 1 → use it; >1 → use **latest** + **loud multi-line warning** (count, capped list of others, "pass `--run-id <id>` to pick another"). Non-blocking; same in CI and interactively. |
| `--signoz --run-id <id>` | SigNoz | analyze that specific run |
| *(none)* | — | synthetic stub, unchanged |

Parse-time validation:
- `--telemetry-from` is mutually exclusive with `--signoz` (hard error).
- **`--run-id` requires `--signoz`** — `--run-id` alone is a hard error.
  `--signoz` is the source selector; `--run-id` only refines it.

Alternative flag spelling considered: a single optional-valued `--run-id`
(`--run-id` alone = auto-latest, `--run-id X` = specific), dropping `--signoz`.
Rejected — a single flag doing double duty is less discoverable in `--help`,
and an explicit `--signoz` source selector reads better.

## 7. Verification plan

- **Unit:** row-merge from a captured API-response fixture; window resolution;
  CLI arg parsing; the metrics-join SQL string. `signoz-sql`'s HTTP call is
  mocked — no live dependency in unit tests.
- **Integration / parity (manual, gated on network):** run stage 01 with
  `--run-id 20260615T073440Z` and diff the produced `Signals` against the
  file-capture
  [`01_signals.json`](../../experiments/baseline-main__enh2/artifacts/01_signals.json).
  Expect the trace-derived anomalies (tail latency, decode-bound,
  output-len-to-cap, prompt-size tail) to reproduce, **plus** new GPU/DCGM
  signals.
- **Safety:** confirm the configured key is Viewer-scoped — a write/DDL query
  must be rejected by SigNoz, not by us.

## 8. Open items for implementation

- **Read-only key — resolved.** The existing `.env` `SIGNOZ_API_KEY` is already
  Viewer-scoped; reuse it (no separate `_RO` key needed). `.env` carries a
  warning that any replacement key must be Viewer-scoped. The prompt adds the
  soft `SELECT`-only instruction (§2.4, §4.4).
- Console-script name (`signoz-sql`) vs `python -m …` invocation for the agent.
- Whether the canonical `metric_samples` CTE should expose `unix_milli` (it
  does) so the agent can time-bucket metrics itself — vs adding a fixed
  time-aligned variant. Leaning: `unix_milli` is enough; the agent buckets in
  its own SQL. (Run scoping is by `run_id`, so no time window is needed for
  selection — this is only about intra-run time-series shaping.)
