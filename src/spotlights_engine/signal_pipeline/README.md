# `signal_pipeline`

Signal-based discovery pipeline (MVP). In: a captured workload's telemetry
+ a subject system's repository. Out: a list of evidence-backed code
changes with rationales and applied diffs.

This README covers **how to run it and iterate on it**. The architecture
("why these five stages") lives in
[`docs/signal-based/signal_discovery_flow.md`](../../../docs/signal-based/signal_discovery_flow.md);
the data contracts in
[`docs/signal-based/mvp_module_apis.md`](../../../docs/signal-based/mvp_module_apis.md).

---

## The five stages

| # | Stage | What it does |
|---|---|---|
| 01 | **Signal extraction** | Reads raw OTel telemetry from a captured run, produces `Signals` (`WorkloadProfile`, `TraceSummary[]`, `Anomaly[]`). Pure-agent path: `claude -p` with `Bash` + `Read`, no Python parser. |
| 02 | **ProjectTree extraction** | Walks the subject repo and produces a structural map (modules, main files with roles, dependencies, descriptions). |
| 03 | **Candidate generation** | Reasons over `Signals` + `ProjectTree` to find what's worth optimizing. **Findings only — no proposed fixes.** Drills into source via `Read`. |
| 04 | **Change generation** *(per candidate)* | Turns each `Candidate` into a `Change` spec (`change_type`, `mechanism`, `expected_effect`, `evaluation_metric`). Spec only — no edits. |
| 05 | **Execution backend** *(per change)* | Applies the `Change` to the subject repo via `claude -p` with edit permissions. Produces an `ExecutionResult` with file edits + rationale. |

Knowledge retrieval, validation, and archive are deferred per the design
doc and aren't implemented here.

---

## Quick start

```bash
# One-shot: signal extraction → projecttree → candidates → changes → results.
# `--telemetry-from` accepts either a pre-cooked `01_signals.json` or a
# directory of raw OTel files (the agent figures out which).
env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL \
  uv run signal-pipeline \
    --run-dir runs/my-first-run \
    --subject-root ../vllm \
    --telemetry-from data/20260525T202105Z_util0.4_mem16_lru
```

The `env -u` prefix is required when running from inside a Claude Code
session — see [Troubleshooting](#troubleshooting) below.

A full pipeline run with all five stages takes **~13 min and ~$2** in
API costs at the time of writing (see the improvements backlog for
levers). Stages 01 + 02 are independent and could be parallelized;
they currently run sequentially.

---

## Run-dir layout

Each invocation writes to a single `--run-dir`. Single-artifact stages
write one file; per-candidate fan-out stages write a directory plus a
manifest:

```
runs/<run-id>/
├── input.json                 # the SignalPipelineInput passed in
├── status.json                # per-stage state machine
├── 01_signals.json            # Signals — workload + traces + anomalies
├── 02_projecttree.json        # ProjectTree
├── 03_candidates.json         # list[Candidate]
├── 04_changes/
│   ├── _manifest.json         # upstream_candidates_hash, covered_ids
│   └── <candidate_id>.json    # one Change per candidate
├── 05_results/
│   ├── _manifest.json         # upstream_changes_hash, covered_ids
│   └── <candidate_id>.json    # one ExecutionResult per candidate
└── _logs/
    └── <NN>_<name>/           # raw stdout/stderr from claude -p,
                               #  prompts, parsed events
```

`runs/` is gitignored. The same `--run-dir` can be re-used: `--resume`
(default) skips stages whose canonical artifact (or fan-out manifest)
is already complete; `--no-resume` re-runs everything from the
requested start.

---

## Iterating on a run

### Run a single stage

```bash
# Re-run only candidate generation (stage 03), keep upstream as-is.
uv run signal-pipeline --run-dir runs/x --subject-root ../vllm --only-stage 03
```

`--only-stage NN` is `--from-stage NN --to-stage NN --no-resume`. Stage
selection respects declared upstream — `--only-stage 03` requires
stages 01 and 02 to be on disk and complete.

### Run a range

```bash
uv run signal-pipeline --run-dir runs/x --subject-root ../vllm \
  --from-stage 03 --to-stage 04
```

### Hand-edit a stage's output and continue

Single-artifact stage:

```bash
uv run signal-pipeline --run-dir runs/x --subject-root ../vllm \
  --inject 03=path/to/my_candidates.json \
  --from-stage 04
```

Fan-out stage (whole directory replace, all per-candidate files):

```bash
uv run signal-pipeline --run-dir runs/x --subject-root ../vllm \
  --inject 04=path/to/changes_dir/
```

Fan-out stage (single per-id replacement):

```bash
uv run signal-pipeline --run-dir runs/x --subject-root ../vllm \
  --inject 04/cand-0001=path/to/one_change.json
```

The runner validates all `--inject` flags before writing anything —
hard error with details on schema-invalid payloads, unknown ids,
shape mismatches (single ↔ fan-out), missing-files / orphan-files.
See [`runner.py`](runner.py) for the validation tables.

### Iterate on a prompt

Each LLM stage's prompt is a markdown file the runner reads at
invocation time:

| Stage | Prompt file |
|---|---|
| 01 signal extraction | [`prompts/signal_extraction.md`](prompts/signal_extraction.md) |
| 03 candidate generation | [`prompts/candidate_generation.md`](prompts/candidate_generation.md) |
| 04 change generation | [`prompts/change_generation.md`](prompts/change_generation.md) |
| 05 execution backend | [`prompts/execution_brief.md`](prompts/execution_brief.md) |

Edit, re-run with `--only-stage NN --no-resume`. No code change needed.

The literal `{name}` placeholders in each prompt are filled by the
stage code via `str.format` — preserve them and double-brace any
literal `{` you want in the output.

---

## Troubleshooting

### `claude exit=1: 401 Invalid bearer token`

Caused by `ANTHROPIC_AUTH_TOKEN` (set inside Claude Code sessions)
leaking into the spawned `claude -p` subprocess. The pipeline's own
[`claude_subprocess.py`](claude_subprocess.py) drops both
`ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL`, but main's
`modules_extractor` (used by stage 02) only drops the latter.

**Fix:** prefix invocations with `env -u ANTHROPIC_AUTH_TOKEN -u ANTHROPIC_BASE_URL`,
or run from a regular shell outside Claude Code.

### `signal_pipeline requires the origin/main layout — rebase the branch first`

The runner's `_check_layout()` precondition couldn't import
`spotlights_engine.modules_extractor` and friends. Means the branch
hasn't been rebased onto current `origin/main` (where these modules
live).

### `--from-stage NN requires stage M to be complete`

Stage `NN` declares stage `M` as upstream, and `M` isn't on disk
(or its `status.json` entry isn't `done`). Either run from earlier,
extend the selection, or `--inject` an upstream artifact.

### Stage 02 stalls with no progress (Windows)

The bundled `claude` CLI on Windows is `claude.CMD`, a cmd.exe shim that
buffers the child's stdout. Live readers see no events until the child
exits — indistinguishable from a hang. Both
[`agent.py`](../modules_extractor/agent.py) and
[`claude_subprocess.py`](claude_subprocess.py) bypass the shim by
resolving `claude.exe` directly under
`%APPDATA%\npm\node_modules\@anthropic-ai\claude-code\` (CLI 2.1.140+).

### Stage 03 produces zero candidates

Stage 01's anomalies were too thin (e.g. only `instrumentation_gap`
without workload anomalies) and the candidate-generation prompt didn't
find leverage. Inspect `runs/<id>/01_signals.json`'s `anomalies` list;
if it's empty or all-instrumentation, the model is right to abstain.
If it's not, iterate the candidate-generation prompt.

---

## What lives where

```
signal_pipeline/
├── __init__.py               # public exports: run_pipeline, StageId, …
├── runner.py                 # orchestrator: stage dispatch, resume,
│                             #   --inject validation, status mgmt
├── layout.py                 # RunDirLayout, atomic writes, artifact_hash
├── claude_subprocess.py      # `run_claude` wrapper (stages 03/04/05),
│                             #   env-cleaning to prevent auth leak
├── cli.py                    # `signal-pipeline` script entry
├── schemas.py                # Change, ExecutionResult, FileEdit,
│                             #   Signals (placeholder for the
│                             #   spotlight-observability locked schema)
├── stages/
│   ├── _types.py             # StageContext, StageSpec
│   ├── s01_signal_extraction.py
│   ├── s02_projecttree.py
│   ├── s03_candidate_generation.py
│   ├── s04_change_generation.py
│   └── s05_execution.py
└── prompts/                  # markdown templates, iteration surface
    ├── signal_extraction.md
    ├── candidate_generation.md
    ├── change_generation.md
    └── execution_brief.md
```

Tests live at [`tests/unit/signal_pipeline/`](../../../tests/unit/signal_pipeline/).

---

## See also

- [`docs/signal-based/signal_discovery_flow.md`](../../../docs/signal-based/signal_discovery_flow.md) — architecture (mermaid + demo stories).
- [`docs/signal-based/signal_discovery_overview.md`](../../../docs/signal-based/signal_discovery_overview.md) — 5-min human-friendly read.
- [`docs/signal-based/mvp_module_apis.md`](../../../docs/signal-based/mvp_module_apis.md) — per-module schemas + interaction walks.
- [`docs/signal-based/pipeline-improvements.md`](../../../docs/signal-based/pipeline-improvements.md) — deferred perf / cost backlog. When you implement an item, remove it from that file.
- [`docs/projecttree/`](../../../docs/projecttree/) — schema and HTTP API for the structural map stage 02 produces.
