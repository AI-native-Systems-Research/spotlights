# Telemetry-driven discovery (preview)

> [!IMPORTANT]
> This path is a **preview** (MVP). It runs via a separate entry point — `signal-pipeline`, not `spotlights-engine`.

Given a captured workload's OpenTelemetry traces and the subject repo, `signal-pipeline` produces evidence-backed code changes with rationales and applied diffs, each anchored to an anomaly the capture actually exhibited.

A canonical run on a vLLM/LRU OTel capture takes **~10 min and ~$2** in API costs and yields a handful of candidates.

← Back to [README](../README.md)

## Usage

```bash
signal-pipeline \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts \
  --repo ../vllm \
  --telemetry-from data/20260525T202105Z_util0.4_mem16_lru
```

`--telemetry-from` accepts either a directory of raw OTel files or a pre-cooked `01_signals.json` — the agent figures out which.

## Flags

| Flag | Default | Purpose |
|---|---|---|
| `--repo` | `../vllm` | Subject repo being analyzed. |
| `--telemetry-from` | (none) | Directory of raw OTel files, or a pre-cooked `01_signals.json`. |
| `--artifacts-dir` | `./artifacts` | Per-run stage artifacts and state (`runs/<run-id>/`, resumable). |
| `--output-folder` | `./spotlights-out` | Where `signal_summary.{json,md}` and `spotlight_report.json` render. |
| `--backend-id` | `claude_code` | Execution backend id forwarded to stage 05. |
| `--from-stage` | `01` | Run stages from this id onward (pairs with `--to-stage`). |
| `--to-stage` | `04` | Stop after this stage. Stage 05 mutates the repo, so it's opt-in — pass `--to-stage 05`. |
| `--only-stage` | (none) | Run exactly one stage; implies `--no-resume`. Requires declared upstream stages on disk. |
| `--no-resume` | resume on | Re-run every stage in the selection from scratch. |
| `--no-projecttree-cache` | cache on | Force ProjectTree extraction to re-run instead of reading the cross-run cache (the fresh result still updates it). |
| `--max-candidates` | (none) | Soft cap on candidates — the prompt keeps the top-N by signal strength × significance (not a hard schema cap). |
| `--model` | (none) | Model id forwarded as `--model` to every `claude -p` call, except stages that pin their own model in code. E.g. `claude-sonnet-4-6`. |
| `--inject` | (none) | Substitute a stage's output: `NN=file.json` (single-artifact), `NN=dir/` (fan-out whole-dir), or `NN/<id>=file.json` (fan-out per-id). Repeatable; validated before any write. |
