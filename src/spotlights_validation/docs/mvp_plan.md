# MVP Plan — Validation (Bundle E)

> Full design: [design.md](design.md)

This document describes what changes for the MVP: which steps from the Phase 1 / Phase 2 flow are automated vs. run manually with saved output.

## What changes in the MVP

The full flow (Phase 1 → Phase 2) is preserved, but the first two Phase 1 steps are **run manually** and their output is **saved to disk** before validation begins. The automated pipeline starts from the saved artifacts and picks up from there.

```
MVP flow:

  [Manual, output saved]
  1. Discovery — run discover_test_harness + seed_workload_matrix against the target
     repo; save TestHarnessMap and ValidationWorkloadMatrix to disk.

  [Manual, output saved]
  2. Validation plan — run build_validation_plan with the saved TestHarnessMap,
     ValidationWorkloadMatrix, and archive context; save ValidationPlan to disk.

  [Automated — Phase 2]
  3. run_validation_plan     — execute the validation plan against the source tree.
  4. Assemble ValidationResult.
  5. Write record to archive (Bundle B).
```

The public API (`prepare` / `start_validation`) is still the contract Bundle F calls. In the MVP, `prepare` loads the saved artifacts from disk rather than running discovery and planning live.

## Manual steps

### 1. Discovery

Produces `TestHarnessMap` and `ValidationWorkloadMatrix` for [vllm repo](https://github.com/vllm-project/vllm).

```bash
# From spotlights-validation root
python -m spotlights_validation.cli discover \
    --source-tree https://github.com/vllm-project/vllm \
    --out-dir artifacts/
# Writes: artifacts/harness_map.json, artifacts/workload_matrix.json
```

Run once per target version. Re-run whenever the target's test or benchmark infrastructure changes materially.

### 2. Validation plan

Produces the full `ValidationPlan` from the saved discovery artifacts.

```bash
python -m spotlights_validation.cli plan \
    --harness-map artifacts/harness_map.json \
    --workload-matrix artifacts/workload_matrix.json \
    --archive-context artifacts/archive_context.json \  # optional, empty list if absent
    --out artifacts/validation_plan.json
```

The plan is not prioritied at this stage. `build_validation_plan` will adapt it to the specific `Change` at runtime at future phases after mvp.

## Automated path (Phase 2)

Once the saved artifacts are in place, Bundle F calls the standard public API:

```python
from spotlights_validation import prepare, start_validation

# prepare loads from disk; no live discovery
prep = await prepare(source_tree=target_repo_path, artifacts_dir=Path("artifacts/"))

run = await start_validation(
    change=change,
    execution_result=execution_result,
    preparation=prep,
)
result = await run.result()
```

## Saved artifact locations

| Artifact | Path | Produced by |
|---|---|---|
| `TestHarnessMap` | `artifacts/harness_map.json` | manual discovery |
| `ValidationWorkloadMatrix` | `artifacts/workload_matrix.json` | manual discovery |
| `ValidationPlan` | `artifacts/validation_plan.json` | manual planning |
| Archive context | `artifacts/archive_context.json` | manual (from Bundle B query) |

## Out of scope for MVP

- Live (automated) re-discovery on every run — the saved artifacts are the source of truth.
- Automatic archive context refresh — the context JSON is populated manually before the plan is built.
- The archive write path (Bundle B) — `run_validation_plan` assembles `ValidationResult` but the write is stubbed until Bundle B's write interface is available.
