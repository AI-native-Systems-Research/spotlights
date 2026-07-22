# Troubleshooting

## Re-running the full cycle

By default a run **resumes**. Run identity is a deterministic fingerprint of the inputs (repo + commit, `--include` scope, objective, hints, step config), and the on-disk checkpoint tree under `<artifacts-dir>/spotlights_manager/` is the single source of truth. Re-invoking the engine with the same inputs and the same `--artifacts-dir` picks up where the last run left off — only the steps that still need work re-run.

You'll usually want one of these instead:

**Resume an interrupted run (default).** Just re-run the exact same command. Steps that already completed are skipped; incomplete steps 2–5 continue per module.

```bash
spotlights-engine \
  --repo ../vllm \
  --include vllm/v1/kv_offload \
  --objective "reduce the median TTFT and median TPOT (Time Per Output Token)" \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts
```

**Force a completely fresh run (discard prior progress).** Delete the manager run dir, then re-run. This clears all checkpoints and transcripts so every step starts cold:

```bash
rm -rf ./artifacts/spotlights_manager
# then re-run the engine command above
```

Prefer `--no-resume` if you want the engine to *refuse* to reuse an existing run dir rather than silently resume — it errors out if `<artifacts-dir>/spotlights_manager/` already exists, forcing you to remove it (or point at a fresh `--artifacts-dir`) on purpose:

```bash
spotlights-engine ... --no-resume        # fails fast if a prior run dir exists
```

## Saving each run under its own folders

To keep runs side by side instead of overwriting, give each run its **own** `--output-folder` **and** `--artifacts-dir`. Both matter: `--output-folder` holds the rendered `index.md`/`result.json`, and `--artifacts-dir` holds the resume-key checkpoints. Point them at the same fresh pair and the run is fully self-contained and independent of any other:

```bash
spotlights-engine \
  --repo ../vllm \
  --include vllm/v1/kv_offload \
  --objective "reduce the median TTFT and median TPOT (Time Per Output Token)" \
  --output-folder ./runs/2026-07-22-kv-offload/out \
  --artifacts-dir ./runs/2026-07-22-kv-offload/artifacts
```

Because a fresh `--artifacts-dir` has no prior checkpoints, this always runs cold — no need to clean anything. Reusing an old `--artifacts-dir` with *changed* inputs (a different objective, scope, or target commit) is rejected with a resume-mismatch error rather than silently mixing results; use a new `--artifacts-dir` for the new inputs.

## `artifacts_dir resolves inside repo_path`

The engine refuses an `--artifacts-dir` that lives inside `--repo`, so checkpoints never contaminate the target repo (Codex runs with `-C <repo_path>`). Point `--artifacts-dir` somewhere outside the target repo.
