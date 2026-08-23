# Installing Spotlights

This page covers installing the engine, the bundled Claude Code slash commands, the `doctor` preflight, and the full configuration reference.

← Back to [README](../README.md)

## Installing the engine

```bash
git clone https://github.com/AI-native-Systems-Research/spotlights.git
cd spotlights
uv sync --all-extras
source .venv/bin/activate
```

## Install the Spotlights skill

The engine ships Claude Code slash commands (currently `/spotlights-objective-setting`, `/spotlights-sort-candidates`, `/spotlights-share-candidates`, and `/spotlights-fix-candidate`) as bundled markdown templates. They are not active until you install them into a Claude Code commands directory. Locally In the project:

```bash
spotlights-engine init           # writes .claude/commands/spotlights-*.md
```

Or install once, system-wide:

```bash
spotlights-engine init --scope user   # writes ~/.claude/commands/spotlights-*.md
```

`init` records what it installed in `<scope-root>/.spotlights/manifest.json` (sha256 per file). Re-running `spotlights-engine init` is a no-op for existing files. To pick up new bundled versions after a package upgrade, use `--force` — files the user has edited (hash differs from the manifest) are preserved.

Open Claude Code in the same directory and the slash commands appear:

```
/spotlights-objective-setting
/spotlights-sort-candidates
/spotlights-share-candidates
/spotlights-fix-candidate
```

## Preflight with `doctor`

Before your first real run, verify the environment end-to-end:

```bash
spotlights-engine doctor
```

`doctor` runs each agent CLI (`claude`, `codex`)
with a one-word prompt. A single probe proves three things at once:

- **install** — the CLI resolves on PATH and executes;
- **auth** — it exits cleanly with parseable output (an unauthenticated CLI errors out);
- **pricing readiness** — the model the CLI reports has a matching row in the cost
  rate table, so a real run will not silently drop that model's cost.

It exits non-zero on any failure. Because it launches the CLIs, `doctor` spends a
tiny amount per probe — that is the only way to verify auth and the reported model.
If a probe reports a model with no rate row, add it to the table or point
`SPOTLIGHTS_RATES_FILE` at a table that prices it. See [docs/cost-and-manifest.md](cost-and-manifest.md).

## Engine Configuration

All flags are optional once `--repo` and the agent CLIs are available.

| Flag | Default | Purpose |
|---|---|---|
| `--repo` | `../vllm` | Target repo path. |
| `--repo-url` | inferred from `--repo` | Canonical target repo URL recorded in the public run manifest. |
| `--include` | (all modules) | Restrict to slash-form leaf qualified names. |
| `--objective` | `"reduce hot-path latency on common workloads"` | Threaded into discovery + deep research. |
| `--hint` (repeatable) | `[]` | Workload hints; map to `SpotlightContext.workload_hints`. |
| `--output-folder` | `./spotlights-out` | Where `index.md` and module pages land. |
| `--artifacts-dir` | `./artifacts` | Checkpoints + raw transcripts (resume key). |
| `--max-parallel` | `1` | Modules processed concurrently. |
| `--max-parallel-pairs` | `5` | Within-step parallelism for step 4. |
| `--max-parallel-candidates` | `5` | Within-step parallelism for step 5. |
| `--max-findings-per-module` | `30` | Cap on findings produced by step 3 per module. |
| `--review-iterations N` | `3` | Number of candidate-discovery review iterations after the bootstrap pass (step 2). Iter 0 is Claude bootstrap; iters 1..N alternate Codex/Claude, so `N=3` runs 4 iterations total (Claude → Codex → Claude → Codex). `0` disables the review session (bootstrap pass only). |
| `--no-review` | off | Alias for `--review-iterations 0`. Mutually exclusive with `--review-iterations`. |
| `--enable-claude-search` | off | Also run the Claude runner in step 3 (default: Codex only). Enable when using the `claude` CLI directly against Anthropic (its `WebSearch` works); leave off behind a LiteLLM server, where `WebSearch` is currently unreliable. |
| `--no-deep-research` | off | Skip step 3 entirely (no research session, no findings); step 4 short-circuits to zero proposals, steps 1/2/5 unchanged. Cheap candidates-only mode. The step-3-only knobs (`--enable-claude-search`, `--no-candidate-hotspots`, `--max-findings-per-module`) become no-ops but still count in the resume fingerprint, so resume a run with the same flags it was started with. |
