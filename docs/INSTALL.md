# Installing Spotlights

This page covers installing the engine, the bundled Claude Code slash commands, the `doctor` preflight, and the full configuration reference.

← Back to [README](../README.md)

## Installing the engine

**Prerequisites:** Python ≥ 3.11, plus `git` and `curl`. [uv](https://docs.astral.sh/uv/) is installed for you if it is missing.

### Quick install (recommended)

Installs `spotlights-engine` (and the sibling CLIs) into an isolated venv and puts them on your PATH in `~/.local/bin` — no shell activation needed afterwards:

```bash
curl -fsSL https://raw.githubusercontent.com/AI-native-Systems-Research/spotlights/main/install.sh | sh
```

The script installs `uv` if missing, installs the engine, and fixes up your PATH. It is idempotent — re-run it to upgrade an existing install in place.

Pin a version with `SPOTLIGHTS_VERSION` (any git tag, branch, or commit):

```bash
curl -fsSL https://raw.githubusercontent.com/AI-native-Systems-Research/spotlights/main/install.sh | SPOTLIGHTS_VERSION=v0.1.0 sh
```

If `spotlights-engine` is not found afterwards, open a new terminal (or run `uv tool update-shell`) so `~/.local/bin` is on your PATH.

If you already have `uv` and would rather skip the script, it runs this underneath:

```bash
uv tool install --force "git+https://github.com/AI-native-Systems-Research/spotlights.git@main"
```

### From source (development)

Gives you an editable install, which the CLIs automatically prefer over a `uv tool` install:

```bash
git clone https://github.com/AI-native-Systems-Research/spotlights.git
cd spotlights
uv sync --all-extras
source .venv/bin/activate
```

## Install the Spotlights skill

The engine ships Claude Code slash commands (currently `/spotlights-objective-setting`, `/spotlights-sort-candidates`, `/spotlights-share-candidates`, and `/spotlights-apply-candidate`) as bundled markdown templates. They are not active until you install them into a Claude Code commands directory. By default `init` installs them once, system-wide, so they are available from every directory:

```bash
spotlights-engine init           # writes ~/.claude/commands/spotlights-*.md
```

Or scope them to a single checkout — useful when you want the commands pinned to one project's version, or committed alongside it:

```bash
spotlights-engine init --scope project   # writes ./.claude/commands/spotlights-*.md
```

Re-running `spotlights-engine init` only adds files that are missing; anything already on disk is left alone. To pick up new bundled versions after a package upgrade, use `--force`, which rewrites every file the bundle ships — so any edits you made to those files are discarded. Files Spotlights does not ship (your own notes inside a skill directory, for instance) are never touched either way. `init` records what it installed in `<scope-root>/.spotlights/manifest.json` (sha256 per file); that is a record of the install, not an input to the overwrite decision.

If you set `CLAUDE_CONFIG_DIR` to move Claude Code's config directory, a user-scope `init` follows it and writes to `$CLAUDE_CONFIG_DIR/commands/` instead. What matters is whether the variable is *set*, not whether it has a value: set to the empty string it names a `commands/` directory relative to your working directory, which is what Claude Code itself reads in that case. The value is used literally — no `~` expansion, so `CLAUDE_CONFIG_DIR=~/.claude` in a config file that does not expand tildes means a directory actually named `~`.

Symlinks are followed wherever they appear, so managing `~/.claude`, `commands/`, a single skill directory or file, or `.spotlights/` through a dotfiles repo (stow, chezmoi, plain `ln -s`) works — including on a fresh clone whose links do not point at anything yet, where `init` creates what they point at.

Open Claude Code (in any directory for a user-scope install, or in the project directory for `--scope project`) and the slash commands appear:

```
/spotlights-objective-setting
/spotlights-sort-candidates
/spotlights-share-candidates
/spotlights-apply-candidate
```

## Preflight with `doctor`

Before your first real run, verify the environment end-to-end:

```bash
spotlights-engine doctor
```

`doctor` runs each agent CLI (`claude`, `codex`) with a one-word prompt, using the
model your config resolves to rather than the CLI's own default — otherwise it can
pass on a model no real run would ask for. A single probe proves three things at once:

- **install** — the CLI resolves on PATH and executes;
- **auth** — it exits cleanly with parseable output (an unauthenticated CLI errors out);
- **pricing readiness** — the model the CLI reports has a matching row in the cost
  rate table, so a real run will not silently drop that model's cost.

It exits non-zero on any failure. Because it launches the CLIs, `doctor` spends a
tiny amount per probe — that is the only way to verify auth and the reported model.
If a probe reports a model with no rate row, add it to the table or point
`SPOTLIGHTS_RATES_FILE` at a table that prices it. See [docs/cost-and-manifest.md](cost-and-manifest.md).

`doctor` also prints a `models` check naming the effective model for each CLI and
where the value came from — the bundled `models.yaml`, a file named by
`$SPOTLIGHTS_MODELS_FILE`, or the CLI's own config. With `codex` left blank you
will see *two* Codex probes rather than one: step 2 pins its built-in `gpt-5.5`
while steps 3 and 5 inherit `~/.codex/config.toml`, so both models get checked and
a green result cannot hide an unprobed step. Setting `codex` collapses that back to
a single probe. See [Choosing models](agent-cli-setup.md#choosing-models).

## Engine Configuration

All flags are optional once `--repo` and the agent CLIs are available.

| Flag | Default | Purpose |
|---|---|---|
| `--repo` | `../vllm` | Target repo path. |
| `--repo-url` | inferred from `--repo` | Canonical target repo URL recorded in the public run manifest. |
| `--include` | (all modules) | Restrict to slash-form qualified names; a parent includes its full module subtree. |
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
| `--claude-model ID` | from `models.yaml` (ships blank → the CLI's own default) | Model id passed as `claude --model` at every step. Overrides `models.yaml` / `$SPOTLIGHTS_MODELS_FILE` for one run. Pass it empty (`--claude-model ""`) to force "inherit `~/.claude/settings.json`" even when the file pins something. A set value is part of the resume fingerprint, so resume a run with the model it was started with. See [Choosing models](agent-cli-setup.md#choosing-models). |
| `--codex-model ID` | from `models.yaml` (ships blank) | Model id passed to Codex at every step. Same precedence and empty-string rule as `--claude-model`. One wart: candidate discovery (step 2) keeps a built-in `gpt-5.5` default for resume compatibility, so a blank value leaves step 2 on `gpt-5.5` while steps 3 and 5 inherit `~/.codex/config.toml`; setting a value makes all three agree. |
