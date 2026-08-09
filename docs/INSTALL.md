# Installing Spotlights

This page is the full setup guide: the two external agent CLIs Spotlights drives, an optional LiteLLM proxy, the engine itself, the `doctor` preflight, and the bundled Claude Code slash commands. If you just want to see what Spotlights does first, start with the [README](../README.md).

← Back to [README](../README.md)

## Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- The `claude` CLI on PATH, authenticated. Used by the modules extractor, candidate discovery, and the Claude executors for steps 4 and 5.
- The `codex` CLI on PATH, authenticated. Used by `module_deep_research` and the Codex executors for steps 2 and 5.
- A target repo on disk (the quickstart uses vLLM).

> [!NOTE]
> The module deep-research step runs Codex only by default. Pass `--enable-claude-search` to also fan out to the Claude runner; individual runner failures are treated as recoverable, so one flaky provider does not fail the whole step. Enable this when running the `claude` CLI directly against Anthropic — its `WebSearch` tool works well there. Leave it off when routing through a LiteLLM server, where `WebSearch` currently does not work reliably.

## Install the `claude` CLI

Pick one install method. See the [official setup docs](https://docs.claude.com/en/docs/claude-code/setup) for the full matrix (Windows, WSL, Linux package managers, npm, version pinning).

```bash
# macOS / Linux / WSL — native installer (auto-updates)
curl -fsSL https://claude.ai/install.sh | bash

# macOS — Homebrew
brew install --cask claude-code

# Any platform with Node.js 18+ — npm (do NOT use sudo)
npm install -g @anthropic-ai/claude-code
```

Windows PowerShell: `irm https://claude.ai/install.ps1 | iex`.

A Pro, Max, Team, Enterprise, or Console plan is required (the free Claude.ai plan does not include Claude Code). After install, open a new terminal and authenticate by running `claude` once and following the browser prompt:

```bash
claude          # first run: log in via browser
claude --version
claude doctor   # deeper environment check
```

## Install the `codex` CLI

Pick one install method. See the [Codex CLI repo](https://github.com/openai/codex) for the full matrix (Windows, manual binary downloads, API-key auth).

```bash
# macOS / Linux — native installer
curl -fsSL https://chatgpt.com/codex/install.sh | sh

# macOS — Homebrew
brew install --cask codex

# Any platform with Node.js — npm
npm install -g @openai/codex
```

Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"`.

A ChatGPT Plus, Pro, Business, Edu, or Enterprise plan is the easiest auth path; an OpenAI API key also works with extra config. After install, open a new terminal and authenticate by running `codex` once and choosing "Sign in with ChatGPT":

```bash
codex           # first run: pick "Sign in with ChatGPT"
codex --version
```

## Optional: route the CLIs through a LiteLLM proxy

Use this when Anthropic, OpenAI, or Google models are exposed through an OpenAI-compatible LiteLLM gateway.

**`claude` CLI** — edit `~/.claude/settings.json` and set `ANTHROPIC_BASE_URL` plus the per-tier model overrides to models served by your proxy:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://your-litellm-host.example.com",
    "ANTHROPIC_AUTH_TOKEN": "<your-litellm-key>",
    "ANTHROPIC_MODEL": "<opus-model-id>",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "<opus-model-id>",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "<sonnet-model-id>",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "<haiku-model-id>"
  },
  "model": "opus"
}
```

**`codex` CLI** — edit `~/.codex/config.toml` and define a `litellm` provider, then select it as the default or as a named profile:

```toml
model = "<your-openai-compatible-model-id>"
model_provider = "litellm"
model_reasoning_effort = "xhigh"

[model_providers.litellm]
name = "LiteLLM"
base_url = "https://your-litellm-host.example.com/v1"
env_key = "LITELLM_API_KEY"
wire_api = "responses"
```

Swap the host and model IDs for your LiteLLM deployment. After editing config or
environment, re-run `claude --version` / `codex --version` from a fresh shell.

## Environment variables reference

Both of these are **optional** — a default run needs neither. They only apply when
overriding the cost rate table. Spotlights does not load a `.env` file; set them in
your shell environment. (Proxy/model configuration for the `claude` and `codex` CLIs
lives in their own config files — see the LiteLLM proxy section above.)

| Variable | Set in | Purpose |
| --- | --- | --- |
| `SPOTLIGHTS_RATES_FILE` | shell environment | Override the bundled cost rate table with your contracted LiteLLM rates. |
| `SPOTLIGHTS_EXTERNAL_RATES_FILE` | shell environment | Override the bundled external-model rate table. |

## Verify all CLIs from a fresh shell

Open a new terminal (so any PATH changes from the installers are picked up) and confirm all binaries resolve and report a version:

```bash
which claude   && claude   --version
which codex    && codex    --version
```

If any command is not found, re-open your terminal so shell PATH updates from the installers are picked up. The native `claude` installer drops its binary at `~/.local/bin/claude`; the `codex` binary location depends on the install method (for example, Homebrew vs npm vs native installer) — check the installer's final output if a binary still is not on PATH.

## Install the engine

```bash
git clone https://github.com/AI-native-Systems-Research/spotlights.git
cd spotlights
uv sync --all-extras
source .venv/bin/activate
```

## Preflight with `doctor`

Before your first real run, verify the environment end-to-end:

```bash
spotlights-engine doctor
```

Unlike a PATH-only check, `doctor` actually runs each agent CLI (`claude`, `codex`)
with a one-word prompt. A single probe proves three things at once:

- **install** — the CLI resolves on PATH and executes;
- **auth** — it exits cleanly with parseable output (an unauthenticated CLI errors out);
- **pricing readiness** — the model the CLI reports has a matching row in the cost
  rate table, so a real run will not silently drop that model's cost.

It exits non-zero on any failure. Because it launches the CLIs, `doctor` spends a
tiny amount per probe — that is the only way to verify auth and the reported model.
If a probe reports a model with no rate row, add it to the table or point
`SPOTLIGHTS_RATES_FILE` at a table that prices it.

## Install the Spotlights skill

The engine ships Claude Code slash commands (currently `/spotlights-objective-setting` and `/spotlights-sort-candidates`) as bundled markdown templates. They are not active until you install them into a Claude Code commands directory — same model as [spec-kit](https://github.com/github/spec-kit). From the project you want to optimize:

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
```

## Configuration

All flags are optional once `--repo` and the agent CLIs are available.

| Flag | Default | Purpose |
|---|---|---|
| `--repo` | `../vllm` | Target repo path. |
| `--include` | (all modules) | Restrict to slash-form leaf qualified names. |
| `--objective` | `"reduce hot-path latency on common workloads"` | Threaded into discovery + deep research. |
| `--hint` (repeatable) | `[]` | Workload hints; map to `SpotlightContext.workload_hints`. |
| `--output-folder` | `./spotlights-out` | Where `index.md` and module pages land. |
| `--artifacts-dir` | `./artifacts` | Checkpoints + raw transcripts (resume key). |
| `--max-parallel` | `1` | Modules processed concurrently. |
| `--max-parallel-pairs` | `5` | Within-step parallelism for step 4. |
| `--max-parallel-candidates` | `5` | Within-step parallelism for step 5. |
| `--max-findings-per-module` | `30` | Cap on findings produced by step 3 per module. |
| `--enable-claude-search` | off | Also run the Claude runner in step 3 (default: Codex only). Enable when using the `claude` CLI directly against Anthropic (its `WebSearch` works); leave off behind a LiteLLM server, where `WebSearch` is currently unreliable. |
