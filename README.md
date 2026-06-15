<p align="center">
  <img src="docs/logo.png" alt="Spotlights" width="160">
</p>

<h1 align="center">Spotlights</h1>

<p align="center"><b>Find the few places in a codebase worth optimizing — and see the evidence for why.</b></p>

Point Spotlights at a repo and a goal (e.g., `reduce p99 latency`, `minimize memory allocations`) and it returns a ranked, browsable map of the few places worth touching — each with a concrete, evidence-backed proposal grounded in the code, the literature, and your goal.

Most code-research tools either scan broadly and return shallow hits, or dive deeply into a single file you already picked. **Spotlights acts as the targeting system**: it decides which functions and code regions across the whole repo are worth deep investigation for your goal, then spends real research effort on each one.

The core philosophy behind the project: execution tooling — coding agents, evolutionary search, experiment harnesses — is abundant and improving fast. The harder, less-solved problem is knowing **where to point it**. Spotlights treats that as a first-class discovery problem: build a map of the codebase, converge independent signal sources onto that map, and let the places where evidence piles up — the *spots that light up* — surface as candidates worth optimizing.

> **Current Status:** today Spotlights runs on two signal sources — **code structure** and the **deep research literature** — plus a **preview** of a third path driven by **runtime telemetry** (see [Telemetry-driven discovery (preview)](#telemetry-driven-discovery-preview)). A **validation module** that confirms proposed changes preserve correctness and improve the targeted metric is in active development (see [Validation](#validation)). Repository history and paper-driven discovery are also in active development. See [Signal sources & roadmap](#signal-sources--roadmap).

## How it works

A single structural map of the repo is the substrate. Signal sources attach to it, and a candidate is a region of that map where a signal indicates something worth investigating. The engine narrows in three tiers:

1. **Structural map** — static analysis extracts the project's modules and how they fit together.
2. **Candidates** — per module, pick the functions and code regions most worth investigating for the stated goal.
3. **Proposals** — for each candidate, produce evidence-backed change proposals, with citations and rationale. Proposals aren't limited to local tweaks: when the literature supports it, a proposal can be a genuinely new approach — applying a technique from a recent paper, or building a new kernel — not just a refinement of what's already there.

When proposals are executed by downstream backends, the **validation module** gates the result: it confirms correctness, measures performance delta, and checks intent alignment before outcomes enter the experimental archive. See [Validation](#validation).

The output is a tree of Markdown files.

*Signal sources* drive the second and third tiers. The first one implemented is a deep-research engine that pulls findings from the web, arXiv, blogs, and documentation, and grounds proposals in that literature alongside the code itself. The architecture is built so that additional signal sources — telemetry, repo history, and others — plug into the same map and the same candidate/proposal pipeline.

Discovery runs in two directions. The primary, live direction starts from the system: a candidate surfaces, relevant findings are gathered, and a concrete change is proposed (signal → candidate → proposal). The inverse direction starts from an idea: given a promising paper or technique, Spotlights searches the map for where it could apply — down to whether it's worth building a new sub-system for it (technique → location). The second direction shares the same map and pipeline and is an active area of exploration.

## Signal sources & roadmap

The module map is a shared coordinate system: every signal source projects onto it, and candidates are the nodes where signals converge. This is where the project is headed.

| Signal source | What it contributes | Status |
|---|---|---|
| **Code structure** | Static analysis builds the module tree that grounds every candidate in a specific code region. | **Live** |
| **Research literature** | A deep-research engine retrieves findings from the web, arXiv, blogs, and docs, and derives evidence-backed proposals for each candidate. | **Live** |
| **Runtime telemetry** | OpenTelemetry traces, logs, and profiles surface bottlenecks visible only under load, not in the source. Wired through the [`signal-pipeline`](#telemetry-driven-discovery-preview) flow. | **Live (preview)** |
| **Repository history** | Issues, pull requests, and commit history capture known limitations, past reasoning, and undocumented benchmarks that never reach code comments. | In development |
| **Technique-driven discovery** | Start from a paper or technique and search the codebase for where it could apply — the inverse of starting from a bottleneck. | Exploring |
| *…and more* | *The list isn't closed — if you have a signal source in mind, propose one via [Contributing](#contributing).* | *Open* |

Contributions to any of these are welcome — see [Contributing](#contributing).

## Install

**Prerequisites:**

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- The `claude` CLI on PATH, with auth configured via its own login state or supported environment variables. Used by the modules extractor, candidate discovery, and the Claude executors for steps 4 and 5.
- The `codex` CLI on PATH, with auth configured via its own login state or supported environment variables. Used by `module_deep_research` and the Codex executors for steps 2 and 5.
- The `gemini` CLI on PATH, with API-key or gateway auth configured. Used by `module_deep_research` alongside Codex and Claude when available.
- A target repo on disk (the quickstart below uses vLLM).

The module deep-research step fans out to Codex, Claude, and Gemini by default,
then treats individual runner failures as recoverable so one flaky provider does
not fail the whole step. Install, authenticate, and verify all three CLIs for
best coverage.

### Install the `claude` CLI

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

### Install the `codex` CLI

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

### Install the `gemini` CLI

Install Gemini CLI with npm and verify it from a fresh shell:

```bash
npm install -g @google/gemini-cli
which gemini && gemini --version
```

For direct Google API-key auth:

```bash
export GEMINI_API_KEY="<your-gemini-api-key>"
gemini --prompt "Reply with exactly: OK" --output-format json
```

### Optional: route the CLIs through a LiteLLM proxy

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

**`gemini` CLI** — point Gemini at the proxy root, not the OpenAI-compatible `/v1` path. Unlike `claude`, the Gemini CLI's `settings.json` has no `env` block, so authentication and proxy variables go in `~/.gemini/.env` (auto-loaded by every gemini invocation). `~/.gemini/settings.json` then handles auth pinning and — critically when routing through a LiteLLM proxy — the per-tier model overrides so flash-tier subcalls (web search, classifier, summarization) don't try to hit unprefixed model IDs your team isn't allowed.

Folder trust note: `~/.gemini/.env` is only loaded for trusted workspaces. The simplest fix for headless/subprocess runs is exporting `GEMINI_CLI_TRUST_WORKSPACE=true` in your shell profile (the only gemini-related export still needed). If you'd rather keep folder trust on, accept the trust prompt once per workspace in interactive `gemini`.

```bash
mkdir -p ~/.gemini

# Persist auth + proxy variables — auto-loaded by every gemini invocation.
export GEMINI_API_KEY=<your-litellm-virtual-key>
export GOOGLE_GEMINI_BASE_URL=https://your-litellm-host.example.com
export GEMINI_API_KEY_AUTH_MECHANISM=bearer
export GEMINI_CLI_TRUST_WORKSPACE=true

# Pin auth, set the pro-tier model, and remap every flash-tier alias to a
# model id your LiteLLM team is allowed to access. Without the customAliases
# block, web search / classifier / summarization calls send the bare
# `gemini-3-flash-preview` id and the proxy 401s.
cat > ~/.gemini/settings.json <<'JSON'
{
  "model": { "name": "gcp/gemini-3.1-pro-preview" },
  "modelConfigs": {
    "customAliases": {
      "gemini-3-flash-base":   { "extends": "base", "modelConfig": { "model": "gcp/gemini-3-flash-preview" } },
      "gemini-3.5-flash-base": { "extends": "base", "modelConfig": { "model": "gcp/gemini-3-flash-preview" } },
      "gemini-2.5-flash-base": { "extends": "base", "modelConfig": { "model": "gcp/gemini-3-flash-preview" } },
      "classifier":            { "extends": "base", "modelConfig": { "model": "gcp/gemini-3-flash-preview" } }
    }
  },
  "advanced": { "ignoreLocalEnv": true },
  "security": { "auth": { "selectedType": "gemini-api-key" } }
}
JSON


gemini --prompt "Reply with exactly: OK" --output-format json
```

Swap `gcp/gemini-3.1-pro-preview` and `gcp/gemini-3-flash-preview` for whatever pro/flash IDs your gateway exposes. `advanced.ignoreLocalEnv: true` prevents a project-root `.env` from shadowing `~/.gemini/.env` (gemini auto-loads the first `.env` it finds walking up from cwd; without this flag, any project's own `.env` wins).

For IBM LiteLLM-backed live research from Python, use
`GeminiExecOptions.ibm_litellm(...)`. Plain `GeminiExecOptions()` keeps native
Gemini CLI auth/model/proxy behavior.

Swap the host and model IDs for your LiteLLM deployment. After editing config or
environment, re-run `claude --version` / `codex --version` / `gemini --version`
from a fresh shell.

### Verify all CLIs from a fresh shell

Open a new terminal (so any PATH changes from the installers are picked up) and confirm all binaries resolve and report a version:

```bash
which claude && claude --version
which codex  && codex  --version
which gemini && gemini --version
```

If any command is not found, re-open your terminal so shell PATH updates from the installers are picked up. The native `claude` installer drops its binary at `~/.local/bin/claude`; the `codex` and `gemini` binary locations depend on the install method (for example, Homebrew vs npm vs native installer) — check the installer's final output if a binary still is not on PATH.

### Install the engine

```bash
git clone https://github.com/AI-native-Systems-Research/spotlights.git
cd spotlights
uv sync --all-extras
source .venv/bin/activate
```

### Install the Spotlights skill

The engine ships Claude Code slash commands (currently `/spotlights-objective-setting`) as bundled markdown templates. They are not active until you install them into a Claude Code commands directory — same model as [spec-kit](https://github.com/github/spec-kit). From the project you want to optimize:

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
```

## Quickstart on a vLLM subset

Clone vLLM next to this repo, then:

```bash
spotlights-engine \
  --repo ../vllm \
  --include v1.kv_offload \
  --objective "reduce the median TTFT and median TPOT (Time Per Output Token)" \
  --hint "Multi-turn agentic workload" \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts
```

On disk:

```
spotlights-out/
  index.md                          # repo-level summary, one row per module
  result.json                       # full structured run output
  modules/
    v1.kv_offload.md                # module page: candidates table
    v1.kv_offload/
      <symbol-slug>__<candidate-id>.md   # full per-candidate write-up + proposals
artifacts/
  spotlights_manager/               # checkpoints, raw transcripts (resumable)
```

`--include` accepts one or more dot-form leaf qualified names. Repeat the flag or pass several values after a single flag:

```bash
spotlights-engine --include v1.kv_offload v1.attention.paged_kv ...
spotlights-engine --include v1.kv_offload --include v1.attention.paged_kv ...
```

### Example output

A sample run on this subset is checked in under `examples/vllm_subset/`: browse the rendered `index.md` and per-module pages under `modules/`, or inspect the raw `result.json`.

## Telemetry-driven discovery (preview)

A second entry point. Given a captured workload's OpenTelemetry traces and the subject repo, the `signal-pipeline` CLI runs five stages — signal extraction, ProjectTree extraction, candidate generation, change generation, execution — to produce evidence-backed code changes with rationales and applied diffs. A canonical run on a vLLM/LRU OTel capture takes ~10 min and ~$2 in API costs and yields a handful of candidates anchored to the captured anomalies.

```bash
signal-pipeline \
    --output-folder ./spotlights-out \
    --artifacts-dir ./artifacts \
    --repo ../vllm \
    --telemetry-from <path-to-otel-capture-or-signals.json>
```

`--telemetry-from` accepts either a directory of raw OTel files or a pre-cooked `01_signals.json`. For stage-by-stage details, prompt iteration, the artifacts-dir layout, the `--inject` workflow for hand-edited intermediates, model selection, and troubleshooting, see [`src/spotlights_engine/signal_pipeline/README.md`](src/spotlights_engine/signal_pipeline/README.md). Architecture and contracts live in [`docs/signal-based/`](docs/signal-based/).

**Status:** MVP. Knowledge retrieval and archive are deferred per the design doc.

## Validation

The validation module (Component E in the [system design](docs/architecture/spotlighs_design.md)) is the gate between a proposed change and the experimental archive. It confirms that changes preserve correctness, improve the targeted metric, and don't regress on others. No change enters the archive without a measured verdict.

Validation runs in three phases:

1. **Discovery** — scans the target repo's test and benchmark infrastructure (CI configs, test directories, benchmark scripts, workload configs, GitHub issues/PRs) to build a `TestHarnessMap` and seed a `ValidationWorkloadMatrix`. This is LLM-driven, not hardcoded, so it generalizes across target systems and languages.
2. **Planning** — prioritizes discovered validation entries toward the candidate's affected components, ranked by historical failure signal and halt conditions. Optionally augmented with change-specific entries when the `Change` object is available.
3. **Execution** — runs the target's own test suites and benchmarks (baseline vs. post-change), checks intent alignment against `Change.expected_effect`, and produces a structured pass/fail/conditional verdict with measurements.

Design and contracts: [`docs/architecture/spotlights_validation_design.md`](docs/architecture/spotlights_validation_design.md). Implementation plan: [`docs/validation/implementation_plan.md`](docs/validation/implementation_plan.md).

**Status:** The validation executor is available. Discovery and planning are in development.

## Configuration

All flags are optional once `--repo` and the agent CLIs are available.

| Flag | Default | Purpose |
|---|---|---|
| `--repo` | `../vllm` | Target repo path. |
| `--include` | (all modules) | Restrict to dot-form leaf qualified names. |
| `--objective` | `"reduce hot-path latency on common workloads"` | Threaded into discovery + deep research. |
| `--hint` (repeatable) | `[]` | Workload hints; map to `SpotlightContext.workload_hints`. |
| `--output-folder` | `./spotlights-out` | Where `index.md` and module pages land. |
| `--artifacts-dir` | `./artifacts` | Checkpoints + raw transcripts (resume key). |
| `--max-parallel` | `1` | Modules processed concurrently. |
| `--max-parallel-pairs` | `5` | Within-step parallelism for step 4. |
| `--max-parallel-candidates` | `5` | Within-step parallelism for step 5. |
| `--max-findings-per-module` | `30` | Cap on findings produced by step 3 per module. |

Agent authentication is handled by the underlying `claude` and `codex` CLIs; no engine config file is required for the happy path.

## Where Spotlights fits

Spotlights decides *what* to optimize and proposes *how*. Its candidates and proposals are designed to hand off to execution backends — coding agents, evolutionary search, or experiment frameworks — and the [validation module](#validation) gates the results before they enter the archive. Think of it as the scouting engine that points existing optimization machinery at the places most worth its effort, then confirms the outcome.

## Contributing

The project is organized around signal sources, and the most useful contributions add or sharpen one:

- **New signal sources** — telemetry, repo history, and technique-driven discovery are the active frontiers (see the [roadmap](#signal-sources--roadmap)). The candidate/proposal pipeline and the module map are shared, so a new source mostly means projecting its evidence onto existing candidates.
- **Engine improvements** — better module extraction, candidate ranking, proposal quality, parallelism, or resumability.
- **Adapters** — support for target repos and goals beyond the vLLM examples.

Open an issue to discuss a direction before a large change. Bug reports and example runs on new repos are also valuable.

## License

Apache-2.0 — see [LICENSE](LICENSE).
