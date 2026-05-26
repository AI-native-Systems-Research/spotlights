<p align="center">
  <img src="docs/logo.png" alt="Spotlights" width="160">
</p>

<h1 align="center">Spotlights</h1>

<p align="center"><b>Find the few places in a codebase worth optimizing — and see the evidence for why.</b></p>

Point Spotlights at a repo and an objective (`reduce TTFT`, `raise throughput under sustained load`) and it returns a ranked, browsable map of the few places worth touching — each with a concrete, evidence-backed proposal grounded in the code, the literature, and your objective.

Most code-research tools either scan broadly and return shallow hits, or dive deeply into a single file you already picked. Spotlights does the part in between: it decides *which* symbols across the whole repo are worth deep investigation for your objective, then spends real research effort on each one.

The bet behind the project: execution tooling — coding agents, evolutionary search, experiment harnesses — is abundant and improving fast. The harder, less-solved problem is knowing **where to point it**. Spotlights treats that as a discovery problem in its own right: build a map of the codebase, converge independent signal sources onto that map, and let the places where evidence piles up — the *spots that light up* — surface as candidates worth optimizing.

> **Status, honestly:** today Spotlights runs end-to-end on two signal sources — **code structure** and the **research literature**. Runtime telemetry, repository history, and paper-driven discovery are in active development. See [Signal sources & roadmap](#signal-sources--roadmap).

## How it works

A single structural map of the repo is the substrate. Signal sources attach to it, and a candidate is a region of that map where a signal indicates something worth investigating. The engine narrows in three tiers:

1. **Structural map** — static analysis extracts the project's modules and how they fit together.
2. **Candidates** — per module, pick the symbols most worth investigating for the stated objective.
3. **Proposals** — for each candidate, produce evidence-backed change proposals, with citations and rationale. Proposals aren't limited to local tweaks: when the literature supports it, a proposal can be a genuinely new approach — applying a technique from a recent paper, or building a new kernel — not just a refinement of what's already there.

The output is a tree of plain Markdown files you browse in any viewer (GitHub, VS Code preview, Obsidian).

What feeds tiers 2 and 3 is a *signal source*. The first one implemented is a deep-research engine that pulls findings from the web, arXiv, blogs, and documentation, and grounds proposals in that literature alongside the code itself. The architecture is built so that additional signal sources — telemetry, repo history, and others — plug into the same map and the same candidate/proposal pipeline.

Discovery runs in two directions. The primary, live direction starts from the system: a candidate surfaces, relevant findings are gathered, and a concrete change is proposed (signal → candidate → proposal). The inverse direction starts from an idea: given a promising paper or technique, Spotlights searches the map for where it could apply — down to whether it's worth building a new kernel for it (technique → location). The second direction shares the same map and pipeline and is an active area of exploration.

![Architecture](docs/architecture.png)

## Signal sources & roadmap

The module map is a shared coordinate system: every signal source projects onto it, and candidates are the nodes where signals converge. This is where the project is headed.

| Signal source | What it contributes | Status |
|---|---|---|
| **Code structure** | Static analysis builds the module tree that grounds every candidate in a specific code region. | **Live** |
| **Research literature** | A deep-research engine retrieves findings from the web, arXiv, blogs, and docs, and derives evidence-backed proposals for each candidate. | **Live** |
| **Runtime telemetry** | OpenTelemetry traces, logs, and profiles surface bottlenecks visible only under load, not in the source. | In development |
| **Repository history** | Issues, pull requests, and commit history capture known limitations, past reasoning, and undocumented benchmarks that never reach code comments. | In development |
| **Technique-driven discovery** | Start from a paper or technique and search the codebase for where it could apply — the inverse of starting from a bottleneck. | Exploring |

Contributions to any of these are welcome — see [Contributing](#contributing).

## Install

**Prerequisites:**

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- The `claude` CLI on PATH, with auth configured via its own login state or supported environment variables. Used by the modules extractor, candidate discovery, and the Claude executors for steps 4 and 5.
- The `codex` CLI on PATH, with auth configured via its own login state or supported environment variables. Used by `module_deep_research` (hard-coded today) and the Codex executors for steps 2 and 5.
- A target repo on disk (the quickstart below uses vLLM).

Both CLIs are required for the default end-to-end path; the engine will not run without one of them today. Install, authenticate, and verify each before launching the engine.

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

### Optional: route both CLIs through a LiteLLM proxy

If you can't (or don't want to) authenticate against Anthropic and OpenAI directly — for example, when running inside a corporate environment that exposes models via a LiteLLM proxy — you can point each CLI at the proxy instead of its native backend.

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

**`codex` CLI** — edit `~/.codex/config.toml` and define a `litellm` provider, then select it as the default:

```toml
model = "<your-model-id>"
model_provider = "litellm"
model_reasoning_effort = "xhigh"

[model_providers.litellm]
name = "LiteLLM"
base_url = "https://your-litellm-host.example.com/v1"
experimental_bearer_token = "<your-litellm-key>"
wire_api = "responses"
```

Swap the host and model IDs for whatever your LiteLLM deployment exposes. After editing either file, re-run `claude --version` / `codex --version` from a fresh shell to confirm the CLI still launches; the engine will then route all of its agent calls through the proxy.

### Verify both CLIs from a fresh shell

Open a new terminal (so any PATH changes from the installers are picked up) and confirm both binaries resolve and report a version:

```bash
which claude && claude --version
which codex  && codex  --version
```

If either command is not found, re-open your terminal so shell PATH updates from the installers take effect. The native `claude` installer drops its binary at `~/.local/bin/claude`; the `codex` binary location depends on the install method (e.g. `/opt/homebrew/bin/codex` for Homebrew, `~/.local/bin/codex` for the native installer) — check the installer's final output if `codex` still isn't on PATH.

### Install the engine

```bash
git clone https://github.com/Video-AI/spotlights.git
cd spotlights
uv sync --all-extras
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

Spotlights decides *what* to optimize and proposes *how* — it is not itself an execution engine. Its candidates and proposals are designed to hand off to whatever runs and validates changes: coding agents, evolutionary search, or experiment frameworks. Think of it as the front-end that points existing optimization machinery at the places most worth its effort.

## Contributing

The project is organized around signal sources, and the most useful contributions add or sharpen one:

- **New signal sources** — telemetry, repo history, and technique-driven discovery are the active frontiers (see the [roadmap](#signal-sources--roadmap)). The candidate/proposal pipeline and the module map are shared, so a new source mostly means projecting its evidence onto existing candidates.
- **Engine improvements** — better module extraction, candidate ranking, proposal quality, parallelism, or resumability.
- **Adapters** — support for target repos and objectives beyond the vLLM examples.

Open an issue to discuss a direction before a large change. Bug reports and example runs on new repos are also valuable.

## License

Apache-2.0 — see [LICENSE](LICENSE).