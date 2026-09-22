<p align="center">
  <img src="docs/logo.png" alt="Spotlights" width="160">
</p>

<h1 align="center">Spotlights</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.20446"><img src="https://img.shields.io/badge/arXiv-2609.20446-b31b1b.svg" alt="arXiv:2609.20446"></a>
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/python-%E2%89%A5%203.11-blue.svg" alt="Python ≥ 3.11">
</p>

<p align="center"><b>Find the few places in a codebase worth optimizing — and see the evidence for why.</b></p>

Point Spotlights at a repo and a goal (e.g., `reduce p99 latency`, `minimize memory allocations`) and it returns a ranked, browsable map of the few places worth touching — each with a concrete, evidence-backed proposal grounded in the code, the literature, and your goal.

Most code-research tools either scan broadly and return shallow hits, or dive deeply into a single file you already picked. **Spotlights acts as the targeting system**: it decides which functions and code regions across the whole repo are worth deep investigation for your goal, then spends real research effort on each one.

The core philosophy behind the project: execution tooling — coding agents, evolutionary search, experiment harnesses — is abundant and improving fast. The harder, less-solved problem is knowing **where to point it**. Spotlights treats that as a first-class discovery problem: build a map of the codebase, converge independent signal sources onto that map, and let the places where evidence piles up — the *spots that light up* — surface as candidates worth optimizing.

<p align="center">
  <a href="https://arxiv.org/abs/2609.20446">Paper</a> ·
  <a href="docs/INSTALL.md">Install</a> ·
  <a href="#quickstart">Quickstart</a> ·
  <a href="docs/cost-and-manifest.md">Cost</a> ·
  <a href="docs/prep-evolve.md">prep-evolve</a> ·
  <a href="docs/one-shot-apply.md">one-shot apply</a> ·
  <a href="docs/telemetry-preview.md">Telemetry (preview)</a> ·
  <a href="docs/spotlight-report.md">SpotlightReport</a> ·
  <a href="#signal-sources--roadmap">Roadmap</a>
</p>

> **Current Status:** today Spotlights runs on two signal sources — **code structure** and the **deep research literature** — plus a **preview** of a third path driven by **runtime telemetry** (see [Telemetry-driven discovery (preview)](docs/telemetry-preview.md)). Repository history and paper-driven discovery are also in active development. See [Signal sources & roadmap](#signal-sources--roadmap).

## How it works

A single structural map of the repo is the substrate. Signal sources attach to it, and a candidate is a region of that map where a signal indicates something worth investigating. The engine narrows in three tiers:

1. **Structural map** — static analysis extracts the project's modules and how they fit together.
2. **Candidates** — per module, pick the functions and code regions most worth investigating for the stated goal.
3. **Proposals** — for each candidate, produce evidence-backed change proposals, with citations and rationale. Proposals aren't limited to local tweaks: when the literature supports it, a proposal can be a genuinely new approach — applying a technique from a recent paper, or building a new kernel — not just a refinement of what's already there.

The output is a tree of Markdown files.

*Signal sources* drive the second and third tiers. The first one implemented is a deep-research engine that pulls findings from the web, arXiv, blogs, and documentation, and grounds proposals in that literature alongside the code itself. The architecture is built so that additional signal sources — telemetry, repo history, and others — plug into the same map and the same candidate/proposal pipeline.

Discovery runs in two directions. The primary, live direction starts from the system: a candidate surfaces, relevant findings are gathered, and a concrete change is proposed (signal → candidate → proposal). The inverse direction starts from an idea: given a promising paper or technique, Spotlights searches the map for where it could apply — down to whether it's worth building a new sub-system for it (technique → location). The second direction shares the same map and pipeline and is an active area of exploration.

## What you get

Spotlights writes a browsable tree of Markdown. Each candidate is a full write-up — the current approach, why it matters for your goal, and concrete proposals grounded in the code and the literature. A real candidate from the checked-in example run (`examples/vllm_subset/`):

> **[`LRUCachePolicy.evict`](examples/vllm_subset/modules/vllm_v1_kv_offload/LRUCachePolicy.evict__cand-vllm_v1_kv_offload-0003.md)** — `vllm/v1/kv_offload/cpu/policies/lru.py` (lines 56–78) · estimated impact: **medium**
>
> **Description.** LRU batch eviction that scans evictable blocks from the oldest end, skipping protected keys until n victims are collected.
>
> **Why it matters.** Agentic multi-turn workloads reuse prompt/prefix blocks across turns, and pure recency evicts them too aggressively. Better victim scoring raises primary-tier hit rate and cuts promotion stalls that affect median TTFT.
>
> **Proposal — Add an S3-FIFO cache policy as an alternative to pure-recency LRU** to protect hot multi-turn prefixes from one-hit block churn.
>
> Grounded in *FIFO Queues are All You Need for Cache Eviction* (<https://s3fifo.com/blog/2023/08/01/fifo-queues-are-all-you-need-for-cache-eviction/>): a small probationary FIFO demotes one-hit blocks before they pollute the main region, while a ghost FIFO detects returning prefixes and promotes them on the next store — preserving `CachePolicy`'s API and atomic evict contract.

On disk:

```
spotlights-out/
  index.md                          # repo-level summary, one row per module
  result.json                       # full structured run output
  run_manifest.json                 # copy of the public run manifest (see artifacts/ below)
  experiment.html                   # one self-contained page, written by 'spotlights-engine report'
  modules/
    vllm_v1_kv_offload.md           # module page: candidates table
    vllm_v1_kv_offload/
      <symbol-slug>__cand-<module>-NNNN.md   # per-candidate write-up + proposals
  evolve/
    vllm_v1_kv_offload/
      cand-<module>-NNNN/           # per-candidate 'prep-evolve'
        coral/
        nous/
        skydiscover/
  apply/
    vllm_v1_kv_offload/
      cand-<module>-NNNN/           # per-candidate 'apply'
        APPLY-NOTES.md
        apply.patch
        apply.prompt.txt
        manifest.json               # that session's tokens and cost
  sorted/                           # written by /spotlights-sort-candidates
    sorted_candidates.md            # ranked table linking each candidate
    sorted_candidates.json          # machine-readable ranking
    share-bundle/                   # written by /spotlights-share-candidates
    share-candidates.zip            # written by /spotlights-share-candidates
artifacts/
  spotlights_manager/               # checkpoints, raw transcripts (resumable)
    run_manifest.json               # provenance, token usage, and rate-table cost
```

Browse the full rendered run under [`examples/vllm_subset/`](examples/vllm_subset/).

## Requirements

> [!IMPORTANT]
> Spotlights drives **two** external agent CLIs — `claude` and `codex` — both installed and authenticated.

| Agent | Install | Used for |
|---|---|---|
| [Claude Code](https://docs.anthropic.com/en/docs/claude-code) | `curl -fsSL https://claude.ai/install.sh \| bash` | modules extractor, candidate discovery, Claude executors (steps 4–5) |
| [Codex](https://github.com/openai/codex) | `curl -fsSL https://chatgpt.com/codex/install.sh \| sh` | `module_deep_research`, Codex executors (steps 2, 5) |

> [!TIP]
> Tested and recommended models: **Opus 4.7** for Claude Code and **GPT-5.5** for Codex.

For LiteLLM gateway config, picking which model each CLI runs on, and a quick response check, see [docs/agent-cli-setup.md](docs/agent-cli-setup.md).

## Commands

| Command | Purpose |
|---|---|
| `spotlights-engine` | Main engine: structural map → candidates → proposals. Subcommands: `doctor`, `init`, `prep-evolve`, `apply`, `report`. |
| `signal-pipeline` | Telemetry-driven discovery (**preview**) — a separate entry point. |

## Skills

| Skill | Purpose |
|---|---|
| `/spotlights-objective-setting` | Optional interview that helps you frame a sharp optimization objective and prints ready-to-paste `--objective`/`--hint` flags. You can also write those flags by hand. |
| `/spotlights-sort-candidates` | Ranks a finished run's candidates by estimated impact, writing `sorted_candidates.md` (a summary table linking each candidate) and `sorted_candidates.json` under `<output-folder>/sorted/`. |
| `/spotlights-share-candidates` | Packages the top-N candidates from `sorted_candidates.md` into a self-contained ZIP an external teammate can unzip and open by double-clicking `index.html` — no server, works offline. |
| `/spotlights-apply-candidate` | Implements one candidate as a reviewable patch, in-session, in a throwaway git worktree. Produces `apply.patch` + `apply.prompt.txt` + `APPLY-NOTES.md` (no `manifest.json` from the interactive path). Runs no tests and no benchmarks; your checkout is never modified. |

<a id="quickstart"></a>
## Quickstart (Example on a vLLM subset)

**Prerequisites:** Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

**1. Install the engine** ([Quick install](docs/INSTALL.md#quick-install-recommended))

Quick install — puts `spotlights-engine` on your PATH (in `~/.local/bin`) via [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --force "git+https://github.com/AI-native-Systems-Research/spotlights.git@main"
```

Pin a version by replacing `main` with any git tag, branch, or commit. No shell activation needed after this.

> [!NOTE]
> Once the repo is public, this simplifies to a one-liner (TBD until then):
> ```bash
> curl -fsSL https://raw.githubusercontent.com/AI-native-Systems-Research/spotlights/main/install.sh | sh
> ```

Or install from source for development ([Development install](docs/INSTALL.md#from-source-development)):

```bash
git clone https://github.com/AI-native-Systems-Research/spotlights.git
cd spotlights
uv sync --all-extras
source .venv/bin/activate
```

**2. Install the bundled slash commands** ([Install the Spotlights skill](docs/INSTALL.md#install-the-spotlights-skill))

```bash
spotlights-engine init      # installs the /spotlights-* Claude Code slash commands
```

Installs system-wide into `~/.claude/commands/` by default, so the commands work from any directory. Pass `--scope project` to install into `./.claude/commands/` for this checkout only.

**3. Verify the environment end-to-end** ([Preflight with `doctor`](docs/INSTALL.md#preflight-with-doctor))

```bash
spotlights-engine doctor
```

**4. Run the engine.** For Example: Clone [vLLM](https://github.com/vllm-project/vllm), decide your objective, then run the engine:

> [!TIP]
> **Framing the objective (optional).** The run needs an `--objective` and, optionally, one or more `--hint`s. You can write them by hand, as below. Or, in Claude Code, run the optional [`/spotlights-objective-setting`](#skills) interview — it walks you through your goal and prints a ready-to-paste flag line (`--objective "…" --hint "…"`). Either path produces the same flags; the skill is just a convenience, never a required step.

```bash
spotlights-engine \
  --repo ../vllm \
  --include vllm/v1/kv_offload \
  --objective "reduce the median TTFT and median TPOT (Time Per Output Token)" \
  --hint "Multi-turn agentic workload" \
  --output-folder ./spotlights-out \
  --artifacts-dir ./artifacts
```

> [!IMPORTANT]
> Keep `--output-folder` and `--artifacts-dir` **outside** the `--repo` folder being scanned (as above: the repo is `../vllm`, the outputs land in the current directory). Pointing them inside the scanned repo mixes run output into the target's working tree and lets the engine pick up its own artifacts as source to analyze.

A single-module run like this takes roughly **30–45 minutes** and a **few dollars** in API cost against the paid `claude`/`codex` CLIs. Cost scales with the number of included modules — the full 8-module example run checked in under [`examples/vllm_subset/`](examples/vllm_subset) cost about **$26** at default rates. Exact cost depends on your model pricing; see [docs/cost-and-manifest.md](docs/cost-and-manifest.md).

**Limiting the scope with `--include`.** `--include` narrows the run to one or more slash-form qualified names — a package, a whole subtree, or a single leaf module:

```bash
# one module
spotlights-engine --include vllm/v1/kv_offload ...

# several scopes at once: repeat the flag, or pass multiple values after one flag
spotlights-engine --include vllm/v1/kv_offload vllm/v1/attention/paged_kv ...
spotlights-engine --include vllm/v1/kv_offload --include vllm/v1/worker ...
```

**5. Rank the candidates by impact (optional).** A finished run can surface ~100 candidates; the bundled `/spotlights-sort-candidates` slash command ranks them for your objective. In Claude Code, from the same directory:

```
/spotlights-sort-candidates
```

Point it at `./spotlights-out/result.json` and it writes a ranked `./spotlights-out/sorted/sorted_candidates.md` (a summary table linking each candidate to its write-up) plus a machine-readable `./spotlights-out/sorted/sorted_candidates.json`.

**6. Render the run as one browsable page (optional).** `spotlights-engine report` turns a finished run into a single self-contained `experiment.html` beside its `result.json` — inlined CSS and JS, no external assets, every chart paired with a table view, openable offline by double-clicking:

```bash
spotlights-engine report ./spotlights-out
```

Only `result.json` is required. The run manifest, a `sorted/` ranking and the `evolve/` and `apply/` trees are folded in when present — so if you ranked the candidates first, the page comes out ranked. Pass `-o` to write the page somewhere else.

**7. Share the top candidates (optional).** Once a run is ranked, the bundled `/spotlights-share-candidates` slash command packages the top-N candidates into a self-contained ZIP an external teammate can unzip and open by double-clicking `index.html` — no server, works offline. In Claude Code:

```
/spotlights-share-candidates
```

It asks for the folder containing `sorted_candidates.md` (e.g. `./spotlights-out/sorted/`) and a top-N (default `5`), then writes a `share-bundle/` folder and a `share-candidates.zip` next to it. The bundle contains rendered HTML for each candidate plus the original markdown, with external references (arxiv, doi, docs) kept clickable and module breadcrumbs left as plain text. If `prep-evolve` or `apply` has run, their bundles are folded in automatically: each candidate gains its evolve engines and its one-shot patch — the latter as a colorized diff with an apply recipe — with nothing verified and the page saying so.

## Signal sources & roadmap

The module map is a shared coordinate system: every signal source projects onto it, and candidates are the nodes where signals converge. This is where the project is headed.

| Signal source | What it contributes | Status |
|---|---|---|
| **Code structure** | Static analysis builds the module tree that grounds every candidate in a specific code region. | **Live** |
| **Research literature** | A deep-research engine retrieves findings from the web, arXiv, blogs, and docs, and derives evidence-backed proposals for each candidate. | **Live** |
| **Runtime telemetry** | OpenTelemetry traces, logs, and profiles surface bottlenecks visible only under load, not in the source. Wired through the [`signal-pipeline`](docs/telemetry-preview.md) flow. | **Live (preview)** |
| **Repository history** | Issues, pull requests, and commit history capture known limitations, past reasoning, and undocumented benchmarks that never reach code comments. | In development |
| **Technique-driven discovery** | Start from a paper or technique and search the codebase for where it could apply — the inverse of starting from a bottleneck. | Exploring |
| *…and more* | *The list isn't closed — if you have a signal source in mind, propose one via [Contributing](#contributing).* | *Open* |

Contributions to any of these are welcome — see [Contributing](#contributing).

## More

- **[Evolve bundles (`prep-evolve`)](docs/prep-evolve.md)** — turn a chosen candidate into a ready-to-run bundle for an external evolver (skydiscover, coral, nous).
- **[One-shot apply (`apply`)](docs/one-shot-apply.md)** — one Claude Code session turns a candidate into a reviewable patch plus its verification recipe. No fitness loop, no evaluator to write.
- **[Telemetry-driven discovery (preview)](docs/telemetry-preview.md)** — the `signal-pipeline` path: OpenTelemetry traces → candidates anchored to captured anomalies.
- **[Cost and the run manifest](docs/cost-and-manifest.md)** — how runs are priced, the rate-table format, and overriding it for your LiteLLM contract.
- **[The `SpotlightReport` format](docs/spotlight-report.md)** — the machine-readable `result.json` schema: candidates, proposals, findings, and how they join.
- **[Installation & configuration](docs/INSTALL.md)** — both CLIs, LiteLLM proxy, `doctor`, and the full flag reference.

## Where Spotlights fits

Spotlights decides *what* to optimize and proposes *how*. Its candidates and proposals are designed to hand off to execution backends — coding agents, evolutionary search, or experiment frameworks. Think of it as the scouting engine that points existing optimization machinery at the places most worth its effort.

## Contributing

The project is organized around signal sources, and the most useful contributions add or sharpen one:

- **New signal sources** — telemetry, repo history, and technique-driven discovery are the active frontiers (see the [roadmap](#signal-sources--roadmap)). The candidate/proposal pipeline and the module map are shared, so a new source mostly means projecting its evidence onto existing candidates.
- **Engine improvements** — better module extraction, candidate ranking, proposal quality, parallelism, or resumability.
- **Adapters** — support for target repos and goals beyond the vLLM examples.

Open an issue to discuss a direction before a large change. Bug reports and example runs on new repos are also valuable.

## License

Apache-2.0 — see [LICENSE](LICENSE).
