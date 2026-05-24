# Spotlight Engine

Given a target repo and an objective, Spotlight Engine proposes evidence-backed,
high-leverage code changes. It works in three tiers:

1. **Structural map** — extract the project's modules.
2. **Candidates** — identify symbols worth investigating per module.
3. **Proposals** — produce literature-derived and agent-derived change proposals
   for each candidate.

The output is a tree of plain Markdown files you browse in any viewer (GitHub,
VS Code preview, Obsidian).

## Architecture

![Architecture](docs/architecture/spotlights_deep_research_path.png)

See [architecture overview](docs/architecture/spotlights_deep_research_path_architecture.md)
for details.

## Install

Prerequisites:

- Python ≥ 3.11
- [`uv`](https://github.com/astral-sh/uv)
- The `claude` CLI on `PATH`, with auth configured via its own login state or
  supported environment variables. Used by the modules extractor, candidate
  discovery, and the Claude executors for steps 4 and 5.
- The `codex` CLI on `PATH`, with auth configured via its own login state or
  supported environment variables. Used by `module_deep_research` (hard-coded
  today) and the Codex executors for steps 2 and 5.
- A target repo on disk (the quickstart below uses [vLLM](https://github.com/vllm-project/vllm)).

Both CLIs are required for the default end-to-end path; the engine will not
run without one of them today.

```bash
git clone https://github.com/Video-AI/spotlight-engine.git
cd spotlight-engine
uv sync --all-extras
```

## Quickstart on a vLLM subset

Clone vLLM next to this repo, then:

```bash
spotlight-engine \
  --repo ../vllm \
  --include v1.kv_offload \
  --objective "reduce decode latency on long-context workloads" \
  --hint "prefill-heavy traffic, batch size 1-8" \
  --hint "Hopper GPUs, FP8 KV cache" \
  --output-folder ./spotlight-out \
  --artifacts-dir ./artifacts
```

Expected stdout shape:

```
[1/5] modules_extractor … 1 module kept (v1.kv_offload)
[2/5] candidate_discovery (v1.kv_offload) … 6 candidates
[3/5] module_deep_research (v1.kv_offload) … 4 findings
[4/5] proposal_from_finding_creator (v1.kv_offload) … 9 proposals attached
[5/5] agent_proposals (v1.kv_offload) … 7 agent proposals attached
results: ./spotlight-out/index.md
```

On disk:

```
spotlight-out/
  index.md                          # repo-level summary, one row per module
  modules/
    v1.kv_offload.md                # module page: candidates table
    v1.kv_offload/
      <symbol-slug>__<candidate-id>.md   # full per-candidate write-up + proposals
artifacts/
  spotlights_manager/               # checkpoints, raw transcripts (resumable)
```

`--include` accepts one or more dot-form leaf qualified names. Repeat the flag
or pass several values after a single flag:

```bash
spotlight-engine --include v1.kv_offload v1.attention.paged_kv ...
spotlight-engine --include v1.kv_offload --include v1.attention.paged_kv ...
```

## Configuration

All flags are optional once `--repo` and the agent CLIs are available.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--repo` | `../vllm` | Target repo path. |
| `--include` | (all modules) | Restrict to dot-form leaf qualified names. |
| `--objective` | `"reduce hot-path latency on common workloads"` | Threaded into discovery + deep research. |
| `--hint` (repeatable) | `[]` | Workload hints; map to `SpotlightContext.workload_hints`. |
| `--output-folder` | `./spotlight-out` | Where `index.md` and module pages land. |
| `--artifacts-dir` | `./artifacts` | Checkpoints + raw transcripts (resume key). |
| `--max-parallel` | `1` | Modules processed concurrently. |
| `--max-parallel-pairs` | `5` | Within-step parallelism for step 4. |
| `--max-parallel-candidates` | `5` | Within-step parallelism for step 5. |
| `--max-findings-per-module` | `10` | Cap on findings produced by step 3 per module. |
| `--no-resume` | resume on | Refuse to start over an existing run dir. |
| `--debug-first-n-pairs` | off | Cap step 4 (debug only). |
| `--debug-first-n-candidates` | off | Cap step 5 (debug only). |

Agent authentication is handled by the underlying `claude` and `codex` CLIs;
no engine config file is required for the happy path.

## How it fits together

The Spotlight Engine project is split across three repos:

- `spotlight-engine` (this repo) — the spine: schemas, orchestration, CLI.
- `spotlight-observability` — telemetry collection and signal extraction.
- `spotlight-knowledge` — literature index and experimental archive.

Dependency direction: `spotlight-engine` depends on the two leaf repos. The
graph is a tree, not a cycle.

## Library usage

```python
from pathlib import Path

from spotlights_engine import (
    SpotlightsManagerConfig,
    ModuleFilter,
    run_with_telemetry,
)
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.schemas.pipeline import SpotlightsManagerInput

result = run_with_telemetry(
    SpotlightsManagerInput(
        repo_path=Path("../vllm"),
        context=SpotlightContext(
            objective="reduce decode latency on long-context workloads",
            workload_hints=["prefill-heavy traffic", "Hopper GPUs"],
        ),
    ),
    config=SpotlightsManagerConfig(
        artifacts_dir=Path("./artifacts"),
        output_folder=Path("./spotlight-out"),
        module_filter=ModuleFilter(include=["v1.kv_offload"]),
    ),
)
print(result.renderer_result.index_path)
```

The `schemas` module (`Candidate`, `Change`, re-exported `Signal` types) is
the public contract surface; everything else may change without notice.

## Status & contributing

Stage: **alpha**. The schemas are stable; orchestration internals and per-step
agent prompts are still moving. File issues at
<https://github.com/Video-AI/spotlight-engine/issues>.

## License

Apache-2.0 — see [LICENSE](LICENSE).
