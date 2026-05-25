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

<img src="docs/architecture/spotlights_deep_research_path.png" alt="Architecture" width="600">


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
  --objective "reduce the media TTFT and median TPOT (Time Per Output Token)" \
  --hint "multi-turn agentic workload" \
  --output-folder ./spotlight-out \
  --artifacts-dir ./artifacts
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

A sample run on this subset is checked in under
[examples/vllm_subset/](examples/vllm_subset/): browse the rendered
[index.md](examples/vllm_subset/index.md) and per-module pages under
[modules/](examples/vllm_subset/modules/), or inspect the raw
[result.json](examples/vllm_subset/result.json).

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
| `--max-findings-per-module` | `30` | Cap on findings produced by step 3 per module. |


Agent authentication is handled by the underlying `claude` and `codex` CLIs;
no engine config file is required for the happy path.



## License

Apache-2.0 — see [LICENSE](LICENSE).
