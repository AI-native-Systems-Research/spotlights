# Evolve bundles (`prep-evolve`)

Spotlights decides *what* to optimize; evolvers (evolutionary code-search backends) do the *how*. The `prep-evolve` subcommand is the bridge: pick a candidate from a finished run and it generates a self-contained, ready-to-run bundle for one of three external evolvers. Run the engine first (see the [README](../README.md) quickstart), then use this page.

← Back to [README](../README.md)

## Evolve bundles (`prep-evolve`)

Spotlights decides *what* to optimize; **evolvers** (evolutionary code-search backends) do the *how*. The `prep-evolve` subcommand is the bridge: pick a candidate from a finished run and it generates a self-contained, ready-to-run **evolve bundle** for one of three external evolvers. Each bundle contains the native config, a seed/target laid out the way that evolver expects, the findings/proposals digest folded into the prompt the evolver reads, and an evaluator scaffold. It does **not** run the evolve; it hands you a directory to `cd` into plus the exact launch command.

| Evolver (`--evolver`) | Edit scope | Native config | Run command |
|---|---|---|---|
| `skydiscover` | single file (mutates the `# EVOLVE-BLOCK-START/END` region) | `config.yaml` + `seed.<ext>` (you write `evaluator.py`) | `skydiscover-run seed.<ext> evaluator.py -c config.yaml` |
| `coral` | multi-file (agent edits a git worktree) | `task.yaml` (you write `eval/grader.py`) | `coral start --config task.yaml` |
| `nous` (alias `agentic-strategy-evolution`) | multi-file (experiment arms with `code_changes[]`) | `campaign.yaml` + `bundle.yaml` + `prompts/methodology/` | `NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml --bundle bundle.yaml` |

Pass `--evolver all` to emit one bundle per compatible evolver (skydiscover is reported as skipped for multi-file selections).

### Usage

`prep-evolve` consumes a completed run's `result.json` plus a `(module, candidate)` selection. Run the engine first (see [Quickstart](../README.md#quickstart)), then point at the same target repo:

```bash
spotlights-engine prep-evolve \
  --result   ./spotlights-out/result.json \
  --module   vllm/v1/kv_offload \
  --candidate cand-0002 \
  --repo     ../vllm \
  --evolver  skydiscover \
  --out      ./evolve_bundles
```

`--module` takes the slash-form qualified name shown in `index.md`; `--candidate` takes the candidate id from the module page. The target repo is resolved from `--repo` (preferred); if omitted, `--index path/to/index.md` supplies the `Repo path:` recorded by the run. One of the two must resolve to an existing directory, or the command fails before writing anything.

### What lands on disk

One directory per `(candidate × evolver)`, named `<repo>__<module>__<candidate>__<evolver>/`. Alongside the evolver-native files, every bundle (except the single-file Nous campaign) includes:

- `README.md` — the copy-paste run command, the in-scope files, and the **evaluation-gap warning**.

The findings/proposals digest is embedded directly in each evolver's native config (the grader/evaluator prompt or system message) rather than written as a standalone file.

> [!IMPORTANT]
> **The evaluation gap is real.** Every evolver needs a project-specific measurement loop (build the target, run a benchmark, parse the metric). `prep-evolve` parses the correctness/performance oracle out of the candidate's `evolve_rationale` and objective and pre-fills the evaluator/grader scaffold, but the performance measurement is left as a clearly-marked `# TODO`. The bundle is launchable end-to-end immediately, but **results are not meaningful until you complete the evaluator** — each bundle's `README.md` states what Spotlights believes the oracle is.

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--result` | (required) | Path to a finished run's `result.json`. |
| `--module` | (required) | Slash-form qualified name, e.g. `vllm/v1/attention`. |
| `--candidate` | (required) | Candidate id, e.g. `cand-0002`. |
| `--evolver` | (required) | `skydiscover` \| `coral` \| `nous` \| `agentic-strategy-evolution` \| `all`. |
| `--out` | (required) | Parent directory the bundle dir is written under. |
| `--repo` | (none) | Target repo path; wins over `--index`. |
| `--index` | (none) | Rendered `index.md`, used only as a `--repo` fallback. |
| `--scope` | `candidate` | `candidate` or `module-main-files` (CORAL/Nous only — adds the module's `main_files` as editable targets). |
| `--direction` | (inferred) | `minimize` \| `maximize`; overrides the direction inferred from the objective verb. |
| `--model` | (evolver default) | Override the default evolver LLM model. |
| `--force` | off | Re-run over an existing bundle, overwriting every generated file. |

Re-runs are guarded by default: if the bundle directory already exists, the command fails unless `--force` is set. With `--force`, every generated file is overwritten — including a hand-edited evaluator/grader — so copy out any evaluator work you want to keep before re-running.
