# Evolve bundles (`prep-evolve`)

Spotlights decides *what* to optimize; **evolvers** (evolutionary code-search backends) do the *how*. The `prep-evolve` subcommand is the bridge: pick a candidate from a finished run and it generates a self-contained, ready-to-run **evolve bundle** for one of three external evolvers. Each bundle contains the native config, a seed/target laid out the way that evolver expects, the findings/proposals digest folded into the prompt the evolver reads, and an evaluator scaffold. It does **not** run the evolve; it hands you a directory to `cd` into plus the exact launch command.

← Back to [README](../README.md)

| Evolver (`--evolver`) | Edit scope | Native config | Run command |
|---|---|---|---|
| [`skydiscover`](https://github.com/skydiscover-ai/skydiscover) | single file (mutates the `# EVOLVE-BLOCK-START/END` region) | `config.yaml` + `seed.<ext>` (you write `evaluator.py`) | `skydiscover-run seed.<ext> evaluator.py -c config.yaml` |
| [`coral`](https://github.com/Human-Agent-Society/CORAL) | multi-file (agent edits a git worktree) | `task.yaml` (you write `eval/grader.py`) | `coral start --config task.yaml` |
| [`nous`](https://github.com/AI-native-Systems-Research/agentic-strategy-evolution) (alias `agentic-strategy-evolution`) | multi-file (experiment arms with `code_changes[]`) | `campaign.yaml` + `bundle.yaml` + `prompts/methodology/` | `NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml --bundle bundle.yaml` |

Pass `--evolver all` to emit one bundle per compatible evolver (skydiscover is reported as skipped for multi-file selections).

### Usage

`prep-evolve` consumes a completed run and a target repo. Point `--result` at the
run directory (it finds `result.json` and reads the repo path from `index.md`);
omit `--candidate` to generate a bundle for every candidate:

```bash
# every candidate → spotlights-out/evolve/<module>/<candidate>/skydiscover/
spotlights-engine prep-evolve --result ./spotlights-out --evolver skydiscover

# one candidate
spotlights-engine prep-evolve --result ./spotlights-out \
  --candidate cand-vllm_v1_kv_offload-0002 --evolver skydiscover

# top 10 of a ranking → still under spotlights-out/evolve/… (not inside sorted/)
spotlights-engine prep-evolve --result ./spotlights-out/sorted \
  --evolver skydiscover --top-n 10
```

`--result` accepts any run artifact and self-locates the rest: a run directory, a
`result.json` file, an `index.md`, a `sorted/` directory, or a
`sorted_candidates.json`/`.md`. `--candidate` takes the candidate id from the
module page and is optional (omitting it processes all candidates); the module is
inferred from the id, so `--module` is no longer required. When `--result` points
at a sorted source, omitting `--candidate` processes the ranked candidates in rank
order, and `--top-n <N>` builds only the top N (default `all`); `--top-n` with a
plain `result.json` is an error. The target repo is resolved from `--repo`, else
from the `Repo path:` line in the run directory's `index.md`. Bundles are written
to `<base>/evolve/<module>/<candidate>/<evolver>/`, where `<base>` is `--out` when
given, else the run directory. Candidates an evolver cannot handle are skipped
with a warning, and existing bundles are skipped unless `--force` is set.

### What lands on disk

Bundles are written to `<base>/evolve/<module>/<candidate>/<evolver>/`, mirroring
the `modules/` tree. Alongside the evolver-native files, every bundle (except the
single-file Nous campaign) includes:

- `README.md` — the copy-paste run command, the in-scope files, and the **evaluation-gap warning**.

The findings/proposals digest is embedded directly in each evolver's native config (the grader/evaluator prompt or system message) rather than written as a standalone file.

> [!IMPORTANT]
> **The evaluation gap is real.** Every evolver needs a project-specific measurement loop (build the target, run a benchmark, parse the metric). `prep-evolve` parses the correctness/performance oracle out of the candidate's `evolve_rationale` and objective and pre-fills the evaluator/grader scaffold, but the performance measurement is left as a clearly-marked `# TODO`. The bundle is launchable end-to-end immediately, but **results are not meaningful until you complete the evaluator** — each bundle's `README.md` states what Spotlights believes the oracle is.

### Flags

| Flag | Default | Purpose |
|---|---|---|
| `--result` | (required) | A finished run's `result.json`, the run directory containing it, an `index.md`, a `sorted/` dir, or a `sorted_candidates.{json,md}`. |
| `--module` | (inferred) | Slash-form qualified name; inferred from the candidate id when omitted. |
| `--candidate` | (all) | Candidate id, e.g. `cand-…-0002`. Omit to process every candidate. |
| `--top-n` | `all` | For a sorted `--result`: build only the top N ranked candidates. Error with a plain source. |
| `--out` | (run dir) | Base directory; the `evolve/…` tree is written under it. Defaults to the run directory. |
| `--evolver` | (required) | `skydiscover` \| `coral` \| `nous` \| `agentic-strategy-evolution` \| `all`. |
| `--repo` | (none) | Target repo path; wins over `--index`. |
| `--index` | (none) | Rendered `index.md`, used only as a `--repo` fallback. |
| `--scope` | `candidate` | `candidate` or `module-main-files` (CORAL/Nous only — adds the module's `main_files` as editable targets). |
| `--direction` | (inferred) | `minimize` \| `maximize`; overrides the direction inferred from the objective verb. |
| `--model` | (evolver default) | Override the default evolver LLM model. |
| `--force` | off | Re-run over an existing bundle, overwriting every generated file. |

Re-runs resume cleanly: an existing bundle directory is skipped (not an error)
unless `--force` is set. With `--force`, every generated file in that bundle is
overwritten — including a hand-edited evaluator/grader — so copy out any evaluator
work you want to keep before re-running.
