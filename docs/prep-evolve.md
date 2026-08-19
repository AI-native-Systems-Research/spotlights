# Evolve bundles (`prep-evolve`)

Spotlights decides *what* to optimize; **evolvers** (evolutionary code-search backends) do the *how*. The `prep-evolve` subcommand is the bridge: pick a candidate from a finished run and it generates a self-contained, ready-to-run **evolve bundle** for one of three external evolvers. Each bundle contains the native config, a seed/target laid out the way that evolver expects, the findings/proposals digest folded into the prompt the evolver reads, and an evaluator scaffold. It does **not** run the evolve; it hands you a directory to `cd` into plus the exact launch command.

← Back to [README](../README.md)

| Evolver (`--evolver`) | Edit scope | Native config | Run command |
|---|---|---|---|
| [`skydiscover`](https://github.com/skydiscover-ai/skydiscover) | single file (mutates the `# EVOLVE-BLOCK-START/END` region) | `config.yaml` + `seed.<ext>` (you write `evaluator.py`) | `skydiscover-run seed.<ext> evaluator.py -c config.yaml` |
| [`coral`](https://github.com/Human-Agent-Society/CORAL) | multi-file (agent edits a git worktree) | `task.yaml` (you write `grader/grader.py`) | `coral start --config task.yaml` |
| [`nous`](https://github.com/AI-native-Systems-Research/agentic-strategy-evolution) (alias `agentic-strategy-evolution`) | multi-file (experiment arms with `code_changes[]`) | `campaign.yaml` + `bundle.yaml` + `prompts/methodology/` | `NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml` |

Pass `--evolver all` to emit one bundle per compatible evolver (skydiscover is reported as skipped for multi-file selections).

### Usage

`prep-evolve` consumes a completed run and a target repo. Point `--result` at the
run directory (it finds `result.json`) and pass `--repo` at the checkout you want
the bundle to target; omit `--candidate` to generate a bundle for every candidate:

```bash
# every candidate → spotlights-out/evolve/<module>/<candidate>/skydiscover/
spotlights-engine prep-evolve --result ./spotlights-out \
  --repo ../vllm --evolver skydiscover

# one candidate
spotlights-engine prep-evolve --result ./spotlights-out \
  --repo ../vllm --candidate cand-vllm_v1_kv_offload-0002 --evolver skydiscover

# top 10 of a ranking → still under spotlights-out/evolve/… (not inside sorted/)
spotlights-engine prep-evolve --result ./spotlights-out/sorted \
  --repo ../vllm --evolver skydiscover --top-n 10
```

> [!TIP]
> **Prefer `--repo` over the `index.md` fallback.** If you omit `--repo`,
> the repo is read from the `Repo path:` line in the run's `index.md`. That path
> is whatever was recorded when the run was produced, so it is often stale or
> wrong on another machine or checkout — it may not exist, or (worse) point at a
> *different* checkout whose code no longer matches the recorded line ranges,
> which surfaces later as a confusing staleness error. Passing `--repo`
> explicitly makes the target unambiguous.
>
> **Check out the same commit the run was produced against.** `prep-evolve`
> validates each candidate's recorded symbol/line range against the *live* repo
> before writing a bundle. Point `--repo` at a checkout on the commit recorded
> in the run's `run_manifest.json` under `target.commit_sha` (from
> `target.repo_url`). On any other commit the code may have shifted, and the
> gate fails with a staleness error even though the run itself is fine:
>
> ```bash
> # commit the run targeted, e.g. from examples/vllm_subset/run_manifest.json
> git -C ../vllm checkout 83ad767eed3be3ee7f2df63be693bfaca5c7c922
> spotlights-engine prep-evolve --result ./spotlights-out --repo ../vllm --evolver skydiscover
> ```

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

### Evolver reference

The `--evolver` flag selects one of three external evolvers. They differ in edit
scope, what you must hand-author, and how they install — all three share the same
prerequisites (bottom of section).

#### `skydiscover`

LLM-driven evolutionary search over a single marked code region.

- **Repository:** https://github.com/skydiscover-ai/skydiscover
- **Edit scope:** single file — only the region between `# EVOLVE-BLOCK-START` and `# EVOLVE-BLOCK-END` changes.
- **Native config:** `config.yaml` + `seed.<ext>`
- **You must write:** `evaluator.py` ([what to write](https://github.com/skydiscover-ai/skydiscover/blob/main/README.md#%EF%B8%8F-what-you-write))
- **Install:** `pip install skydiscover`
- **Run:** `skydiscover-run seed.<ext> evaluator.py -c config.yaml`

#### `coral`

Agentic multi-file evolver — an LLM agent edits a git worktree of the repo.

- **Repository:** https://github.com/Human-Agent-Society/CORAL
- **Edit scope:** multi-file — the agent may edit any in-scope file in a git worktree.
- **Native config:** `task.yaml`
- **You must write:** the grader (`grader/grader.py`) + seed directories
- **Install:**
  ```bash
  # Shell installer
  curl -fsSL https://raw.githubusercontent.com/Human-Agent-Society/CORAL/main/install.sh | sh
  ```
  ```text
  # Claude Code plugin
  /plugin marketplace add Human-Agent-Society/CORAL
  /plugin install coral@coral-marketplace
  ```
- **Run:** `coral start --config task.yaml`
- **Quickstart:** open the target repo (make sure `task.yaml`'s `repo_path` points to the same location — update it if needed), then ask Claude Code — replacing `<CORAL_BUNDLE_PATH>` with the path to wherever you unpacked this bundle:
  > use coral to optimize this — start from the bundle at `<CORAL_BUNDLE_PATH>`. Don't change what task.yaml defines — the goal, in-scope file, oracle, metric, and direction are fixed. Fill only the gaps: write the grader, set up the seed, and add whatever's needed so the grader cleanly scores the seed. The seed should be the full repo so the agent can read everything, but the grader must reject any attempt that modifies or adds a file outside the allowlist — configured as `target_files` (or `allowed_paths`) under `grader.args` in `task.yaml`.

#### `nous` (alias `agentic-strategy-evolution`)

Runs experiment "arms" that apply `code_changes[]` across the target.

- **Repository:** https://github.com/AI-native-Systems-Research/agentic-strategy-evolution
- **Edit scope:** multi-file — experiment arms with `code_changes[]`.
- **Native config:** `campaign.yaml`
- **You must write:** nothing to hand-author — the agents discover metrics and evaluate on their own.
- **Install:** `pip install "git+https://github.com/AI-native-Systems-Research/agentic-strategy-evolution.git@reflective"`
- **Run:** `NOUS_CAMPAIGN_PARENT=$PWD/nous_runs nous run campaign.yaml`

#### Prerequisites (all evolvers)

Python 3, an LLM API key for the model named in the config (e.g. `ANTHROPIC_API_KEY`),
the target repo checked out at the run's commit, and a server with all the hardware
the benchmark needs (e.g. a GPU) to build and measure the target.
