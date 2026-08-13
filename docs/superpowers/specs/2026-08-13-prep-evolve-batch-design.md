# prep-evolve: batch-first, run-directory-aware CLI

**Date:** 2026-08-13
**Status:** Design — approved for planning
**Branch:** `feat/prep-evolve-batch`

## Problem

Running `prep-evolve` today requires five arguments, and the two hardest to
supply — `--module` and `--candidate` — are exactly the ones a user must dig out
of the rendered `index.md`:

```bash
spotlights-engine prep-evolve \
  --result   ./spotlights-out/result.json \
  --module   vllm/v1/kv_offload \
  --candidate cand-0002 \
  --repo     ../vllm \
  --evolver  skydiscover \
  --out      ./evolve_bundles
```

On a module page the candidate id lives *inside the link filename*
(`…__cand-vllm_v1_kv_offload-0009.md`), not as visible text, so the user reads a
page, mentally extracts both the module qn and the candidate id, and retypes
them. The remaining three arguments (`--result`, `--repo`, `--out`) are stable
boilerplate.

Two facts from the real run output (`examples/vllm_subset/`) drive the
simplification:

- **The candidate id is globally unique and already encodes its module** —
  e.g. `cand-vllm_v1_kv_offload-0009`. All 173 candidate ids in the example run
  are unique across every module. So `--module` carries no information the
  candidate id doesn't already have.
- **`result.json` does not record the local repo path** — only the rendered
  `index.md` does (`- **Repo path:** …`). That is why `--repo`/`--index` exist.

## Goals

- Collapse the required arguments to the minimum a user cannot derive.
- Make the "browse `index.md`, then generate" flow require no hand-extraction.
- Support generating bundles for **all** candidates in one run, without
  depending on any ranking artifact (`sorted_candidates.json`).
- Lay bundle output out in a tree that mirrors the `modules/` pages the user
  already browses.

## Non-goals

- No new ranking computed here. `prep-evolve` never *ranks* candidates; it only
  *consumes* an existing `sorted_candidates.json` when the user points `--result`
  at one. Selection stays "one named candidate", "all candidates", or "the
  ranking's top-N" — never a heuristic pick made by this tool.
- No *hard* dependency on `sorted_candidates.json`. Plain sources work without it;
  it is an optional ranked source, not a required input.
- No change to bundle *contents* (the per-candidate files each evolver emits are
  unchanged).

## Design

### 1. `--result` accepts any run artifact and self-locates the rest

Every accepted form resolves to the same triple — the `result.json` to load, the
**run directory** (the folder holding `result.json`), and the sibling `index.md`
used as the `--repo` fallback — plus, for the two sorted forms, a **ranking
source**. `--repo`/`--index` remain explicit overrides that win over the
auto-resolved value.

| `--result` points at | run directory | `result.json` | ranking source |
|---|---|---|---|
| a run directory (`./spotlights-out`) | itself | `<dir>/result.json` | — |
| a `result.json` (or any other file) | its parent | the file itself | — |
| an `index.md` | its parent | `<parent>/result.json` | — |
| a `sorted/` directory | its **parent** | `<parent>/result.json` | `<dir>/sorted_candidates.json` |
| a `sorted_candidates.json` | parent's parent | `<run>/result.json` | the file itself |
| a `sorted_candidates.md` | parent's parent | `<run>/result.json` | sibling `sorted_candidates.json` |

Notes:

- A directory is a **run dir** when it contains `result.json`, else a **sorted
  dir** when it contains `sorted_candidates.json`/`.md`; otherwise it is an
  error. The two sorted *file* forms and the sorted *dir* form all live one level
  below the run dir (`<run>/sorted/…`), so the run dir is resolved by going up.
- The candidate **data** always comes from `result.json` (the sorted files carry
  only the ranking — rank/id/symbol/score/locations — not the `proposals`,
  `evolve_rationale`, or `findings` a bundle needs). A sorted source only
  supplies the **order** and the **top-N cut** (§2); each ranked id is then
  resolved back to its full candidate in `result.json`.
- The ranking is read from `sorted_candidates.json` (the machine-readable file);
  pointing at `sorted_candidates.md` reads its sibling `.json`. A sorted source
  with no `sorted_candidates.json` is an error.
- The output base still defaults to the run directory (§4) — bundles land at
  `<run>/evolve/…`, never inside `sorted/`.

### 2. `--candidate` optional (batch-first); `--top-n` for ranked sources

- **`--candidate` given:** generate a bundle for that one candidate only
  (today's behavior). An explicit candidate always wins — `--top-n` and any
  ranking are ignored.
- **`--candidate` omitted, plain source** (run dir / `result.json` / `index.md`):
  generate bundles for **every candidate in `result.json`**, iterating
  `module_runs` in document order. No ranking file is read.
- **`--candidate` omitted, sorted source**: generate bundles for the ranked
  candidates in `sorted_candidates.json`, in rank order. `--top-n <N>` builds only
  the top `N`; `--top-n all` (the default) builds every ranked candidate. Each
  ranked id is resolved to its full candidate in `result.json`; a ranked id that
  is missing or stale in `result.json` is skipped-with-a-warning (§5).

`--top-n` is meaningful only with a sorted source — "top" needs a ranking.
Passing `--top-n <N>` with a plain source is an error (a plain `result.json` has
no rank order to take the top of). Selection never invents a default candidate;
omission means "all" (or "top-N of the ranking"), never a heuristic pick.

### 3. `--module` is dropped

The candidate id uniquely identifies its module run, so the tool locates the
module by scanning `module_runs` for the id. `--module` is removed from the
required set. If still passed (back-compat), it is validated against the
candidate's resolved module and the command errors on a mismatch. In batch mode
`--module` is irrelevant.

### 4. Uniform output layout; `--out` only relocates the base

The layout is **always**:

```
<base>/evolve/<module>/<candidate>/<evolver>/
```

where `<module>` is the flattened module name matching the existing `modules/`
convention (`vllm/v1/kv_offload` → `vllm_v1_kv_offload`), `<candidate>` is the
full candidate id, and `<evolver>` is the evolver key. The base is:

- `--out <dir>` when given → `<dir>/evolve/…`
- else the **run directory** — the directory containing `result.json`. This is
  the `--result` directory itself when `--result` is a directory, or the parent
  of the file when `--result` is a bare `result.json`. Either way →
  `spotlights-out/evolve/…`.

This replaces the old flat `repo__module__cand__evolver` directory name
entirely — one consistent structure regardless of how output is located. The
`<evolver>/` level groups a candidate's bundles so `--evolver all` drops
`skydiscover/`, `coral/`, `nous/` side by side under the same candidate.

Example (`--out my-out`):

```
my-out/
└── evolve/
    └── vllm_v1_kv_offload/
        ├── cand-vllm_v1_kv_offload-0001/
        │   └── skydiscover/
        │       ├── seed.py
        │       ├── config.yaml
        │       └── README.md
        └── …
```

### 5. Batch resilience

- **Incompatible candidates are skipped, not fatal.** An evolver may reject a
  candidate (skydiscover requires a single file, a real line range, and a
  `#`-comment language). In batch this is a skip-with-warning that keeps the run
  going, mirroring how `--evolver all` already skips unsupported evolvers. A
  single-candidate request that is unsupported still errors loudly (unchanged).
- **Existing bundles are skipped unless `--force`.** With hundreds of bundle
  dirs, re-runs should resume cleanly: an existing bundle dir is skipped (not a
  hard error) when `--force` is absent, and overwritten when it is present.
  User-written evaluator files (`evaluator.py`, `eval/grader.py`) are never
  generated and therefore never clobbered regardless.
- **`--evolver all` in batch** is a candidate × evolver fan-out under the same
  skip-and-warn rules.
- **Summary line.** The command prints a final tally, e.g.
  `168 bundles written, 5 skipped`, followed by one line per skip with its
  reason.

## Bundle contents (unchanged, for reference)

A single skydiscover bundle is three files: `seed.py` (the candidate's whole
source file with `# EVOLVE-BLOCK-START/END` markers around its line range,
read from the live repo), `config.yaml` (skydiscover native config whose
`system_message` embeds the objective + findings/proposals digest), and
`README.md` (run command, in-scope files, correctness/performance oracle, and
the evaluation-gap warning). `coral` emits `task.yaml` + `README.md`; `nous`
emits a single `campaign.yaml`. Generating any bundle reads the target repo, so
the checkout must be present.

## Resulting CLI

```bash
# every candidate → spotlights-out/evolve/<module>/<candidate>/skydiscover/
spotlights-engine prep-evolve --result ./spotlights-out --evolver skydiscover

# one candidate → spotlights-out/evolve/vllm_v1_kv_offload/cand-…-0002/skydiscover/
spotlights-engine prep-evolve --result ./spotlights-out \
  --candidate cand-vllm_v1_kv_offload-0002 --evolver skydiscover

# top 10 of the ranking → still under spotlights-out/evolve/… (not inside sorted/)
spotlights-engine prep-evolve --result ./spotlights-out/sorted --evolver skydiscover --top-n 10

# a bare index.md or sorted_candidates.json works too; the rest is self-located
spotlights-engine prep-evolve --result ./spotlights-out/index.md --evolver coral
spotlights-engine prep-evolve --result ./spotlights-out/sorted/sorted_candidates.json --evolver coral --top-n 25

# relocate output; same structure underneath
spotlights-engine prep-evolve --result ./spotlights-out --evolver skydiscover --out my-out
```

Required now: `--result`, `--evolver`. Optional: `--candidate` (default: all, or
the ranking's top-N when `--result` is a sorted source), `--top-n` (default:
`all`; sorted sources only), `--out` (default: the run directory — the folder
containing `result.json`), `--repo`/`--index` (default: from run dir's
`index.md`), plus the existing `--scope`, `--direction`, `--model`, `--force`.

## Backward compatibility

- `--result` pointing at a `result.json` file keeps working.
- `--module` is accepted and validated if passed, so existing scripts do not
  break; it is simply no longer required.
- Passing `--candidate` reproduces single-bundle behavior; the only visible
  change for existing single-candidate invocations is the output path
  (`<base>/evolve/<module>/<candidate>/<evolver>/` instead of the old flat
  `repo__module__cand__evolver` dir).

## Open questions

None outstanding. The layout, selection semantics, argument set, and batch
behavior are settled above.
