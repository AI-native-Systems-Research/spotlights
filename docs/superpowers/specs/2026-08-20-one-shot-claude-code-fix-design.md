# One-shot Claude Code fix — design

**Date:** 2026-08-20
**Status:** approved design, not yet planned

## Problem

After a Spotlights run produces ranked candidates, the only supported next step
is `prep-evolve`: generate a bundle for an external evolver (skydiscover, CORAL,
nous) and run an evolutionary search. That path is powerful but expensive — each
evolver needs a project-specific fitness function, and every bundle ships with a
deliberately unfinished evaluator (the "evaluation gap").

Many candidates don't warrant a search. They warrant *one attempt*: read the
candidate, read the research findings behind it, implement the change, hand back
something reviewable. That is what a coding agent is for.

This design adds that cheap arm: a one-shot Claude Code fix per candidate,
producing a patch plus the recorded verification recipe.

## Scope and non-goals

**In scope:** turning one candidate into one reviewable patch, with the
candidate's oracles carried alongside it.

**Explicitly not in scope:**

- Running tests. See "No verification here" below.
- Benchmarking or measuring the performance metric.
- Scoring, ranking, or comparing fixes.
- Modifying the target repo in any way.

The deliverable is *a proposal faithfully implemented and documented*, not a
measured win. The evolvers exist for the measured-win case; the whole point of
this path is that it has no fitness loop.

## No verification here

The skill and the stage run no tests and no benchmarks. Two independent reasons:

**The machine may not be able to.** Spotlights' own vLLM example records
`performance_oracle: TTFT, TPOT` — metric *names*, with no harness, no workload,
and no baseline. Measuring them needs a GPU and a serving benchmark. A laptop
cannot produce that number, and a fabricated one is worse than none.

**The oracle is a prose regex.** `parse_correctness_oracles`
(`prep_evolve/extract.py:62-82`) scrapes the candidate's LLM-written
`evolve_rationale` for `pytest`-shaped strings and test-file-shaped paths, then
prefixes bare paths with `pytest `. Nothing checks the file exists; nothing runs
it against the baseline. `pytest tests/v1/worker/test_gpu_model_runner.py` reads
as authoritative and is a regex hit on a sentence.

**A fresh worktree can't run them anyway.** The isolation strategy below gives
the agent a clean checkout with no build artifacts, no venv, and no compiled
extensions. For a repo like vLLM that worktree is unusable for execution. This
is consistent only because we decided not to execute.

> If a future version wants the stage to verify, worktree isolation stops
> working for any repo that needs a build, and it needs the real checkout or a
> container instead. That is a different design, not an increment on this one.

### The oracle still travels with the patch

The oracle is not CORAL's concept — it is Spotlights' own candidate-admission
bar. `candidate_discovery/prompts_data/bootstrap.md` requires every candidate to
name a concrete correctness oracle, and `review.md` *deletes* candidates that
lack one. What is evolver-specific is the **grader**: the machinery that
executes an oracle inside a scoring loop.

So the fix path carries the oracle forward verbatim as the verification recipe
for whoever has the hardware, and never pretends to enforce it.

## Architecture

```
one_shot_fix/prompts.py          ← the fix prompt. single source of truth.
        │
        ├─ spotlights-engine fix              → resolve, worktree, validate,
        │                                       claude -p, collect patch + notes
        │     └─ --print-prompt               → print the prompt, exit
        │
        └─ /spotlights-fix-candidate          → interactive: pick candidate,
                                                get prompt via --print-prompt,
                                                do the work in-session
```

The engine stage is the product. The skill is the interactive front door. The
prompt is shared; the mechanics are not (see "Shared parts").

### Reuse of `prep_evolve`

`one_shot_fix` imports `resolve`, `extract`, `spec`, `digest`, and
`validate_target` from `prep_evolve`. Those five modules are already
evolver-agnostic — only `adapters/` and `yaml_emit.py` are evolver-specific — so
the dependency is honest even though the package name reads oddly from outside.

One new consumer does not pay for moving them to a neutral `candidate_context/`
package plus the resulting import and test churn. **A third consumer is the
trigger for that move.**

### Shared parts

`prompts.py` is shared between the stage and the skill. The worktree / diff /
notes plumbing lives in Python, where it is unit-testable and cannot forget
`git add -N` and silently drop an added file.

`--print-prompt` therefore does more than print: it **creates and validates the
worktree** (steps 2-4 below), then prints the prompt together with the worktree
path and exits, leaving the worktree in place. Otherwise the skill would have to
reimplement the staleness gate — the design's only correctness check — in
markdown, or validate against `--repo`, which is the mismatch this design
explicitly rejects.

That leaves exactly one duplication: **collection**. The skill restates
`git add -N` → `git diff` → write notes → remove worktree in markdown. A short
mechanical checklist in two places, which must be kept in sync.

It buys a genuinely interactive skill — you can steer, interrupt, and ask why
mid-fix — rather than watching a nested `claude -p` you cannot influence. A
future `fix --collect` mode would remove even this duplication by handing
collection back to Python; deferred because it adds a third mode for one
checklist.

## `spotlights-engine fix`

Per candidate:

1. **Resolve** run + repo + candidate. `--result` self-locates exactly as
   `prep-evolve` does (run dir, `result.json`, `index.md`, `sorted/`, or
   `sorted_candidates.{json,md}`).
2. **Capture the base SHA** via `capture_revision`. Unlike `prep-evolve`, which
   tolerates a non-git target, `fix` requires a real git checkout and fails
   loudly without one — a worktree cannot be made otherwise.
3. **Create the worktree:**
   `git -C <repo> worktree add --detach <tmp> <base-sha>`.
   `--detach` is mandatory: without it git creates a branch named after the path
   basename *in the target repo*, violating the repo-untouched requirement.
4. **Validate, against the worktree.** The five checks from
   `validate_candidate_target`: path containment, file existence, line bounds,
   excerpt sha256, and the two-part staleness heuristic (a symbol identifier
   token within `_SYMBOL_WINDOW` = 5 lines of the recorded range, and every
   container of a qualified name still present somewhere in the file).

   Running these *inside the worktree* rather than at `--repo` is deliberate:
   the bytes validated are then exactly the bytes the agent edits, the recorded
   sha256 is truthful, and your checkout may be dirty while a fix runs.
   `prep-evolve` has no such concern because it validates and points the evolver
   at the same path; the worktree introduces the asymmetry.

   With no tests being run, **this gate is the only correctness check in the
   design.** It is what stops the agent editing the wrong function after a file
   has drifted.
5. **Build the prompt** (`prompts.py`): objective and direction, in-scope files
   with validated line ranges, current approach and evolve rationale, research
   findings with URLs, existing proposals as seeds, the oracles verbatim, and
   the base commit.
6. **Run** `claude -p <prompt>` in the worktree, via a `claude_exec` modeled on
   `agent_proposals/claude_exec.py`: env scrubbing, `max_turns`, wallclock cap,
   `--output-format stream-json`, usage capture for costing.
7. **Collect:** `git add -N .` (so added files appear in the diff), then
   `git diff` → `fix.patch`; write `FIX-NOTES.md`.
8. **Clean up in a `finally`:** `git worktree remove --force` plus
   `git worktree prune`. Without this, a timed-out or crashed run leaves stale
   entries in `.git/worktrees`, and a sweep leaves N of them.

**Failure semantics** match `prep_evolve`'s batch loop: a single explicit
`--candidate` raises; a `--top-n` sweep records a per-candidate skip with a
reason and continues. A step-4 failure removes the worktree first.

**`--print-prompt`** runs steps 1-5, then prints the prompt and the worktree
path and exits, leaving the worktree in place for the caller. It does not run
`claude`, collect artifacts, or remove the worktree. This is what the skill
consumes — it gets a validated worktree and the prompt in one call, so the
staleness gate is never reimplemented in markdown.

**Batching** starts single-candidate (`--candidate`). `--top-n N` / all is a
loop over the same path once one candidate works, with the same argument shape
as `prep-evolve`. Worktrees are created and removed one at a time.

## `/spotlights-fix-candidate`

A bundled skill in `templates/commands/fix-candidate/`, installed by
`spotlights-engine init` as `.claude/commands/spotlights-fix-candidate/`
(the `spotlights-` prefix is applied at install time). Run from the target repo:

1. Resolve the run directory, repo, and candidate. With no candidate given,
   show the ranked list from `sorted_candidates.json` and ask which — the only
   interactive step.
2. Run `spotlights-engine fix --print-prompt --candidate <id>`, which returns
   the prompt and the path to a created, validated worktree. On a
   `StalenessError`, stop and print the commit to check out.
3. Implement the change in that worktree, in-scope files only, in-session — so
   you can steer and review before any patch is written.
4. Collect: `git add -N .` → `git diff` → `fix.patch`, write `FIX-NOTES.md`,
   then `git worktree remove --force` and `git worktree prune`.
5. Print where the artifacts landed and the apply-and-verify commands.

There is no plan-approval step. If the change cannot be made within scope, the
skill writes notes explaining why and produces no patch, rather than a patch
that cannot be trusted.

## Artifacts

`<run-dir>/fix/<module-slug>/<candidate-id>/`, mirroring the existing
`evolve/<module-slug>/<candidate-id>/<evolver>/` layout and reusing `slug_for`.
`<run-dir>` is `--out` when given, else the run directory — same rule as
`prep-evolve`.

**`fix.patch`** — `git diff` output against the base commit, with the SHA in a
header comment. Not `format-patch`: that needs a commit, and the repo stays
untouched.

**`FIX-NOTES.md`** — candidate id, module, objective, direction, **base
commit**, in-scope files with line ranges, what changed and why, findings used
with URLs, **the oracles verbatim** (correctness commands and performance
metrics), an explicit statement that nothing was verified here, and:

Run this from the directory containing `fix.patch` — `-C <repo>` chdirs before
resolving the patch path, so a bare relative `fix.patch` would resolve under
`<repo>` instead:

```bash
git -C <repo> checkout <base-sha>
git -C <repo> apply --check "$PWD/fix.patch" && git -C <repo> apply "$PWD/fix.patch"
pytest tests/v1/worker/test_gpu_model_runner.py   # ← the recorded oracle
```

`git -C <repo> apply -3 "$PWD/fix.patch"` falls back to a three-way merge if
the patch does not apply cleanly; `patch -p1 < fix.patch` works without git,
run from the repo root. Recording the base commit is load-bearing: `git diff`
embeds no base, and applied to the wrong commit a patch either fails or
misapplies.

## Tests

- **`prompts.py`** — oracles appear verbatim; base SHA present; scope confined
  to the candidate's files; findings and proposals included. Reuses
  `prep_evolve` fixtures.
- **Validation ordering** — the gate runs against worktree content, not
  `--repo`; a dirty `--repo` does not affect the outcome.
- **Batch loop** — skip-with-reason on staleness, raise on a single explicit
  `--candidate`.
- **Collection** — an added file appears in the patch (the `git add -N`
  regression); worktree removed on the failure path.
- **`claude_exec`** — against a fake `claude` binary, following
  `tests/unit/agent_proposals/_fakes.py`.

The skill itself is untested, consistent with the existing bundled skills.

## Docs

- `docs/one-shot-fix.md` — the command, the skill, the artifacts, and an
  explicit statement of what is not verified.
- A README row alongside `prep-evolve`.
- A pointer from `docs/prep-evolve.md` framing this as the cheap arm: a proposal
  faithfully implemented, with no fitness loop.

## Decisions and their reasons

| Decision | Reason |
|---|---|
| Not a fourth `--evolver` | It is not an evolver — no population, no fitness function. Bolting it onto the adapter registry would misrepresent it. |
| Engine stage is the product; skill is the front door | The stage is what scales to a sweep and what carries cost accounting; the skill is what you reach for on one candidate. |
| No `fix-context` subcommand | Collapsed into `fix --print-prompt`. One command, one prompt. |
| `--print-prompt` also creates and validates the worktree | Otherwise the skill would reimplement the staleness gate in markdown, or validate against `--repo` — the mismatch this design rejects. Leaves only collection duplicated. |
| No tests, no benchmarks | The machine may lack the hardware; the oracle is an unvalidated prose regex; a fresh worktree has no build. |
| Oracles carried verbatim | They are Spotlights' candidate-admission bar, and the verification recipe for hardware that can run them. |
| Staleness gate kept | With no tests, editing the right lines is the only correctness property left. |
| Validate inside the worktree | Makes validated bytes identical to edited bytes, and allows a dirty checkout. |
| Patch only, repo untouched | The run directory is what Spotlights owns; a patch plus its notes is a self-contained handoff to the machine that can test it. |
| Throwaway `--detach` worktree | Your checkout is never modified and a dirty tree is irrelevant. Same pattern CORAL uses per agent. |
| No plan approval | Keeps the stage and the skill behaviorally identical, and the patch is reviewable before anything is applied. |
| Prompt shared, mechanics in Python | Prevents drift on the part that matters, keeps git plumbing testable, and preserves a genuinely steerable skill. |
