# One-shot apply (`apply`)

`prep-evolve` is the expensive arm: it hands a candidate to an evolutionary
search, which needs a project-specific fitness function before it produces
anything meaningful. Many candidates don't warrant a search. They warrant *one
attempt*: read the candidate, read the research behind it, implement the
change, hand back something reviewable.

`spotlights-engine apply` is that arm. Per candidate it creates a throwaway
detached git worktree at the base commit, runs one `claude -p` session inside
it, and writes `apply.patch`, `apply.prompt.txt`, and `APPLY-NOTES.md`. Your checkout is never
modified, and a dirty working tree is irrelevant.

← Back to [README](../README.md) · The other arm: [prep-evolve](prep-evolve.md)

## What is not verified

**Nothing.** The patch is not verified: there is no harness, no tests are run, no
workload, and no baseline.

The oracles still travel with the patch, verbatim, in `APPLY-NOTES.md` — as the
verification recipe. The deliverable is *a proposal faithfully implemented and
documented*, not a measured win.

## Usage

```bash
# one candidate
spotlights-engine apply --result ./spotlights-out --repo ../vllm \
  --candidate cand-vllm_v1_kv_offload-0002

# the top 5 of a ranking (artifacts still land under the run dir)
spotlights-engine apply --result ./spotlights-out/sorted --repo ../vllm --top-n 5

# every candidate in the run
spotlights-engine apply --result ./spotlights-out --repo ../vllm
```

`--result` accepts any run artifact and self-locates the rest — a run
directory, a `result.json`, an `index.md`, a `sorted/` directory, or a
`sorted_candidates.{json,md}`.

`--repo` **must** point at a real git checkout. `apply` fails loudly otherwise: a
worktree needs a commit, and a patch without a recorded base is not applicable.

| Flag | Purpose |
|---|---|
| `--result` | The finished run (any artifact of it). Required. |
| `--repo` | Target repo path. Must be a git checkout. Wins over `--index`. |
| `--index` | Rendered `index.md`; used only as a `--repo` fallback. |
| `--module` | Slash-form qualified name (optional; inferred from `--candidate`). |
| `--candidate` | One candidate id. Omit to sweep every candidate. |
| `--out` | Artifacts base dir. Default: the run directory. |
| `--direction` | `minimize` \| `maximize`; overrides the direction inferred from the objective. |
| `--top-n` | With a sorted `--result`: only the top N ranked candidates. Default `all`. |
| `--max-turns` | Agent turn cap per candidate. Default `40`. Must be `>= 1` (exit 2 otherwise). |
| `--wallclock` | Wall-clock cap in seconds per candidate. Default `1800`. Must be `>= 1` (exit 2 otherwise). |
| `--print-prompt` | Clear each selected candidate's apply dir, write `apply.prompt.txt` as its only file, print those directories, exit. Creates no worktree; runs no agent. |

## Artifacts

```
<base>/apply/<module>/<candidate-id>/
├── apply.patch        # git diff against the base commit, SHA in a header comment
├── apply.prompt.txt   # the prompt the agent was given, verbatim
├── APPLY-NOTES.md     # the travelling documentation
└── manifest.json      # this session's tokens, cost, provenance, timing
```

`--print-prompt` writes into these same directories — one per candidate it
selects — but leaves **only** `apply.prompt.txt` in each: no agent ran, so there
is no patch, nothing to document, and nothing to account for. Any of the other
three left by an earlier apply of that candidate is deleted, so each directory
describes the handoff in progress rather than the session it replaces.

`apply.patch` is a `git diff`, not `format-patch`: the latter needs a commit, and
the repo stays untouched.

When the agent makes no in-scope edit — a legitimate outcome — the patch is
absent but `APPLY-NOTES.md` and `apply.prompt.txt` are still written, and a stale
patch left over from an earlier run of the same candidate is deleted, so the
directory can never hold a patch that the notes go on to deny exists.

`apply.prompt.txt` records candidate, module, base commit and repo in a header,
then the prompt body — written on every path, patch or no patch. It is meant to
be reusable on its own: take the file, and the body is what a fresh agent needs
to attempt the same change. It also answers the question the notes cannot:
whether a disappointing outcome came from the agent or from what the agent was
*told*. "No in-scope edit was possible" and "the target the prompt named was the
wrong one" read the same in the notes and differently here.

The **body** carries no filesystem path, so it is byte-identical for a given
candidate no matter which path rendered it. The **header** is where the two
differ, and it says which one you are reading:

- From the run path, `WORKTREE:` and `WORKTREE_PARENT:` name the throwaway
  checkout the agent edited. They are **dead by the time you read them** — the
  run removes the worktree before writing this file — and the `NOTE:` block says
  so. They are kept because they correlate the artifact with the run's logs.
- From `--print-prompt`, there is no worktree to name. `WORKTREE:` holds prose
  instead of a path, and the header carries a `DIRTY:` line plus the one command
  that turns the base commit into a checkout you can work in:

```bash
git -C <repo> worktree add --detach <dir> <base-sha>
```

`DIRTY:` says whether `--repo` had uncommitted changes when the staleness gate
read it. The gate reads your working tree while you will work at the base commit,
so the two can differ — see [Inspecting the prompt](#inspecting-the-prompt).

`apply.patch` carries a header comment recording the candidate, module, repo,
and base commit, plus the apply command:

```bash
# spotlights one-shot apply
# candidate: cand-vllm_v1_kv_offload-0002
# module:    vllm/v1/kv_offload
# repo:      ../vllm
# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922
# apply with (from the directory containing this patch):
#   git -C ../vllm checkout 83ad767eed3be3ee7f2df63be693bfaca5c7c922
#   git -C ../vllm apply "$PWD/apply.patch"
```

`APPLY-NOTES.md` records the candidate, module, objective, **base commit**,
in-scope files with line ranges, what changed and why, the findings used with
URLs, the oracles verbatim, an explicit no-verification statement, and the
full apply-and-verify recipe:

```bash
# run from the directory containing apply.patch — the same directory APPLY-NOTES.md sits in
git -C <repo> checkout <base-sha>
git -C <repo> apply --check "$PWD/apply.patch" && git -C <repo> apply "$PWD/apply.patch"

# the recorded correctness oracle — run it on a machine that can:
pytest tests/v1/worker/test_gpu_model_runner.py
```

`-C <repo>` makes git chdir into `<repo>` *before* resolving the patch path,
so a bare relative `git -C <repo> apply apply.patch` fails with `error: can't
open patch 'apply.patch'` — it looks for `apply.patch` under `<repo>`, not under
the directory it actually lives in. Always pass the absolute `"$PWD/apply.patch"`
(captured from the directory holding the patch, before the `-C` command runs).

If the patch does not apply cleanly, `git -C <repo> apply -3 "$PWD/apply.patch"`
falls back to a three-way merge (same directory, same rule); `patch -p1 <
apply.patch` works without git, run from the repo root instead.

### `manifest.json`

What this candidate's apply cost. One `claude -p` session runs per candidate,
so this file accounts for 100% of the model spend behind the directory.

- `models_used` — the four disjoint token buckets (input, output, cache read,
  cache create), grouped per model.
- `total_tokens` — their sum, across every group.
- `cost` — priced through the contracted rate table. `external_cost` — the
  same tokens at public list price, which is the figure to quote externally.
  When `priced_token_share` is below 1.0 the dollar amount is partial and
  `coverage.unpriced_models` names what was left out.
- `target` / `spotlights` — the repo, commit, and objective this patch was
  produced against, plus the engine commit that produced it. Note that
  `target.commit_sha` is the repo's HEAD *when apply ran*, which may differ
  from the commit the originating run analyzed.
- `timing.wall_clock_s` / `api_time_s` — when the two are exactly equal, the
  stream reported no API duration and the wall clock stood in.
- `notes` — why anything above is incomplete.

Same field names and same blocks as `run_manifest.json` wherever the two
describe the same thing, so one reader parses both. Apply's spend is
deliberately **not** in `run_manifest.json`: apply runs after the run, against
a repo state the run never analyzed, and can run many times over one run's
candidates.

A session killed by `--wallclock` loses its usage totals — the timeout severs
the CLI's exit handshake after the work is already done — so the file is
written with `models_used: []`, zeroed costs, a real `wall_clock_s`, and the
reason in `notes`. Nothing changes name or disappears.

Summing a sweep, plus the companion line that says which manifests are
degraded — each of those contributes `0.0` to the sum and nothing in the sum
itself says so:

````bash
jq -s 'map(.cost.amount_usd) | add' apply/*/*/manifest.json
jq -r 'select(.notes != "") | "\(.candidate_id): \(.notes)"' apply/*/*/manifest.json
````

The interactive `/spotlights-apply-candidate` path writes three files, not
four: the agent *is* the session there, so there is no `claude -p` stream to
parse and no usage totals to report.

## The interactive front door: `/spotlights-apply-candidate`

The stage runs a nested `claude -p` you cannot influence. When you want to be
in the loop on one candidate — steer it, interrupt it, ask why mid-change —
use the bundled skill instead (install it with `spotlights-engine init`):

```
/spotlights-apply-candidate
```

It picks a candidate, calls `spotlights-engine apply --print-prompt` to get the
validated prompt, reads it from the written `apply.prompt.txt`, creates a
throwaway worktree at the recorded base commit, does the work in-session, then
collects the remaining artifacts and removes both the worktree and its temp-dir
parent. The prompt is shared between the two paths, so they cannot drift on the
part that matters — and the skill no longer has to save a copy of it, because
the engine already wrote the file.

### Naming the candidate up front

Choosing a candidate is the skill's **only** interactive step. Invoked bare, it
reads `<run>/sorted/sorted_candidates.json`, prints the ranked list, and asks
which one you want.

If you already know the id — from a module page, from `sorted_candidates.md`, or
because an `apply --top-n` sweep skipped it — name it in the invocation and the
picker never runs:

```
/spotlights-apply-candidate cand-vllm_v1_kv_offload-0002
```

The skill still needs the run directory and the target repo. It will ask for
those if the conversation hasn't already established them, so supplying all
three sends it straight to creating the worktree:

```
/spotlights-apply-candidate cand-vllm_v1_kv_offload-0002 --result ./spotlights-out --repo ../vllm
```

Everything after the command name is read by the skill, not parsed by argparse.
`--result`/`--repo` are borrowed from the CLI because they are familiar; plain
prose works identically:

```
/spotlights-apply-candidate implement cand-vllm_v1_kv_offload-0002 from ./spotlights-out against ../vllm
```

The id is the same value `spotlights-engine apply --candidate` takes, so you can
move a candidate between the two paths without translating anything.

> [!TIP]
> Passing a candidate to the **skill** is how you stay in the loop on one candidate.
> Passing it to the **stage** (`spotlights-engine apply --candidate <id>`) runs the
> same prompt unattended. Use `--print-prompt` if you want to read the prompt and
> check its scope bound before deciding which way to go — it runs no agent, costs
> nothing, and creates nothing you have to clean up.

## Inspecting the prompt

`--print-prompt` does the resolution and the validation — locate the run, resolve
the repo, run the staleness gate against `--repo`, render the prompt — and then
stops. It runs no agent, spends nothing, and creates no worktree, so there is
nothing to clean up afterwards. It takes the same selectors the run path does:

```console
$ spotlights-engine apply --print-prompt \
    --result ./spotlights-out --repo ../vllm --candidate cand-vllm_v1_kv_offload-0002
cand-vllm_v1_kv_offload-0002: spotlights-out/apply/vllm_v1_kv_offload/cand-vllm_v1_kv_offload-0002 (1 file, prompt only)

$ spotlights-engine apply --print-prompt --result ./spotlights-out/sorted --repo ../vllm --top-n 3
```

The prompt is **not** printed to stdout — it is in `apply.prompt.txt` in each
printed directory, and stdout stays one machine-readable line per candidate.
stderr closes with the same tally `prep-evolve` prints, counting prompts:

```
apply: 3 prompt(s) written, 0 skipped
```

A stale candidate in a sweep is skipped with a reason rather than aborting the
rest, and the exit code is still `0` when anything was written — so read the
tally or the `skipped` lines, not `$?`.

Read the prompt, and the commit to work from, out of the file:

```bash
cat "<dir>/apply.prompt.txt"                # the whole block
grep '^BASE:' "<dir>/apply.prompt.txt"      # the commit to check out
grep '^DIRTY:' "<dir>/apply.prompt.txt"     # was --repo clean when the gate ran?
```

To actually implement the change, make your own checkout at `BASE` — the header
gives you the command — and work in that. Your own checkout is never modified
and may be dirty throughout.

Two consequences worth knowing:

- **The gate reads your working tree, not the base commit.** Uncommitted work
  does not refuse the run (that is `prep-evolve`'s behaviour too, and refusing
  would make the command unusable mid-task), and `DIRTY:` records it. But if the
  dirty file is the candidate's own, the line ranges in the prompt were validated
  against bytes that are not the bytes at `BASE`. Local edits that gut the symbol
  give a loud `StalenessError`; local edits that *put the symbol where the
  candidate says it is* pass quietly and point the scope bound at the wrong code.
  `git stash` first if you want the gate to read exactly what you will edit.
- **A directory holding only `apply.prompt.txt` is a prompt handoff, not a
  failed apply.** On the run path the artifacts are written together, so a lone
  prompt file used to mean a half-finished write; `--print-prompt` produces that
  state deliberately.
- **It clears each selected candidate's apply directory.** See below.

> [!WARNING]
> `--print-prompt` **deletes** any `apply.patch`, `APPLY-NOTES.md` and
> `manifest.json` an earlier apply of the same candidate left in that directory,
> so that only the new `apply.prompt.txt` remains. It is how a candidate gets
> restarted: those files describe a session the handoff supersedes, and a patch
> sitting next to a prompt that did not produce it is worse than no patch. The
> deleted patch is **not recoverable** — the worktree that produced it is long
> gone. Copy the directory elsewhere first if you want to keep it, or read the
> prompt straight out of the existing `apply.prompt.txt` instead, which the run
> path already wrote.
> In a sweep this happens for **every candidate the invocation selects** — all of
> them with no `--candidate`, or the top N with `--top-n N` — so check what
> `--top-n` covers before restarting a run whose earlier patches you want.

## Sharing the patch

`/spotlights-share-candidates` folds the `apply/` tree into its offline bundle
automatically — no flag. Each exported candidate that has an `apply.patch` gets
a page with the notes rendered, the patch as a collapsible colorized diff, an
apply recipe using your own checkout path, and an `apply.zip`; the ranked index
gains an `apply` badge and a link. Candidates whose directory holds only
`APPLY-NOTES.md` are skipped.

The artifacts are copied byte-for-byte, so the patch a recipient applies is
exactly the one written here — including its recorded base commit.
