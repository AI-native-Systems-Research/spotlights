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
| `--print-prompt` | Create and validate the worktree, print it and the prompt, exit. **Requires `--candidate`.** |

## Artifacts

```
<base>/apply/<module>/<candidate-id>/
├── apply.patch        # git diff against the base commit, SHA in a header comment
├── apply.prompt.txt   # the prompt the agent was given, verbatim
└── APPLY-NOTES.md     # the travelling documentation
```

`apply.patch` is a `git diff`, not `format-patch`: the latter needs a commit, and
the repo stays untouched.

When the agent makes no in-scope edit — a legitimate outcome — the patch is
absent but `APPLY-NOTES.md` and `apply.prompt.txt` are still written, and a stale
patch left over from an earlier run of the same candidate is deleted, so the
directory can never hold a patch that the notes go on to deny exists.

`apply.prompt.txt` is the exact block `--print-prompt` emits — candidate, module,
base commit, repo, worktree paths, then the prompt body — written on every path,
patch or no patch. It is meant to be reusable on its own: take the file, and the
prompt is what a fresh agent needs to attempt the same change. It also answers
the question the notes cannot: whether a disappointing outcome came from the agent
or from what the agent was *told*. "No in-scope edit was possible" and "the target
the prompt named was the wrong one" read the same in the notes and differently
here. Both `--print-prompt` and the run path render it from the same function, so
the file the `/spotlights-apply-candidate` skill saves by teeing that stdout is
byte-identical to the one the CLI writes itself.

The two worktree paths in it are **dead by the time you read it** — the run
deletes the worktree before writing the artifact, and the skill deletes it at its
own last step. They are kept as a record that correlates the artifact with the
run, and the file's `NOTE:` block says so, along with the one command that turns
it back into something runnable:

```bash
git -C <repo> worktree add --detach <dir> <base-sha>
```

That is what makes the file reusable rather than merely readable: the prompt body
names the old worktree as its working directory, so a reader needs somewhere to
put a live one. Dropping the two fields instead would have left that body line in
place with nothing to explain it.

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

## The interactive front door: `/spotlights-apply-candidate`

The stage runs a nested `claude -p` you cannot influence. When you want to be
in the loop on one candidate — steer it, interrupt it, ask why mid-change —
use the bundled skill instead (install it with `spotlights-engine init`):

```
/spotlights-apply-candidate
```

It picks a candidate, calls `spotlights-engine apply --print-prompt` to get the
prompt *and* a created, validated worktree in one call, does the work
in-session, then collects the same three artifacts and removes both the
worktree and its `WORKTREE_PARENT` scaffolding directory. The prompt is shared
between the two paths, so they cannot drift on the part that matters.

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
> same prompt unattended. Use `--print-prompt` if you want to inspect the prompt
> and the validated worktree before deciding which way to go — it leaves the
> worktree in place for you.

## Sharing the patch

`/spotlights-share-candidates` folds the `apply/` tree into its offline bundle
automatically — no flag. Each exported candidate that has an `apply.patch` gets
a page with the notes rendered, the patch as a collapsible colorized diff, an
apply recipe using your own checkout path, and an `apply.zip`; the ranked index
gains an `apply` badge and a link. Candidates whose directory holds only
`APPLY-NOTES.md` are skipped.

The artifacts are copied byte-for-byte, so the patch a recipient applies is
exactly the one written here — including its recorded base commit.
