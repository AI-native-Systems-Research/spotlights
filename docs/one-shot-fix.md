# One-shot fix (`fix`)

`prep-evolve` is the expensive arm: it hands a candidate to an evolutionary
search, which needs a project-specific fitness function before it produces
anything meaningful. Many candidates don't warrant a search. They warrant *one
attempt*: read the candidate, read the research behind it, implement the
change, hand back something reviewable.

`spotlights-engine fix` is that arm. Per candidate it creates a throwaway
detached git worktree at the base commit, runs one `claude -p` session inside
it, and writes `fix.patch` plus `FIX-NOTES.md`. Your checkout is never
modified, and a dirty working tree is irrelevant.

← Back to [README](../README.md) · The other arm: [prep-evolve](prep-evolve.md)

## What is not verified

**Nothing.** The fix is not verified: there is no harness, no tests are run, no
workload, and no baseline.

The oracles still travel with the patch, verbatim, in `FIX-NOTES.md` — as the
verification recipe. The deliverable is *a proposal faithfully implemented and
documented*, not a measured win.

## Usage

```bash
# one candidate
spotlights-engine fix --result ./spotlights-out --repo ../vllm \
  --candidate cand-vllm_v1_kv_offload-0002

# the top 5 of a ranking (artifacts still land under the run dir)
spotlights-engine fix --result ./spotlights-out/sorted --repo ../vllm --top-n 5

# every candidate in the run
spotlights-engine fix --result ./spotlights-out --repo ../vllm
```

`--result` accepts any run artifact and self-locates the rest — a run
directory, a `result.json`, an `index.md`, a `sorted/` directory, or a
`sorted_candidates.{json,md}`.

`--repo` **must** point at a real git checkout. `fix` fails loudly otherwise: a
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
<base>/fix/<module>/<candidate-id>/
├── fix.patch        # git diff against the base commit, SHA in a header comment
└── FIX-NOTES.md     # the travelling documentation
```

`fix.patch` is a `git diff`, not `format-patch`: the latter needs a commit, and
the repo stays untouched.

When the agent makes no in-scope edit — a legitimate outcome — only
`FIX-NOTES.md` is written, saying why, and the directory ends up with **no**
`fix.patch`: a stale patch left over from an earlier run of the same candidate
is deleted, so the directory can never hold a patch that the notes go on to
deny exists.

`fix.patch` carries a header comment recording the candidate, module, repo,
and base commit, plus the apply command:

```bash
# spotlights one-shot fix
# candidate: cand-vllm_v1_kv_offload-0002
# module:    vllm/v1/kv_offload
# repo:      ../vllm
# base:      83ad767eed3be3ee7f2df63be693bfaca5c7c922
# apply with (from the directory containing this patch):
#   git -C ../vllm checkout 83ad767eed3be3ee7f2df63be693bfaca5c7c922
#   git -C ../vllm apply "$PWD/fix.patch"
```

`FIX-NOTES.md` records the candidate, module, objective, **base commit**,
in-scope files with line ranges, what changed and why, the findings used with
URLs, the oracles verbatim, an explicit no-verification statement, and the
full apply-and-verify recipe:

```bash
# run from the directory containing fix.patch — the same directory FIX-NOTES.md sits in
git -C <repo> checkout <base-sha>
git -C <repo> apply --check "$PWD/fix.patch" && git -C <repo> apply "$PWD/fix.patch"

# the recorded correctness oracle — run it on a machine that can:
pytest tests/v1/worker/test_gpu_model_runner.py
```

`-C <repo>` makes git chdir into `<repo>` *before* resolving the patch path,
so a bare relative `git -C <repo> apply fix.patch` fails with `error: can't
open patch 'fix.patch'` — it looks for `fix.patch` under `<repo>`, not under
the directory it actually lives in. Always pass the absolute `"$PWD/fix.patch"`
(captured from the directory holding the patch, before the `-C` command runs).

If the patch does not apply cleanly, `git -C <repo> apply -3 "$PWD/fix.patch"`
falls back to a three-way merge (same directory, same rule); `patch -p1 <
fix.patch` works without git, run from the repo root instead.

## The interactive front door: `/spotlights-fix-candidate`

The stage runs a nested `claude -p` you cannot influence. When you want to be
in the loop on one candidate — steer it, interrupt it, ask why mid-change —
use the bundled skill instead (install it with `spotlights-engine init`):

```
/spotlights-fix-candidate
```

It picks a candidate, calls `spotlights-engine fix --print-prompt` to get the
prompt *and* a created, validated worktree in one call, does the work
in-session, then collects the same two artifacts and removes both the
worktree and its `WORKTREE_PARENT` scaffolding directory. The prompt is shared
between the two paths, so they cannot drift on the part that matters.

### Naming the candidate up front

Choosing a candidate is the skill's **only** interactive step. Invoked bare, it
reads `<run>/sorted/sorted_candidates.json`, prints the ranked list, and asks
which one you want.

If you already know the id — from a module page, from `sorted_candidates.md`, or
because a `fix --top-n` sweep skipped it — name it in the invocation and the
picker never runs:

```
/spotlights-fix-candidate cand-vllm_v1_kv_offload-0002
```

The skill still needs the run directory and the target repo. It will ask for
those if the conversation hasn't already established them, so supplying all
three sends it straight to creating the worktree:

```
/spotlights-fix-candidate cand-vllm_v1_kv_offload-0002 --result ./spotlights-out --repo ../vllm
```

Everything after the command name is read by the skill, not parsed by argparse.
`--result`/`--repo` are borrowed from the CLI because they are familiar; plain
prose works identically:

```
/spotlights-fix-candidate implement cand-vllm_v1_kv_offload-0002 from ./spotlights-out against ../vllm
```

The id is the same value `spotlights-engine fix --candidate` takes, so you can
move a candidate between the two paths without translating anything.

> [!TIP]
> Passing a candidate to the **skill** is how you stay in the loop on one fix.
> Passing it to the **stage** (`spotlights-engine fix --candidate <id>`) runs the
> same prompt unattended. Use `--print-prompt` if you want to inspect the prompt
> and the validated worktree before deciding which way to go — it leaves the
> worktree in place for you.
