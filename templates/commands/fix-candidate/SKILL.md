---
name: fix-candidate
description: Use when implementing ONE Spotlights candidate as a reviewable patch in-session — "fix this candidate", "implement candidate cand-...", "one-shot fix". Runs `spotlights-engine fix --print-prompt` to get a validated throwaway worktree plus the fix prompt, does the work interactively, and collects fix.patch + FIX-NOTES.md. Runs no tests and no benchmarks, and never modifies the target repo.
---

# Fix Candidate

Turns one candidate from a finished Spotlights run into a reviewable patch,
implemented **in this session** so you can steer it, interrupt it, and ask why
mid-change. The batch equivalent is `spotlights-engine fix`, which runs a
nested `claude -p` you cannot influence — reach for this skill when you want to
be in the loop on one candidate.

## What this does not do

**No verification happens here.** No tests are run, no benchmarks are measured,
no build is attempted. Three reasons, all of them load-bearing:

- The machine may lack the hardware. A recorded `performance_oracle: TTFT, TPOT`
  is a metric *name* — no harness, no workload, no baseline. A fabricated
  number is worse than none.
- The recorded correctness oracle is a best-effort regex over the candidate's
  LLM-written rationale. `pytest tests/…` reads as authoritative and is a regex
  hit on a sentence; nothing has checked that the file exists.
- The worktree is a fresh detached checkout with no virtualenv, no build
  artifacts, and no compiled extensions. For a repo like vLLM it cannot execute
  anything.

The oracles still travel with the patch, verbatim, as the verification recipe
for whoever has the hardware. Never claim a result you did not measure. Do not
run the oracle commands yourself, even "just to see" — a result you produced on
an unvalidated worktree is exactly the fabricated number this section exists to
prevent.

## Procedure

1. **Resolve the run, repo, and candidate.**
   - Ask for the run directory (the folder holding `result.json`) and the target
     repo path if they are not already obvious from the conversation.
   - If the user did not name a candidate, read `<run>/sorted/sorted_candidates.json`
     (or `sorted_candidates.md`) and show the ranked list, then ask which one.
     **This is the only interactive step before the work starts.**

2. **Get the prompt and a validated worktree** — one command does both.
   `--candidate` is required here (without one, `--print-prompt` refuses to run,
   to avoid leaking one worktree per candidate in a sweep):

   ```bash
   spotlights-engine fix --print-prompt \
     --result "<run-dir>" --repo "<repo>" --candidate "<cand-id>"
   ```

   It prints a block of the form:

   ```
   CANDIDATE: <id>
   MODULE:    <qn>
   BASE:      <sha>
   WORKTREE:  <path>
   WORKTREE_PARENT:  <path>
   PROMPT:
   <the prompt body>
   ```

   `WORKTREE` is a detached checkout at `BASE`, already validated against the
   candidate's recorded symbol and line range, and is **left in place for you**.
   `WORKTREE_PARENT` is the `mkdtemp` scaffolding directory that contains it —
   note it now, you need it for cleanup in step 5.

   If the command fails with a staleness error, stop. Tell the user the repo has
   drifted from the run and print the commit to check out — the base commit is
   in the run's `run_manifest.json` under `target.commit_sha`. Do not work around
   the gate; with no tests being run it is the only correctness check there is.

3. **Implement the change in the worktree**, following the printed prompt.
   Edit only the in-scope files it lists — nothing outside that list, even if it
   looks like an obvious improvement. `cd` into `WORKTREE` and do all editing
   there; never touch the user's own checkout of the repo, which may be dirty
   mid-work and is none of your business.

4. **Collect the artifacts.** `fix.patch` must carry a header naming the
   candidate, module, repo, and base commit — a bare `git diff` embeds none of
   that, and a patch that travels on its own without its base commit either
   fails to apply or misapplies silently. Write the header first, then append
   the diff, from the worktree:

   ```bash
   cd "<WORKTREE>"
   git add -N .                 # REQUIRED: without it, files you ADDED vanish from the diff
   mkdir -p "<run-dir>/fix/<module-slug>/<cand-id>"
   cat > "<run-dir>/fix/<module-slug>/<cand-id>/fix.patch" <<'HEADER'
   # spotlights one-shot fix
   # candidate: <cand-id>
   # module:    <qn>
   # repo:      <repo>
   # base:      <BASE>
   # apply with (from the directory containing this patch):
   #   git -C <repo> checkout <BASE>
   #   git -C <repo> apply "$PWD/fix.patch"
   HEADER
   git diff "<BASE>" >> "<run-dir>/fix/<module-slug>/<cand-id>/fix.patch"
   ```

   Diff against `<BASE>`, never a bare `git diff`. `git add -N .`'s `.`
   pathspec does not merely intent-to-add new paths: for a path that no longer
   exists on disk it stages the *deletion in full*, so a bare index-vs-worktree
   `git diff` has nothing left to report for a file you deleted — or for the
   delete-half of a rename — and that half silently vanishes from the patch.
   `git diff <commit>` compares the working tree against the commit regardless
   of what got staged, so it sees both, and matches the modified/added cases
   byte-for-byte. The `git add -N .` is still required: `git diff <commit>`
   does not surface untracked files on its own.

   Use the quoted `<<'HEADER'` heredoc exactly as shown — quoting the
   delimiter stops the shell from expanding `$PWD` while writing the header,
   so the literal text `"$PWD/fix.patch"` lands in the file. Fill in `<qn>`
   with the module's slash-form qualified name from `MODULE:` (not the slug),
   and `<repo>` / `<BASE>` from the same printed block. This is the exact
   header `spotlights-engine fix` itself writes, field for field, so both
   paths produce the same artifact.

   `<module-slug>` is the module's slash-form qualified name with every
   character outside `[A-Za-z0-9._-]` (including `/`) replaced by `_` (e.g.
   `v1/attention` → `v1_attention`). Then write `FIX-NOTES.md` beside the
   patch containing:

   - candidate id, module, objective, and the **base commit** from `BASE:`
   - the in-scope files with their line ranges
   - what you changed and why
   - the findings you used, with their URLs
   - **the oracles verbatim**, correctness commands and performance metrics
   - an explicit statement that nothing was verified here
   - the apply-and-verify recipe below, run **from the directory `fix.patch` is
     in** (the same directory `FIX-NOTES.md` sits in) — this is the exact recipe
     `spotlights-engine fix` itself writes, so both paths produce the same
     artifact:

     ```bash
     git -C <repo> checkout <BASE>
     git -C <repo> apply --check "$PWD/fix.patch" && git -C <repo> apply "$PWD/fix.patch"

     # the recorded correctness oracle(s) — run them on a machine that can:
     <every recorded correctness oracle, one command per line>
     ```

     Every one of them, not just the first: a candidate can record several,
     and the one you drop may be the suite covering the code you changed.

     If the patch does not apply cleanly, the fallback is
     `git -C <repo> apply -3 "$PWD/fix.patch"` (three-way merge), run from the
     same directory. Without git, `patch -p1 < fix.patch` works, run from the
     repo root instead.

     Do not write `git -C <repo> apply fix.patch` with a bare relative path —
     `-C` makes git chdir into `<repo>` first, so a bare `fix.patch` resolves
     under `<repo>`, not under the directory it actually lives in, and the
     apply fails with "can't open patch 'fix.patch'". Always pass the absolute
     `"$PWD/fix.patch"` (captured from the directory containing the patch,
     before the `-C` command runs).

   Recording the base commit is not optional: `git diff` embeds no base, and
   applied to the wrong commit the patch either fails or misapplies.

5. **Remove the worktree and its scaffolding directory** — always, including
   when you produced no patch:

   ```bash
   git -C "<repo>" worktree remove --force "<WORKTREE>"
   git -C "<repo>" worktree prune
   rm -rf "<WORKTREE_PARENT>"
   ```

   All three matter. Skipping the prune leaves a stale entry in
   `.git/worktrees`. Skipping the `rm -rf` leaves an empty
   `spotlights-fix-XXXX/` directory behind in the system temp directory on
   every single invocation — `worktree remove` deletes the worktree but not the
   parent scaffolding directory it lived in.

6. **Report** where the artifacts landed, whether a patch was produced, and the
   apply-and-verify commands.

## If the change cannot be made

Write `FIX-NOTES.md` explaining why — the scope is wrong, the proposal needs a
file outside it, the research does not actually support the change — and
produce no patch. A missing patch is a fine outcome. A patch that cannot be
trusted is not. There is no plan-approval step in this skill; the patch itself
is the reviewable artifact, and nothing is applied until a human applies it.
