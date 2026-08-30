---
description: Use when implementing ONE Spotlights candidate as a reviewable patch in-session — "apply this candidate", "fix this candidate", "implement candidate cand-...", "one-shot apply". Runs `spotlights-engine apply --print-prompt` to get the validated apply prompt written to apply.prompt.txt, creates a throwaway worktree at the recorded base commit, does the work interactively, and collects apply.patch + APPLY-NOTES.md beside it. Runs no tests and no benchmarks, and never modifies the target repo.
---

# Apply Candidate

Turns one candidate from a finished Spotlights run into a reviewable patch,
implemented **in this session** so you can steer it, interrupt it, and ask why
mid-change. The batch equivalent is `spotlights-engine apply`, which runs a
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

2. **Get the prompt.** One command, and it does not need `--candidate` —
   omit it to write a prompt per candidate, or pass `--top-n N` for the top N:

   ```bash
   spotlights-engine apply --print-prompt \
     --result "<run-dir>" --repo "<repo>" --candidate "<cand-id>"
   ```

   This **clears the candidate's apply directory**: any `apply.patch`,
   `APPLY-NOTES.md` and `manifest.json` a previous apply of the same candidate
   left there is deleted, and the deleted patch cannot be recovered. If the user
   asked to see the prompt for a candidate that has already been applied, say so
   and let them decide before you run it — reading the existing
   `apply.prompt.txt` answers that question without destroying anything.

   It leaves `apply.prompt.txt` as the directory's only file and prints one line
   naming the directory:

   ```
   <cand-id>: <run-dir>/apply/<module-slug>/<cand-id> (1 file, prompt only)
   ```

   The prompt is **not** on stdout. Read it from the file:

   ```bash
   cat "<run-dir>/apply/<module-slug>/<cand-id>/apply.prompt.txt"
   ```

   Do not pipe this command through `tee` and do not save a copy of your own: the
   engine already wrote the file, in the exact form step 5 needs, and a
   second hand-made copy can only disagree with it. Equally, do not reconstruct
   the prompt later from what you remember reading — a paraphrase of the prompt is
   worse than no prompt, because it reads as the real one.

   The file holds a block of the form:

   ```
   CANDIDATE: <id>
   MODULE:    <qn>
   BASE:      <sha>
   REPO:      <repo>
   DIRTY:     <true|false>
   WORKTREE:  (none — create one; it is yours to remove when you are done)
     git -C <repo> worktree add --detach <dir> <sha>
   PROMPT:
   <the prompt body>
   ```

   Read `BASE` off its own line — it is the commit everything else hangs off:
   the worktree you create in step 3, the `git diff` in step 5, and the header
   that travels with the patch. `WORKTREE` deliberately holds prose rather than
   a path, so a script that used to read it as one fails loudly instead of
   `cd`-ing somewhere surprising.

   `DIRTY` says whether `<repo>` had uncommitted changes when the staleness gate
   read it. The gate reads the working tree, while you will work at `BASE`. If it
   says `true`, mention it to the user and carry on — it is information, not a
   blocker. If the candidate's own file was the dirty one, the line ranges in the
   prompt were validated against bytes that are not the bytes at `BASE`.

   If the command fails with a staleness error, stop. Tell the user the repo has
   drifted from the run and print the commit to check out — the base commit is
   in the run's `run_manifest.json` under `target.commit_sha`.
   Do not work around the gate; with no tests being run it is the only
   correctness check there is. Creating the worktree in the next step is not
   working around it: the gate has already run, in Python, before you saw
   this file.

3. **Create the worktree.** `--print-prompt` creates none — that is what makes
   it usable outside this skill — so make your own at `BASE`, mirroring the
   parent/child layout `spotlights-engine apply` itself uses so both paths
   produce the same shape:

   ```bash
   WT_PARENT="$(mktemp -d -t spotlights-apply-XXXXXX)"
   WORKTREE="$WT_PARENT/worktree"
   git -C "<repo>" worktree add --detach "$WT_PARENT/worktree" "<BASE>"
   ```

   Note both paths now — you remove them in step 6, and nothing else will.
   `--detach` matters: a named branch in the user's repo is a side effect this
   skill has no business leaving behind.

   The worktree is a fresh checkout at `BASE` with no virtualenv, no build
   artifacts, and no compiled extensions. It cannot run anything, which is one
   of the reasons step "What this does not do" gives for verifying nothing.

4. **Implement the change in the worktree**, following the prompt you just read.
   Edit only the in-scope files it lists — nothing outside that list, even if it
   looks like an obvious improvement. `cd` into `$WORKTREE` and do all editing
   there; never touch the user's own checkout of the repo, which may be dirty
   mid-work and is none of your business.

5. **Collect the artifacts.** `apply.patch` must carry a header naming the
   candidate, module, repo, and base commit — a bare `git diff` embeds none of
   that, and a patch that travels on its own without its base commit either
   fails to apply or misapplies silently. Write the header first, then append
   the diff, from the worktree:

   ```bash
   cd "$WORKTREE"
   git add -N .                 # REQUIRED: without it, files you ADDED vanish from the diff
   mkdir -p "<run-dir>/apply/<module-slug>/<cand-id>"
   cat > "<run-dir>/apply/<module-slug>/<cand-id>/apply.patch" <<'HEADER'
   # spotlights one-shot apply
   # candidate: <cand-id>
   # module:    <qn>
   # repo:      <repo>
   # base:      <BASE>
   # apply with (from the directory containing this patch):
   #   git -C <repo> checkout <BASE>
   #   git -C <repo> apply "$PWD/apply.patch"
   HEADER
   git diff "<BASE>" >> "<run-dir>/apply/<module-slug>/<cand-id>/apply.patch"
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
   so the literal text `"$PWD/apply.patch"` lands in the file. Fill in `<qn>`
   with the module's slash-form qualified name from `MODULE:` (not the slug),
   and `<repo>` / `<BASE>` from the same printed block. This is the exact
   header `spotlights-engine apply` itself writes, field for field, so both
   paths produce the same artifact.

   `<module-slug>` is the module's slash-form qualified name with every
   character outside `[A-Za-z0-9._-]` (including `/`) replaced by `_` (e.g.
   `v1/attention` → `v1_attention`).

   `apply.prompt.txt` needs nothing from you: step 2 already wrote it, in this
   same directory, and the `mkdir -p` above is a no-op on a directory that
   therefore already exists. Leave it exactly as it is — do not edit, trim,
   re-wrap, or regenerate it. It is the record of what the agent was *told*, and
   it is what separates "the proposal was declined" from "the instruction was
   wrong" when the notes alone cannot say which. That also means it survives the
   case where you produce no patch, which is where it matters most.

   Do check it is there before you finish — if step 2's directory and this one
   disagree, you have written the patch somewhere the prompt is not:

   ```bash
   ls "<run-dir>/apply/<module-slug>/<cand-id>/apply.prompt.txt"
   ```

   Then write `APPLY-NOTES.md` beside the patch containing:

   - candidate id, module, objective, and the **base commit** from `BASE:`
   - the in-scope files with their line ranges
   - what you changed and why
   - the findings you used, with their URLs
   - **the oracles verbatim**, correctness commands and performance metrics
   - an explicit statement that nothing was verified here
   - the apply-and-verify recipe below, run **from the directory `apply.patch` is
     in** (the same directory `APPLY-NOTES.md` sits in) — this is the exact recipe
     `spotlights-engine apply` itself writes, so both paths produce the same
     artifact:

     ```bash
     git -C <repo> checkout <BASE>
     git -C <repo> apply --check "$PWD/apply.patch" && git -C <repo> apply "$PWD/apply.patch"

     # the recorded correctness oracle(s) — run them on a machine that can:
     <every recorded correctness oracle, one command per line>
     ```

     Every one of them, not just the first: a candidate can record several,
     and the one you drop may be the suite covering the code you changed.

     If the patch does not apply cleanly, the fallback is
     `git -C <repo> apply -3 "$PWD/apply.patch"` (three-way merge), run from the
     same directory. Without git, `patch -p1 < apply.patch` works, run from the
     repo root instead.

     Do not write `git -C <repo> apply apply.patch` with a bare relative path —
     `-C` makes git chdir into `<repo>` first, so a bare `apply.patch` resolves
     under `<repo>`, not under the directory it actually lives in, and the
     apply fails with "can't open patch 'apply.patch'". Always pass the absolute
     `"$PWD/apply.patch"` (captured from the directory containing the patch,
     before the `-C` command runs).

   Recording the base commit is not optional: `git diff` embeds no base, and
   applied to the wrong commit the patch either fails or misapplies.

6. **Remove the worktree and its scaffolding directory** — always, including
   when you produced no patch. You created both in step 3, and nothing else
   will remove them:

   ```bash
   git -C "<repo>" worktree remove --force "$WORKTREE"
   git -C "<repo>" worktree prune
   rm -rf "$WT_PARENT"
   ```

   All three matter. Skipping the prune leaves a stale entry in
   `.git/worktrees`. Skipping the `rm -rf` leaves an empty
   `spotlights-apply-XXXX/` directory behind in the system temp directory on
   every single invocation — `worktree remove` deletes the worktree but not the
   parent scaffolding directory it lived in.

7. **Report** where the artifacts landed, whether a patch was produced, and the
   apply-and-verify commands.

## If the change cannot be made

Write `APPLY-NOTES.md` explaining why — the scope is wrong, the proposal needs a
file outside it, the research does not actually support the change — and
produce no patch. Still save `apply.prompt.txt`: a reader deciding whether to
believe the reason needs to see the instruction it was a reason about. A missing
patch is a fine outcome. A patch that cannot be trusted is not. There is no
plan-approval step in this skill; the patch itself is the reviewable artifact,
and nothing is applied until a human applies it.
