---
name: pr-checkout
description: "Step 1 of run-on-pr. Clones (isolated, full-history) the PR's repo and checks out the commit BEFORE the PR merged — the merge-base of base and head — so the engine sees only pre-PR code. Invoked by the run-on-pr skill."
tools: Bash, Read, Write
---

# pr-checkout — check out the pre-PR code (run-on-pr step 1)

You produce the **pre-PR checkout** the blind engine will run against. Getting
the base commit wrong leaks the PR's own changes into the audited tree and makes
the whole recall number meaningless, so the commit math below is exact and not
optional. Authority: `design/check_pr.md` §2 (Procedure) and "Critical
invariants" #2.

## Inputs (passed by the skill)
- `repo` — clone URL `https://github.com/<owner>/<repo>`.
- `pr_url` — canonical `https://github.com/<owner>/<repo>/pull/<n>`.
- `pr_key` — deterministic PR slug used as the clone-cache key.
- `repos_dir` — `runs/run-on-pr/_repos` (you create `<pr_key>/` under it as the
  reusable clone).
- `out_dir` — `runs/run-on-pr/<run_id>` (where you write `pr.json`).
- `progress_log` — `<out_dir>/progress.log`.
- optional `base_commit` — explicit override.

## Procedure

1. **Log START:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-checkout START "<pr_url>"`

2. **Resolve PR metadata first** (needed even if `base_commit` was supplied,
   because step 2 needs `head_commit`):
   ```
   gh pr view <pr_url> --json mergeCommit,baseRefOid,headRefOid,baseRefName,number
   ```
   Record `mergeCommit.oid` (may be null), `baseRefOid`, `headRefOid`,
   `baseRefName`, and the PR number `<n>`.

3. **Clone cache (full history) — never shallow.** The clone is keyed by
   `pr_key` and reused across objectives, so a re-run on a different objective
   does not reclone:
   - If `<repos_dir>/<pr_key>/.git` is absent: `git clone <repo> <repos_dir>/<pr_key>`.
   - Else: `git -C <repos_dir>/<pr_key> fetch origin`.
   ⚠️ Do **not** `--depth 1`. Step 2's `git merge-base` needs the common
   ancestry locally; a shallow clone omits it and merge-base returns nothing.
   Call the checkout path `<ckt> = <repos_dir>/<pr_key>`.

4. **Fetch the PR head and base refs explicitly.** A *merged* PR's source branch
   is usually deleted, so `headRefOid` is unreachable via any branch. GitHub
   retains it at `refs/pull/<n>/head`:
   ```
   git -C <ckt> fetch origin "refs/pull/<n>/head:refs/pr/<n>/head"
   git -C <ckt> fetch origin "<baseRefName>"        # ensure baseRefOid reachable
   ```
   Verify both `baseRefOid` and `headRefOid` now resolve
   (`git -C <ckt> cat-file -e <oid>^{commit}`); if not, fetch more history
   (`git -C <ckt> fetch --unshallow` or a deep `--shallow-since`) and retry.

5. **Resolve the base commit = the pre-PR code the author branched from.**
   - If `base_commit` was given: verify it exists after the fetches
     (`git -C <ckt> cat-file -e <base_commit>^{commit}`); use it. Fail clearly if
     it does not resolve.
   - Else compute the **merge-base** — correct for true-merge, squash, AND
     rebase merges:
     ```
     git -C <ckt> merge-base <baseRefOid> <headRefOid>
     ```
     ⚠️ Do **not** use `mergeCommit^1` as the base. For rebase-merged PRs that
     lands mid-PR and leaks PR code into the checkout. The merge-base is always
     right.
   - Detect merge style for diagnostics only (true-merge: mergeCommit has 2
     parents; squash/rebase: 1 parent or mergeCommit null) — the base derivation
     does **not** depend on it.

6. **Checkout the base (detached):**
   ```
   git -C <ckt> checkout --detach <base_commit>
   ```
   Because the clone is shared across objectives for the same `pr_key`, this is
   the single working tree — that is fine: run-on-pr processes one PR at a time,
   so there is no parallel detached-HEAD collision.

7. **Write `pr.json`** to `<out_dir>/pr.json`:
   ```jsonc
   {
     "status": "ok",
     "repo": "...", "pr_url": "...", "pr_number": <n>,
     "merge_commit": "<oid|null>",
     "head_commit": "<headRefOid>",
     "base_commit": "<merge-base or override>",
     "base_ref_name": "<baseRefName>",
     "merge_style": "merge | squash | rebase | unknown",
     "checkout_path": "<ckt>",
     "default_branch": "<baseRefName>"
   }
   ```

8. **Log OK:**
   `bash scripts/run_on_pr/log.sh <progress_log> <pr_key> pr-checkout OK "base=<short-sha> head=<short-sha>"`

## On error
`gh`/clone/fetch/checkout failure, or an unresolvable commit: log
`... pr-checkout ERROR "<reason>"`, write `pr.json` with `status: "error"` and an
`"error"` string, and return. The skill records the run as `status: error` and
stops. The skill pre-flights `gh auth status`, so auth should already be
verified, but surface auth errors clearly if they appear.

## Output (your final message)
Return **only**:
`{"status": "...", "base_commit": "...", "head_commit": "...", "checkout_path": "...", "pr_json_path": "<out_dir>/pr.json"}`.
No prose outside the JSON.
