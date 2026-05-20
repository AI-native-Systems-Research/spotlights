# Modules-Map Pipeline — High-Level Plan

A small orchestrator that produces a verified architectural map (`modules.json`) for an arbitrary target repository by chaining a **bootstrap** run of Claude Code with **alternating review iterations** by Claude Code and Codex.

Inspired by CORAL's pattern of spawning coding-agent subprocesses against a working directory, but **much narrower in scope**: no graders, no shared hub, no eval loop. Just sequential subprocess runs that read/write files in a run directory.

---

## 1. Goal

Given a target repository path, produce `modules.json` — a structured architectural map that conforms to the schema defined in [modules.txt](modules.txt).

Quality is improved iteratively:
1. **Bootstrap** — Claude Code drafts the first `modules.json` from `modules.txt`.
2. **Review loop** — Claude Code and Codex alternate, each running the audit prompt in [modules_review.txt](modules_review.txt) against the latest JSON, producing a corrected JSON plus a `<changes>` block.
3. **Stop** on a precisely-defined convergence condition, cycle detection, or a max-iteration cap.

---

## 2. Inputs & Outputs

**Inputs**
- `target_repo` — absolute path to the repo being analyzed.
- `bootstrap_prompt` — [modules.txt](modules.txt) (verbatim).
- `review_prompt_template` — [modules_review.txt](modules_review.txt) (with `PASTE_JSON_HERE` replaced each iteration).
- `max_review_iterations` (default: 4 review iters, so 5 total runs incl. bootstrap).
- `agents` — ordered list of runtimes for the review loop, e.g. `["claude_code", "codex", "claude_code", "codex"]`.
- `budget_usd` (optional) — soft ceiling; orchestrator stops before exceeding.

**Run directory layout**

Run artifacts live **outside the target repo** by default — at `~/.cache/modules_pipeline/<repo_slug>/<run_id>/` — so normal outputs cannot pollute the analyzed repo's git state or be accidentally committed. `<run_id>` is `YYYYMMDD-HHMMSS-<6 random hex>` (sortable, unique, human-readable). `<repo_slug>` is the basename of `target_repo` plus a short hash of its absolute path.

```
~/.cache/modules_pipeline/<repo_slug>/<run_id>/
  iter_0_bootstrap/
    modules.json           # validated JSON from claude_code
    prompt.txt             # the exact prompt sent (wrapped, see §5.1)
    validation.json         # parse/shape/path audit
    raw_stdout.log
    raw_stderr.log
    cost.json              # tokens in/out, $ estimate, wallclock
  iter_1_claude_code/
    modules.json
    changes.txt            # the <changes> block written by the reviewer
    diff_from_prev.md      # human-readable diff vs iter_0
    prompt.txt
    validation.json
    raw_stdout.log
    raw_stderr.log
    cost.json
  iter_2_codex/...
  final/
    modules.json           # copy of the last accepted JSON
  run.json                 # metadata: see §2.1
```

### 2.1 `run.json` schema

```json
{
  "run_id": "20260512-143022-a1b2c3",
  "repo_path": "/abs/path/to/target",
  "repo_slug": "CORAL-3f9a2b",
  "started_at": "...",
  "finished_at": "...",
  "iterations": [
    { "n": 0, "runtime": "claude_code", "model": "opus",
      "valid_json": true, "hallucinated_paths": 3, "module_count": 12,
      "tokens_in": 8421, "tokens_out": 2103, "cost_usd": 0.18, "wallclock_s": 47 },
    { "n": 1, "runtime": "claude_code", "model": "opus",
      "valid_json": true, "hallucinated_paths": 0, "module_count": 11,
      "restructure": false, "fix_count": 5, "byte_equal_to_prev": false,
      "tokens_in": 12044, "tokens_out": 2310, "cost_usd": 0.24, "wallclock_s": 51 }
  ],
  "stop_reason": "converged | cycle_detected | max_review_iterations | budget_exceeded | hard_failure",
  "final_iter": 3,
  "total_cost_usd": 0.94
}
```

---

## 3. Architecture (one diagram)

```
                ┌──────────────────────────┐
                │ orchestrator (Python)    │
                │  - parses CLI args       │
                │  - manages run dir       │
                │  - decides stop          │
                │  - tracks cost           │
                └──────────┬───────────────┘
                           │ spawn subprocess (cwd = target_repo)
                           ▼
   ┌──────────────────────────────────────────────────┐
   │ Iter 0: claude_code   <modules.txt>              │  → modules.json
   ├──────────────────────────────────────────────────┤
   │ Iter 1: claude_code   <modules_review.txt + JSON>│  → modules.json'
   ├──────────────────────────────────────────────────┤
   │ Iter 2: codex         <modules_review.txt + JSON>│  → modules.json''
   ├──────────────────────────────────────────────────┤
   │ Iter 3: claude_code   <modules_review.txt + JSON>│  → ...
   └──────────────────────────────────────────────────┘
                           │
                           ▼
                    final/modules.json
```

The target repo is the agents' `cwd` (so they can read source freely) but intended writes go to the run dir, **not** the target repo. The runtime wrapper must grant the iteration directory as an allowed write/read root when the CLI requires it (for example `--add-dir <iter_dir>`).

---

## 4. CORAL: what to lift, what to skip

CORAL solves a superset of this problem. We need a thin slice:

| CORAL piece | Reuse? | What to take |
|---|---|---|
| [`coral/agent/registry.py`](../coral/agent/registry.py) | **Pattern** | Name → runtime-class dict + alias table. |
| [`coral/agent/builtin/claude_code.py`](../coral/agent/builtin/claude_code.py) | **Code shape** | `subprocess.Popen` with `--print`, `--model`, `--permission-mode acceptEdits`, `--add-dir`, stream-json output; the env-cleaning call `_clean_env()` from [`coral/workspace/repo.py`](../coral/workspace/repo.py). |
| [`coral/agent/builtin/codex.py`](../coral/agent/builtin/codex.py) | **Code shape** | `codex exec` invocation pattern: model selection, runtime `-c` options, JSONL output, and session-id extraction. Do **not** copy CORAL's `--dangerously-bypass-approvals-and-sandbox` default unless the caller explicitly points the pipeline at a disposable clone. |
| Worktrees ([`coral/workspace/worktree.py`](../coral/workspace/worktree.py)) | **Skip** | One target repo, agents are read-mostly. |
| Hub / attempts / notes (`coral/hub/`) | **Skip** | No shared state between iterations beyond files. |
| Graders (`coral/grader/`) | **Skip** | We do our own validation in §5.4. |
| Heartbeats, session resume, max-turns | **Skip in v1** | Each iter is a fresh process. (Session resume is a possible future optimization — see §11.) |

Concrete imports the new orchestrator can borrow:
- `coral.workspace.repo._clean_env` — strips env vars that would confuse a nested agent.
- The Popen + log-tailing thread pattern from [`coral/agent/builtin/claude_code.py`](../coral/agent/builtin/claude_code.py).

---

## 5. Iteration mechanics

### 5.1 Prompt wrapper contract

The orchestrator does **not** send `modules.txt` or `modules_review.txt` verbatim to the agent. It wraps them with a small preamble that pins down the I/O contract:

```
<preamble>
You are running as a non-interactive subprocess inside a one-shot pipeline.
- Your current working directory is the repository to analyze.
- Do NOT modify the repository — only read it.
- This wrapper overrides the Output section in the prompt below where they conflict.
- Write valid JSON only, with no markdown fence, to: <ITER_DIR>/modules.json
- For review iterations only: write the complete <changes> block, and nothing else, to: <ITER_DIR>/changes.txt
- Make your final assistant message exactly "DONE" when all required files are written. No commentary.
</preamble>

<the verbatim prompt from modules.txt or modules_review.txt>
```

For the review prompt, the orchestrator also replaces the literal string `PASTE_JSON_HERE` in `modules_review.txt` with the previous iteration's JSON content before wrapping.

This makes the agent → orchestrator contract explicit: **the JSON and review changes live in files at known paths**. The final assistant message is only a status signal; raw stdout may still be JSON/JSONL because both CLIs support structured output modes. This also neutralizes the review prompt's default "return a fenced block" instruction, which is useful for humans but brittle for subprocess orchestration.

### 5.2 Bootstrap (iter 0)

1. Create run dir + `iter_0_bootstrap/`.
2. Wrap `design/modules.txt`; write the wrapped prompt to `iter_0_bootstrap/prompt.txt`.
3. Spawn `claude_code` with cwd = `target_repo`, prompt = wrapped text, and `iter_0_bootstrap/modules.json` as the required output path.
4. Wait for exit; capture stdout/stderr/timing/tokens.
5. Validate (§5.4). If invalid → retry **once** with a fresh wrapped prompt that says "your previous file failed validation: <reason>; rewrite `<ITER_DIR>/modules.json` and finish with DONE"; if still invalid → abort with `stop_reason: hard_failure`.

### 5.3 Review iteration N (N ≥ 1)

1. Read the previous accepted JSON (`iter_{N-1}/modules.json`).
2. Splice it into `design/modules_review.txt` (replace `PASTE_JSON_HERE`).
3. Wrap with the §5.1 preamble; output paths = `iter_N_<runtime>/modules.json` and `iter_N_<runtime>/changes.txt`.
4. Pick this iteration's runtime from `agents[(N-1) % len(agents)]`.
5. Spawn that runtime; wait; capture.
6. Validate (§5.4) and require non-empty `changes.txt`. On validation or protocol failure, retry once with a fresh wrapped prompt that explains the failure. If the retry still fails, mark the iteration as failed and keep prev JSON as current. Two consecutive failed iterations → `stop_reason: hard_failure`.
7. Parse `changes.txt` into `restructure` and `fix_count` metadata for `run.json`.
8. Compute `diff_from_prev.md` (a simple structural diff of the two JSONs).

### 5.4 Validation (precise)

For every emitted `modules.json`, the orchestrator runs:

1. **Parse**: `json.loads()` succeeds.
2. **Shape**: top-level keys `repository` (object with `name`, `type`, `stack`, `summary`) and `modules` (non-empty list). Every module/submodule must have a string `path`, and every `main_files[].path` must be a string; no `path` arrays in v1.
3. **Path-existence check**: for every `path` field (modules, submodules, main_files) verify the path exists under `target_repo`. Record `hallucinated_paths` count in `validation.json` and `run.json`. This is **not** a validation failure (the next reviewer is expected to catch it), just a tracked metric. A drop in this count across iterations is a primary quality signal (§9 M6).
4. **No external schema dependency**: we deliberately do not require a JSON Schema file — the structural check above is enough and matches what `modules.txt` actually specifies.

### 5.5 Stop conditions — precise

Compute these *after* each iteration N:

- **Converged** — true iff:
  - `changes.txt` matches both `^RESTRUCTURE:\s*none` (case-insensitive, on its own logical line) AND a "no fixes" indicator (`FIXES:\s*(none|\(no entries\))` OR zero non-blank lines after `FIXES:` until the next section / EOF), AND
  - **normalized-equal JSON** vs iter N-1, where "normalized" means `json.dumps(obj, sort_keys=True, separators=(",",":"))` of both.
- **Cycle detected** — normalized-JSON of iter N matches **any** prior normalized JSON from iters `max(0, N-4)` through `N-1`. Catches Claude↔Codex ping-pong.
- **Max iterations** — `N >= max_review_iterations`.
- **Budget exceeded** — `sum(cost_usd) + projected_next_iter_cost > budget_usd` (project using mean of prior iters).
- **Hard failure** — two consecutive iterations failed validation.

The first condition that fires wins; record in `run.json:stop_reason`. The `final/modules.json` is the **last successfully validated** JSON.

### 5.6 Target repo mutation guard

Because both agents run with the target repo as their working directory, the orchestrator should detect prompt-contract violations:

1. Before each subprocess starts, capture a repo snapshot:
   - if `target_repo/.git` exists: `git status --porcelain=v1 -z` plus `git ls-files -m -o --exclude-standard -z`;
   - otherwise: a lightweight manifest of file paths, sizes, and mtimes, excluding common generated/vendor directories.
2. After the subprocess exits, capture the snapshot again.
3. If the snapshot changed outside ignored generated files, mark the iteration as a protocol failure, record the changed paths in `run.json`, and stop with `hard_failure` unless the CLI was explicitly run with `--allow-target-writes`.

---

## 6. Component layout (proposed)

```
modules_pipeline/
  __init__.py
  cli.py            # `python -m modules_pipeline run|show|diff|resume`
  orchestrator.py   # iteration loop, stop logic, run-dir bookkeeping
  runtimes.py       # ClaudeCodeOneShot, CodexOneShot (thin subprocess wrappers)
  prompts.py        # load + wrap modules.txt / modules_review.txt
  validate.py       # parse + shape + path-existence audit
  diff.py           # structural diff between two modules.json files
  cost.py           # token + $ estimation per runtime
  run_record.py     # run.json read/write, run_id generation
  resume.py         # detect partial runs, resume from last good iter
```

The two prompt files (`design/modules.txt`, `design/modules_review.txt`) stay untouched — `prompts.py` only reads them.

---

## 7. CLI sketch

```bash
# Analyze the current repo, default loop (claude → codex → claude → codex), max 4 review iters
python -m modules_pipeline run .

# Custom rotation, depth, budget
python -m modules_pipeline run /path/to/repo \
    --agents claude_code,codex,claude_code \
    --max-review-iters 5 \
    --budget-usd 2.00

# Inspect / debug
python -m modules_pipeline show <run_id>            # stop reason, iter count, $ total, hallucination trend
python -m modules_pipeline diff <run_id> 1 2        # structural diff of iter_1 vs iter_2
python -m modules_pipeline resume <run_id>          # continue an interrupted run from last good iter
python -m modules_pipeline list                     # all runs for cwd
```

---

## 8. Decisions made (formerly "open questions")

1. **Prompt delivery**: Always write the wrapped prompt to `iter_N/prompt.txt` for debugging. Codex should receive the prompt through stdin (`codex exec - ...`) because local help documents stdin support. Claude Code should use the current CORAL pattern (`claude -p <prompt>`) unless M0 shows 200KB prompts exceed argv limits; if they do, switch Claude to a verified stdin or prompt-file delivery path before implementing the rest.
2. **Output channel**: file-based, contract enforced by the §5.1 preamble. The final assistant message carries only "DONE" or error text, while raw stdout may be structured CLI events saved to logs. Bootstrap writes `modules.json`; review iterations write both `modules.json` and `changes.txt` under the iteration directory.
3. **Convergence definition**: as written in §5.5 — both the regex on `changes.txt` AND the normalized-JSON-equal must hold. Either alone is too noisy/strict.
4. **Cycle detection**: 4-iteration lookback (§5.5).
5. **Permission/sandbox**: Claude Code uses `--permission-mode acceptEdits` with cwd = `target_repo` and `--add-dir <iter_dir>` so it can write outputs. Codex should prefer `codex exec -C <target_repo> --sandbox read-only --add-dir <iter_dir> --json -` if M0 proves `read-only` plus `--add-dir` permits writing to the iteration directory; otherwise use the narrowest working sandbox and rely on the §5.6 mutation guard. Do not use `--dangerously-bypass-approvals-and-sandbox` against a user's real target repo.
6. **Run dir location**: outside the target repo, under `~/.cache/modules_pipeline/...`. Normal artifacts stay out of the repo; §5.6 catches accidental target writes.
7. **JSON Schema**: not required — the §5.4 structural check is the contract.

### Still-open questions (genuinely)

- **Session resume across iterations** to amortize context cost: out of scope for v1, revisit if average run cost > $X.
- **Parallel reviewers on the same input + merge step**: out of scope for v1.
- **Exact Codex sandbox behavior**: smoke-test whether `--sandbox read-only --add-dir <iter_dir>` allows writing only under the iteration directory. If not, decide between `workspace-write` plus mutation guard or read-only execution plus parsing `--output-last-message`.

---

## 9. Milestones

| # | Milestone | Verifiable when |
|---|---|---|
| **M0** | Both runtimes accept a 200KB prompt through their chosen delivery path and produce files under the iteration directory without mutating the target repo | Two trivial smoke-test scripts pass, including the Codex sandbox/write-root case. **Blocks everything else.** |
| M1 | Bootstrap only | `iter_0_bootstrap/modules.json` exists, parses, has correct top-level shape. |
| M2 | Single review pass | `iter_1_claude_code/{modules.json, changes.txt, diff_from_prev.md}` exist; JSON parses. |
| M3 | Multi-runtime rotation + cost telemetry | `iter_2_codex/...` exists; `run.json` shows per-iter token + $ counts. |
| M4 | Stop logic (all conditions) | Unit tests for the convergence predicate, cycle detector, max-iteration cap, budget guard, and hard-failure latch. |
| M5 | Run record + CLI inspection | `show`, `diff`, `list`, `resume` subcommands return expected output. |
| **M6** | **Output is measurably better than bootstrap** | On CORAL itself as reference: `hallucinated_paths(iter_final) < hallucinated_paths(iter_0)` AND `module_count` is within ±20% of a hand-curated reference list. |

M6 is the only milestone that validates the *pipeline does its job*; M1–M5 only validate that *the pipeline runs*.

---

## 10. Non-goals

- No grading / scoring of the produced JSON beyond the path-existence metric.
- No shared state between iterations beyond files in the run dir.
- No web UI.
- No worktrees, no commits to the target repo.
- No multi-repo batch mode in v1.

---

## 11. Possible future optimizations

- **Session resume per runtime** to keep the agent's repo-exploration cache warm across iterations (cuts redundant filesystem re-reads). Mirrors CORAL's session_id handling in [`coral/agent/builtin/claude_code.py`](../coral/agent/builtin/claude_code.py).
- **Two reviewers in parallel + merge**: Claude and Codex both review iter N's JSON; a third synthesis step (Claude again) merges their `<changes>` blocks.
- **Diff-only review**: instead of re-sending the full prior JSON, send only the diff plus the relevant file paths. Cheaper at scale.
- **GH Action**: trigger on PRs that touch the source tree, post the `diff_from_prev.md` as a PR comment.
