# Implementation Plan: Evidence-Backed Code Optimization Pipeline

> **Pipeline name (placeholder):** `optquest`. Replace freely.
> **Fast-changing items carry a "last verified: 2026-05-13" tag.** Re-verify CLI flags before implementation if more than ~30 days have passed.

---

## 1. Goal

Build a Python pipeline that, given `(repo_url_or_path, module_path)`, produces `changes.json` — a list of optimization proposals tied to concrete code locations in the target module. Each proposal has one of two provenance types: **research-grounded** (backed by exactly one external finding — paper / blog / PR / issue / talk — discovered in Stage 2) or **agent-novel** (proposed by Codex and Claude Code from their own training, *not* mentioned in any Stage-2 finding, on top of a candidate location). Every proposal also carries explicit reasoning emitted by the agent. The pipeline runs Stage 1 (candidate discovery via Claude↔Codex alternating review) and Stage 2 (single GPT Researcher call) in **parallel**, then Stage 3 = 3a (finding↔candidate mapping) + 3b (per-finding research-grounded proposals) + 3c (per-candidate agent-novel proposals) consumes both.

**Acceptance criteria**

- `optquest run --repo <path> --module <subpath>` exits 0 and produces a validated `changes.json` against a published JSON Schema.
- Every record in `changes.json` has: a code location whose file path **exists in the target repo at the resolved git ref**, ≥1 mapped finding with a fetchable URL, and ≥1 ranked change proposal with a non-empty rationale.
- Stage 1 and Stage 2 run concurrently; total wall-clock ≤ max(stage1_time, stage2_time) + small mapping/dispatch overhead.
- Every agent invocation logs `tokens_in`, `tokens_out`, `cost_usd` (where the proxy returns usage data; else 0 with a `cost_unknown: true` flag), `wallclock_s`, `session_id` to a per-run JSONL.
- Pipeline is **read-only** by default against the target repo: no commits, branches, or worktree mutations escape `~/.cache/optquest/<repo_slug>/<run_id>/`.
- Generic across repos. vLLM is a reference for the eval set, not a hardcoded assumption anywhere in code or prompts.

**Version pins (initial; revisit at M1):**

| Component | Pin | Why |
|---|---|---|
| Claude Code CLI | `>= 2.5.0` (the version emitting `total_cost_usd` in the `result` event) | Telemetry stability |
| Codex CLI | latest published release with `codex exec --json --output-schema` and `--add-dir` (released summer 2025; flags still current 2026-05-13) | Schema enforcement |
| Stage 1 / Stage 3a agent models | Whatever Claude Code / Codex CLIs default to; pin explicitly via wrapper config (`--model` for `claude`, `-c model='"…"'` for `codex`) | Decoupled from the rest of the model surface; user controls via their proxy/account |
| Stage 3b agent models | **Both Codex and Claude Code per mapped finding** by default (`--stage3b-mode dual`); the two agents run in parallel against the same finding (each seeing the finding + its mapped candidates) and their proposals are merged + cross-agent-deduped. Switch to `--stage3b-mode alternating` for half the cost (one agent per finding, round-robin) or `--stage3b-mode claude_code` / `--stage3b-mode codex` to pin one. | Dual debiases the per-finding read — same finding, two independent interpretations of how to apply it to the mapped code, with `agreed_with_other_agent: true` as a quality signal |
| Stage 3c agent models | **Both Codex and Claude Code per candidate** by default (`--stage3c-mode dual`); the two agents run in parallel against the same candidate and their proposals are merged + deduped. Switch to `--stage3c-mode alternating` for half the cost (one agent per candidate, round-robin). | Dual is the default because Stage 3c is the *novel* proposal pass — its value comes from cross-checking two independent priors; agreement is a quality signal, disagreement is broader coverage |
| **Stage 2 — GPT Researcher** | `gpt-researcher >= 0.13` (library mode) | Sole Stage-2 provider |
| **Stage 2 — LLM endpoint** | OpenAI-compatible HTTP endpoint at `OPENAI_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com` (LiteLLM proxy). Model name `<MODEL_NAME>` — placeholder, to be filled in | |
| **Stage 2 — retriever (search backend)** | Default `arxiv` (free, key-less); auto-upgrades to `tavily,arxiv` when `TAVILY_API_KEY` is set. See §6.3. | GPT Researcher requires a retriever **separately** from the LLM endpoint |
| Embeddings | **Not used by the MVP mapping.** Stage 3a maps per-finding via coding agents reading the code directly (§7.1). Embedding scaffolding (`BAAI/bge-m3` local fallback; proxy `/v1/embeddings` probe at M0) retained for a v2 hybrid-retrieval pre-filter. | |
| Python | `>= 3.11` (matches GPT Researcher's runtime requirement) | |
| Key libs | `pydantic >= 2.7`, `rank-bm25 >= 0.2.2`, `openai >= 1.40` (the HTTP client for the LiteLLM proxy), `gpt-researcher >= 0.13`, `httpx >= 0.27`, `anyio >= 4`, `sentence-transformers >= 3` (only for bge-m3 fallback) | |

---

## 2. Inputs & Outputs

All four artifacts live under `~/.cache/optquest/<repo_slug>/<run_id>/`. All schemas are checked with `pydantic >= 2.7` models published in `optquest.schema`, which also exports `model_json_schema()` to disk for external validators.

### 2.1 `candidates.json` — Stage 1 output

The audited module is identified by its `module_qualified_name` in the sibling `modules.json` artifact (a `ProjectTree`; see [src/spotlights_engine/schemas/modules.py](../src/spotlights_engine/schemas/modules.py)). Consumers re-resolve the full `Module` record via `ProjectTree.from_json(modules_json).resolve(module_qualified_name)`; we intentionally do **not** duplicate the record into `candidates.json` to keep `modules.json` the single source of truth. Every candidate's `file` MUST lie within the resolved `module.path`.

```json
{
  "module_qualified_name": "foo/bar",
  "candidates": [
    {
      "id": "cand-0001",
      "file": "src/foo/bar/scheduler.py",
      "line_start": 142,
      "line_end": 211,
      "rationale": "Called once per decode step; allocates a new list of pending requests every call; iterates O(N) over the request table to filter."
    }
  ]
}
```

Notes:
- `module_qualified_name` is the `/`-joined preorder path from `ProjectTree.walk()` (e.g. `"v1/engine/core"`), **not** the filesystem path — the latter is `Module.path` and must be obtained by resolving against `modules.json`.


### 2.2 `findings.json` — Stage 2 output

Every finding has exactly five fields: `id`, `title`, `url`, `source_type`, `technique_summary`. See §6.10 for the coercion prompt that produces them.

```json
{
  "schema_version": "1.0",
  "module_qualified_name": "foo/bar",
  "findings": [
    {
      "id": "find-0007",
      "title": "PagedAttention: Memory Management for LLM Serving",
      "url": "https://arxiv.org/abs/2309.06180",
      "source_type": "paper",
      "technique_summary": "Splits KV cache into fixed-size blocks managed by an OS-style page table; eliminates contiguous-allocation fragmentation in batched decoding."
    }
  ]
}
```


### 2.3 `mapping.json` — Stage 3a output (consumed by Stage 3b)

```json
{
  "module_qualified_name": "foo/bar",
  "mappings": [
    {
      "candidate_id": "cand-0001",
      "findings": [
        {
          "finding_id": "find-0007",
          "confidence": "high",
          "reasoning": "scheduler.py touches block allocation for the KV cache; the finding's paged-attention technique applies directly to the loop at line 142"
        },
        {"finding_id": "find-0011", "confidence": "medium", "reasoning": "...", "agent": "codex"}
      ]
    }
  ]
}
```


### 2.4 `changes.json` — final artifact

Candidate-centric. Per candidate, two parallel lists of changes:
- `from_findings` — research-grounded (Stage 3b): each entry links back to the finding it was derived from.
- `from_agents` — agent-novel (Stage 3c): each entry records which coding agent proposed it.

```json
{
  "module_qualified_name": "foo/bar",
  "changes_per_candidate": [
    {
      "candidate_id": "cand-0001",
      "from_findings": [
        {
          "finding_id": "find-0011",
          "title": "Replace per-step pending-list allocation with a reusable ring buffer",
          "description": "Pre-allocate a fixed-capacity request slot pool; reuse across steps; track head/tail with atomic counters."
        }
      ],
      "from_agents": [
        {
          "agent": "codex",
          "title": "Hoist the format-string log call out of the hot dispatch loop",
          "description": "Switch `logger.debug(f\"dispatched {req_id} ...\")` at line 187 to `logger.debug(\"dispatched %s ...\", req_id)` or wrap in `isEnabledFor(DEBUG)` so the level filter short-circuits."
        }
      ]
    }
  ]
}
```

---

## 3. Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │  Driver (Python asyncio + thread pool)      │
                      │  inputs: repo, module_path, scope, budget   │
                      └───────────────┬─────────────────────────────┘
                                      │ fan-out
              ┌───────────────────────┴───────────────────────┐
              │                                               │
              ▼                                               ▼
   ┌──────────────────────────┐                  ┌──────────────────────────┐
   │  Stage 1: Candidates     │                  │  Stage 2: Findings       │
   │  (CORAL-style loop)      │                  │  (single GPT Researcher  │
   │                          │                  │   call)                  │
   │  bootstrap: Claude Code  │                  │                          │
   │   ↓                      │                  │  LLM:       LiteLLM      │
   │  review #1: Codex        │                  │             proxy        │
   │   ↓                      │                  │  retriever: <see §6>     │
   │  review #2: Claude Code  │                  │                          │
   │   ↓ (until stop cond.)   │                  │  → markdown report       │
   │                          │                  │  → coercion call         │
   │  → candidates.json       │                  │  → findings.json         │
   └─────────────┬────────────┘                  └─────────────┬────────────┘
                 │                                             │
                 └─────────────────────┬───────────────────────┘
                                       ▼
                       ┌──────────────────────────────────┐
                       │  Stage 3a: Per-finding fan-out   │
                       │  for each finding (parallel,     │
                       │  semaphore-bounded):             │
                       │    finding[i] → Claude or Codex  │
                       │      (round-robin by index)      │
                       │      reads finding + candidates  │
                       │      + repo (--add-dir)          │
                       │      → edges[{cand_id,conf,why}] │
                       │  invert to candidate-centric →   │
                       │  → mapping.json                  │
                       └─────────────────┬────────────────┘
                                         ▼
                       ┌──────────────────────────────────┐
                       │  Stage 3b: Per-mapped-finding    │
                       │  change generation (Codex +      │
                       │  Claude Code in parallel per     │
                       │  finding; each session reads     │
                       │  finding + its mapped candidates │
                       │  from mapping.json; merge+ cross-│
                       │  agent dedupe)                   │
                       │  → records[] (research_grounded) │
                       └─────────────────┬────────────────┘
                                         │  (3b ∥ 3c)
                       ┌─────────────────▼────────────────┐
                       │  Stage 3c: Per-candidate novel   │
                       │  proposals (Codex + Claude Code  │
                       │  in parallel per candidate;      │
                       │  fed candidates + ALL findings   │
                       │  as a negative list — propose    │
                       │  ONLY changes NOT covered by     │
                       │  findings); merge + dedupe       │
                       │  → novel_records[] (agent_novel) │
                       └─────────────────┬────────────────┘
                                         ▼
                       ┌──────────────────────────────────┐
                       │  Merge + re-pivot                │
                       │  → changes.json                  │
                       │    {records, novel_records,      │
                       │     by_candidate}                │
                       └──────────────────────────────────┘
```

---

## 4. CORAL: What to Lift, What to Skip

Source repo: https://github.com/Human-Agent-Society/CORAL (last verified 2026-05-13). Paths below reference its `main` README's Architecture section; confirm the exact tree to lift from before implementation (§10 Q1).

| CORAL artifact | Decision | Reason |
|---|---|---|
| `coral/agent/runtime.py` (subprocess wrapper for Claude / Codex / OpenCode) | **Lift** — the Popen pattern, log-tailing thread, JSONL parsing of `stream-json` / `--json` | Single best-tested non-interactive wrapper; saves a week. |
| `coral/agent/manager.py` lifecycle (spawn, heartbeat, reboot on max_turns) | **Lift selectively** — keep spawn + max_turns reboot; **skip** heartbeat interrupts and `/loop`-style prompts | Our pipeline is finite, not evolutionary. |
| Agent class registry (`coral/agent/runtime.py` selects by `runtime: claude_code|codex|opencode` from YAML) | **Lift** as a thin `AgentRunner` protocol with two implementations | Same `runtime: …` config UX; lets us add `opencode` later. |
| `_clean_env` style env scrubbing (referenced in CORAL workspace setup) | **Lift** | Prevents `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and telemetry envs from leaking into agent subprocesses unintentionally. **Important interaction with §6:** the orchestrator process holds `OPENAI_API_KEY` / `OPENAI_BASE_URL` for the GPT Researcher call; the agent subprocesses (Claude Code, Codex) must receive their own credentials and *not* inherit the LiteLLM-proxy envs (Codex would otherwise try to call the LiteLLM endpoint instead of OpenAI). |
| Worktrees (`coral/workspace/setup.py` clones a git worktree per agent) | **Skip** for Stages 1 and 3 — both read-mostly; mount the repo read-only (or workspace-write of a separate artifacts dir). **Optionally use** if we ever auto-apply patches in v2. | Worktrees add ~3–5s/agent and a git checkpoint surface area we don't need. |
| `.coral/public/` shared knowledge hub, attempts, notes, skills | **Skip** | Only useful for evolutionary improvement loops — orthogonal to a one-shot candidate-discovery pass. |
| `TaskGrader` / `function_grader` (`coral/grader/`) | **Skip**. Replace with JSON-schema validation (pydantic) per stage. | We don't have a benchmark score; we have schema correctness + URL reachability + path existence. |
| `coral eval` post-commit hook | **Skip** | No commits. |
| `coral ui` web dashboard | **Skip** for MVP; revisit in v2 — the dashboard's leaderboard concept maps cleanly to "ranked changes per candidate". |
| Session resume across iterations (Claude `--resume <uuid>`, Codex `codex exec resume`) | **Skip in MVP**; document as a v2 optimization. The first iteration's full context fits well under the 200K context window of typical frontier models; subsequent reviews can re-prime with a compact handoff. | Resume reduces tokens but doubles the failure modes (session-id loss, model drift across resumes). |

---

## 5. Stage 1 Mechanics — Candidate Discovery

### 5.1 Subprocess invocation (last verified 2026-05-13)

**Claude Code** (https://code.claude.com/docs/en/cli-reference, https://code.claude.com/docs/en/permission-modes):

```
claude -p \
  --output-format stream-json --verbose \
  --input-format stream-json \
  --permission-mode plan \
  --max-turns 30 --max-budget-usd 5.00 \
  --add-dir <artifacts_dir> \
  --resume <prev_uuid?>          # optional, v2
```

- `--permission-mode plan` is the right default for read-only candidate discovery: it explicitly forbids file writes and runs in "research" mode. `acceptEdits` would be used only if we ever need the agent to write its own scratch files inside `--add-dir`. (code.claude.com/docs/en/permission-modes)
- `--output-format stream-json` emits NDJSON; the final `{"type":"result","subtype":"success","total_cost_usd":…,"is_error":false,"duration_ms":…,"num_turns":…,"result":"…","session_id":"…"}` line is the telemetry source (avasdream.com/blog/claude-cli-agentic-wrapper).
- Structured output: `--output-format json --json-schema '{…}'` enforces a schema on the final assistant message; we use this so the agent's emitted candidate list parses cleanly.
- **Argv-length limit**: Linux `MAX_ARG_STRLEN` = 128 KB per single argv (https://github.com/AndyMik90/Auto-Claude/issues/1414, https://wiki.debian.org/CommonErrorMessages/ArgumentListTooLong). Total argv+env ≤ ~2 MB (ulimit -s scaled). **Workaround we adopt:** never pass the prompt as an argv string. Write `prompt.md` and feed it on stdin (`claude -p < prompt.md`) — the `--input-format stream-json` path is documented as the canonical programmatic channel (https://github.com/anthropics/claude-code/issues/24594). For file-content injection, use the `@path` reference syntax in the prompt body rather than inlining file content.
- Cost telemetry comes from the terminal `result` event's `total_cost_usd`, `usage.input_tokens`, `usage.output_tokens`, `duration_ms`, `session_id`.

**Codex** (https://developers.openai.com/codex/cli/reference, https://developers.openai.com/codex/noninteractive, last verified 2026-05-13):

```
codex exec - \                                       # prompt on stdin
  --json \                                            # NDJSON events
  --output-last-message <path> \                      # final assistant message
  --output-schema <path/schema.json> \                # validate model output
  --sandbox read-only \                               # default; safe
  --add-dir <artifacts_dir> \                         # see note below
  --skip-git-repo-check \                             # only outside a repo
  -c model='"gpt-5.5"' \
  -c model_reasoning_effort='"high"' \
  -C <repo_path>
```

- `codex exec` is the documented non-interactive entry point. Plain `codex` is TUI-only and fails with "stdout is not a terminal" under a non-tty parent (smithery.ai/skills/Lucklyric/codex).
- **`--sandbox read-only` + `--add-dir` interaction:** the Codex docs (https://www.simplified.guide/codex/writable-directory-add, https://github.com/openai/codex/issues/2797, https://developers.openai.com/codex/cli/features) consistently describe `--add-dir` as "expose more **writable** roots", and the canonical example is `codex exec -s workspace-write --add-dir /tmp/codex-writable …`. Read-only mode has no writable roots by definition (https://developers.openai.com/codex/concepts/sandboxing). **Conclusion**: `--sandbox read-only --add-dir X` is *not* a supported "read everything, write only to X" combination. **Recommended workaround**: use `--sandbox workspace-write` with `-C <artifacts_dir>` as the workspace root and add the target repo via `--add-dir <repo>`. Codex's `sandbox_workspace_write.exclude_*` keys prevent unintended writes to system tmp. If strict read-only of the repo is mandatory, run the target repo on a read-only bind mount and let workspace-write apply only to the artifacts directory.
- Output channels: `--json` writes NDJSON state events to stdout; `--output-last-message <path>` writes the final assistant message to a file; `--output-schema` validates the final response against a JSON Schema. Use all three.
- **`--output-schema` enforcement strength is open** — see §10 Q3.
- Session id: every `codex exec` prints `session id: <uuid>` near the top of the formatted output and emits a session-init event in the JSON stream. We extract it from the JSON stream for v2 resume.
- Argv limit: same Linux 128 KB MAX_ARG_STRLEN applies. **Workaround**: `codex exec -` reads the prompt from stdin (the dash is explicit and documented at developers.openai.com/codex/noninteractive).
- Cost telemetry: tokens-in/out and tool calls (including `web_search`) appear in the JSON event stream; wallclock is measured by the orchestrator.
- **Credential isolation:** Codex must not see `OPENAI_BASE_URL` from the orchestrator's environment (that variable is intended for the LiteLLM proxy used by GPT Researcher in Stage 2). Strip it from the agent subprocess env via the lifted `_clean_env` and pass Codex's own auth via its config file or `-c` flags.

### 5.2 Bootstrap + alternating review loop

```
i=0: claude_code bootstrap (--permission-mode plan)
     prompt: BOOTSTRAP_PROMPT
     reads:  module_path, optional modules.json, optional bench/profile.json
     emits:  candidates_v0.json (schema-validated)

i=1: codex review
     prompt: REVIEW_PROMPT(candidates_v0.json, role="adversarial reviewer")
     emits:  candidates_v1.json

i=2: claude review
     prompt: REVIEW_PROMPT(candidates_v1.json, role="adversarial reviewer")
     emits:  candidates_v2.json

... until stop condition ...
```

Stop conditions (ALL evaluated each iteration; ANY trigger ends the loop):

1. **Normalized-JSON equality**: `canonical_json(candidates_iN) == canonical_json(candidates_i{N-1})` after sorting by `(file, line_start, line_end)` and stripping whitespace/timestamps.
2. **"No changes" predicate**: the agent's emitted `meta.delta` field is `{"added":[],"removed":[],"modified":[]}`.
3. **Cycle detection**: hash of canonical JSON has appeared before in this run.
4. **Max-iter cap**: default 6; configurable.
5. **Budget guard**: cumulative Stage-1 `cost_usd` ≥ `--stage1-budget-usd` (default $5).
6. **Hard-failure latch**: if any iteration produces invalid JSON twice in a row after retry, abort Stage 1 and emit the last good `candidates.json`.

### 5.3 Scope bounding

Large modules need pruning. Strategy stack, tried in order:

1. `--scope-paths <glob>` CLI flag passed by user (highest precedence).
2. Optional `modules.json` sibling artifact (if a sibling pipeline produced one): a `{file: importance_score}` map.
3. Profiling artifacts (`bench/profile.json`, `*.prof`, `*.pstats`, `*.nsys-rep` paths) listed in the bootstrap prompt as hot-path hints.
4. Guided enumeration: agent runs `ls`/`grep`/`fd` inside the module and proposes a scope before generating candidates (a planning sub-step inside the bootstrap prompt).

### 5.4 Prior art consulted for Stage 1 prompt design

- **Agentless** (https://arxiv.org/abs/2407.01489) — its file-localization → relevant-code-locations decomposition matches our two-stage scope-then-list pattern. Adopted: emit a "scope" sub-output before the candidate list (folded into the bootstrap prompt's "What to read first" step).
- **SWE-agent / SWE-bench solver families** (https://swe-agent.com) — their ACI primitives (open, search, scroll) inform what *not* to do: we don't replicate the agentic environment; the agent uses the underlying CLI's built-in file tools instead.
- **OpenHands** (https://github.com/All-Hands-AI/OpenHands) — evaluated as an M5 baseline. Its general "agent does everything in one loop" is exactly what we *don't* want for the cost-disciplined separation of research from coding.
- **KernelBench's prompting style** (https://scalingintelligence.stanford.edu/blogs/kernelbench/) — for GPU-kernel candidates, their explicit "replace this PyTorch op with a custom kernel" framing carries over: when a candidate's `rationale` flags a kernel launch site or fusion opportunity, the Stage-3b prompt should be augmented with KernelBench-style "rewrite this op" exemplars.
- **PIE (pie4perf)** and **FasterPy** — both prove that performance-aware retrieval (give the model exemplars of past optimization edits) significantly beats blind prompting. We borrow the *pattern* (Stage 2 retrieves real techniques rather than letting Stage 3b invent them blind) but keep the finding record minimal: just the URL and a one-paragraph `technique_summary`. The Stage-3b agent fetches the source itself when it needs implementation detail.

### 5.5 Validation per iteration

- pydantic schema parse → reject on failure (retry once with a strict-mode reminder appended to the prompt).
- Path existence: every `candidates[i].file` resolves to an existing file in the working checkout; bad paths → drop the candidate and continue (don't fail the run); count drops in the per-run JSONL telemetry.
- Line range sanity: `1 ≤ line_start ≤ line_end ≤ file_line_count`; out-of-range candidates dropped the same way.

### 5.6 Concrete Stage-1 prompts (drop-in)

**Bootstrap prompt** (`prompts/stage1_bootstrap.md`):

````
You are a senior performance engineer auditing a module of a Python (or Python+CUDA/C++)
codebase for *high-yield optimization opportunities*. You do NOT write fixes. You enumerate
candidate locations, each with a falsifiable rationale.

## Repository facts
- repo: {repo_url}
- git_sha: {git_sha}
- module_path: {module_path}
- scope: {scope_paths}
- profiling artifacts (if any): {profile_paths}
- modules.json (if any): {modules_json_path}

## What counts as a "high-yield candidate"
A code location worth attention if any of the following are visibly true:
- hot loop or inner kernel (called per token / per batch / per request)
- kernel launch site, allocator-heavy path, or Python ⇄ C boundary
- sync/lock point, GIL-blocking section, or unnecessary thread serialization
- batched-op opportunity (currently per-item) or vectorization-amenable shape
- serialization boundary (pickle, json.dumps, tensor.cpu()) on a hot path
- cache-unfriendly access pattern, redundant recomputation, or O(N) inside O(N)
- memory-bound op that could be fused, paged, or recomputed

## What to read first
1. List `{module_path}` and read every file under it (use scope filter).
2. If `modules.json` exists, prioritize files with highest importance_score.
3. If profiling artifacts exist, list them and use as evidence.
4. Cross-reference recent issues/PRs only by mentioning their *titles* — do NOT
   fabricate URLs.

## Output (REQUIRED — strict JSON)
Emit ONE JSON object matching the schema at @candidates.schema.json. Each candidate
has exactly these fields:
- `id`            — short slug (e.g. `cand-0001`); minted by you for new candidates
- `file`          — repo-root-relative path that exists in this checkout
- `line_start`    — 1-indexed inclusive
- `line_end`      — 1-indexed inclusive, `≥ line_start`
- `rationale`     — ≤ 240 chars, falsifiable (cite the loop / the alloc / the sync)

Also emit `meta.delta = {"added": [...all ids...], "removed":[], "modified":[]}`.
Constraint: `5 ≤ len(candidates) ≤ 40`. Do NOT explain outside the JSON object.
````

**Review prompt** (`prompts/stage1_review.md`):

````
You are reviewing another agent's candidate list. Your job is adversarial: prune
false positives, merge duplicates, sharpen rationales, and add at most 5 *new*
candidates the previous pass missed.

## Inputs
- previous candidates: @candidates_prev.json
- module: {module_path} at {git_sha}
- scope: {scope_paths}

## Rules
- DO NOT inflate the list. If the previous pass was good, return it nearly unchanged.
- Remove any candidate whose `file` does not exist or whose `line_start..line_end`
  is out of range for that file.
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others; list the dropped ones in `meta.delta.removed`).
- For each kept candidate, you MAY tighten `rationale` or adjust `line_start` /
  `line_end`. Don't change `id`.
- For each NEW candidate, the rationale must cite a specific code construct
  (function name, loop, alloc call, sync primitive).

## Output (REQUIRED — strict JSON)
Same schema as the previous pass. Additionally:
- `meta.delta.added`   = ids of candidates you added
- `meta.delta.removed` = ids of candidates you removed
- `meta.delta.modified` = ids whose rationale or line range changed

If you genuinely have no changes, return the input unchanged and set all three
delta arrays to []. Do NOT pad.
````

---

## 6. Stage 2 Mechanics — Deep Research via GPT Researcher

### 6.1 Provider: GPT Researcher

- Repo: https://github.com/assafelovic/gpt-researcher (Apache-2.0)
- Library import: `from gpt_researcher import GPTResearcher`
- Pin: `gpt-researcher >= 0.13`
- Why: importable directly from Python, emits a structured report we coerce to our findings schema, supports arbitrary OpenAI-compatible LLM endpoints via env vars, separates LLM from retriever, MIT-friendly license.

### 6.2 LLM configuration — LiteLLM proxy

GPT Researcher honors `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the process environment (and equivalent config-file keys). The wrapper sets:

```bash
export OPENAI_API_KEY="…"                                            # the proxy key
export OPENAI_BASE_URL="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
# Pin the same model name everywhere GPT Researcher looks for one
export FAST_LLM="openai:<MODEL_NAME>"
export SMART_LLM="openai:<MODEL_NAME>"
export STRATEGIC_LLM="openai:<MODEL_NAME>"
export EMBEDDING="openai:<EMBEDDING_MODEL_NAME>"   # see §6.4
```

…before constructing `GPTResearcher(query=…, report_type="deep_research", report_format="markdown")`. The orchestrator must isolate these vars from the Claude Code / Codex subprocesses (see §4 _clean_env note).

Stage 2 is text-only; no image / vision input.

### 6.3 Retriever (search backend) — default `arxiv`, Tavily recommended

Stage 2 has two retriever modes:

- **Default — `RETRIEVER=arxiv`.** No API key, no install-time secret, no paid egress. Calls the arxiv API directly; authors / abstracts / arxiv categories survive into `findings.json` unchanged. Always on; works out of the box on a node with internet access.
- **Recommended add-on — `RETRIEVER=tavily,arxiv`.** Adds Tavily's AI-curated web search in parallel with the arxiv API. Surfaces what arxiv can't: engineering blogs (Cloudflare/Meta/Netflix/etc.), GitHub issues and PRs, vendor docs (CUDA/PyTorch/JAX/MLIR), Stack Overflow, conference talks, news. Requires `TAVILY_API_KEY`. Paid — see costs below.

**Startup behavior (no flag needed in the common case):** the orchestrator inspects the environment at startup. If `TAVILY_API_KEY` is set, it exports `RETRIEVER=tavily,arxiv`; otherwise it exports `RETRIEVER=arxiv` and prints a one-line warning to stderr:

```
[stage2] running with arxiv only — set TAVILY_API_KEY to enable Tavily for blog / GitHub / vendor-doc coverage
```

An explicit `--retriever <value>` flag remains available for overrides (e.g., `--retriever duckduckgo` for a laptop smoke test, `--retriever custom,arxiv` for an in-VPC redeploy).

GPT Researcher accepts a list of retrievers, runs them in parallel, and merges results before re-ranking ([retriever docs, verified 2026-05-13](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers)).

**What the default costs in coverage:**

| Source type | `arxiv` default | `tavily,arxiv` |
|---|---|---|
| Peer-reviewed / preprint papers | ✓ (clean metadata via API) | ✓ (arxiv primary, Tavily as backup) |
| Engineering blogs (Cloudflare, Meta, Netflix, …) | — | ✓ |
| GitHub issues, PRs, discussions | — | ✓ |
| Vendor docs (CUDA, PyTorch, JAX, MLIR, …) | — | ✓ |
| Conference talks, slide decks (non-arxiv) | — | ✓ |
| Stack Overflow / Reddit / HN | — | ✓ |

Running arxiv-only is a **known degraded mode**, not a failure: smaller findings sets, higher Stage 3 mapping-orphan rates for optimization opportunities documented only in blogs/issues. The pipeline still runs end-to-end. The Stage 2 wrapper sets `findings.query_context.retriever_limitations = "arxiv_only"` in that mode so downstream consumers can interpret the lower hit rate.

**Cost.**

- **arxiv API:** free. Rate-limited to ~1 request / 3 seconds; the underlying `arxiv` Python package handles back-off automatically. ~30K result cap per query (irrelevant for our sub-query sizes).
- **Tavily:** $30/mo for 10K credits, ~$0.008/credit on PAYG ([pricing, verified 2026-05-13](https://docs.tavily.com/documentation/api-credits)). One `deep_research` call burns ~20–60 credits (per-subquery search + extraction). The $30 tier comfortably absorbs ~150–500 runs/month. Free tier is 1K credits/month with no card required, enough for early dev work.

**Not used:**

- **DuckDuckGo** — rate-limited / blocked from cloud IPs in practice. Available via `--retriever duckduckgo` for laptop bring-up only.
- **Exa, Serper, SerpAPI, Bing, Google** — Tavily is GPT Researcher's reference retriever and the best-tested integration. No evidence the others would materially improve recall here. Revisit only if M1 shows Tavily relevance is poor.

**If Tavily egress is ever lost** (e.g., deployment moves into a restricted VPC), in-VPC alternatives: `RETRIEVER=custom,arxiv` with an adapter for an internal search index ([custom retriever docs](https://docs.gptr.dev/docs/gpt-researcher/search-engines/retrievers#custom-retriever)), or `RETRIEVER=searx,arxiv` against a self-hosted [SearXNG](https://github.com/searxng/searxng). The arxiv default keeps working in either case.

### 6.4 Embeddings — not used by MVP mapping; deferred

The per-finding fan-out (§7.1) judges applicability inside each coding-agent call, so no embeddings endpoint is needed for the MVP. This section is a scaffold for a possible v2 hybrid-retrieval pre-filter:

- A v2 `--prefilter embeddings` (not implemented) would call `client.embeddings.create(model="<EMBEDDING_MODEL_NAME>", input=[...])` against the LiteLLM proxy, with a `sentence-transformers` / `BAAI/bge-m3` local fallback.
- The M0 proxy selftest probes `/v1/embeddings` so the endpoint's availability is recorded; result is informational only.

The MVP's only `--prefilter` option is `bm25` (pure `rank_bm25`, no embedding endpoint touched).

### 6.5 Single-call vs deep-research mode

GPT Researcher's `report_type` options (https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started-with-docker, source: `gpt_researcher/utils/enum.py`):

- `research_report` — single retrieval pass, single synthesis; fastest, cheapest.
- `detailed_report` — section-by-section, deeper.
- `deep_research` — multi-agent, tree-of-thought style search, deepest. Best for our use case.

Pin: `report_type="deep_research"`. Expect 5–15 minutes wall-clock per call.

### 6.6 Markdown-to-schema coercion (mandatory)

GPT Researcher emits **markdown**. Our pipeline needs validated JSON matching the `findings.json` schema. The orchestrator therefore makes a **second LLM call** through the same LiteLLM proxy:

```
COERCION CALL
  endpoint: OPENAI_BASE_URL
  model:    <MODEL_NAME>
  system:   "You convert a research report into a strict JSON findings list."
  user:     report_markdown + COERCION_PROMPT (below)
  response_format: {"type": "json_schema", "json_schema": findings_schema}
```

If the proxy's `<MODEL_NAME>` does not support `response_format=json_schema`, the wrapper falls back to plain JSON-mode (`response_format={"type":"json_object"}`) and then validates with pydantic, retrying once on parse failure with a strict reminder. See §10 Q6.

### 6.7 Per-call budget cap

Enforce:

- `--stage2-wallclock-s` (default 1800 s = 30 min) — hard timeout on the GPT Researcher call.
- `--stage2-max-iterations` — pass to GPT Researcher `config_path` to bound the deep-research tree.
- `--stage2-budget-usd` — best-effort; only enforceable if the LiteLLM proxy returns `usage` blocks (§10 Q5).

### 6.8 Validation

- Schema parse → reject on failure (retry coercion once with strict reminder).
- For every `finding.url`: HTTP HEAD with 10s timeout; mark `url_reachable: true|false`; warn if `unreachable_count > 0.2 * total_count`.
- Minimum count: ≥5 findings or Stage 2 is flagged degraded (run still proceeds; Stage 3a will likely produce orphans).

### 6.9 Stage 2 research prompt (drop-in, fed as `query` to GPTResearcher)

````
You are conducting a focused literature/engineering survey to support a code-optimization
pipeline. Your output will be parsed by a downstream mapper that attaches your findings
to specific code locations in the user's module.

## Module under audit
- repo: {repo_url}
- module_path: {module_path}
- one-paragraph summary (from the bootstrap candidate-discovery agent):
  """
  {module_summary}
  """
- top-level kinds of code present: {top_level_tags}   # e.g., ["scheduler","kv-cache","cuda-kernels"]

## What to look for
Surface, in order of preference:
1. Peer-reviewed papers proposing techniques with measured speedups, memory wins, or
   throughput wins relevant to {top_level_tags}.
2. GitHub pull requests / issues IN ANY REPO (not just {repo_url}) that landed measured
   wins in similar systems.
3. Engineering blog posts from vendors and frameworks (NVIDIA, PyTorch, JAX, vLLM, SGLang,
   TGI, DeepSpeed, etc.) describing concrete optimizations with numbers.
4. Conference talks (MLSys, OSDI, SOSP, ASPLOS, SC, PyTorch Conf, GTC) with slides/recordings.

Avoid: pure marketing posts, tutorials with no benchmarks, social-media speculation.

## Bias and constraints
- Prefer sources <= 36 months old unless the technique is foundational.
- For every finding, you MUST resolve a real, fetchable URL. No paraphrased
  recollections without a URL.
- Aim for 8-20 findings. Quality > quantity.

Produce a thorough markdown report. Group findings into logical sections. For each finding,
include at minimum: title, URL, source type (paper/PR/issue/blog/talk/docs/codebase), and a
≤80-word summary of the technique the source proposes. You may include additional context
(gains, components, suggested changes, evidence assessment) in the prose if helpful for
the reader, but only the four fields above plus the summary will survive coercion into
`findings.json` — the downstream Stage-3 agents fetch the source URL themselves when they
need implementation detail.
````

### 6.10 Stage 2 coercion prompt (drop-in, fed to second LLM call)

````
You are converting a research report into a strict JSON findings list.

## Input
A markdown report on optimization techniques relevant to a code module. The report is
appended below the `---` separator.

## Output
A single JSON object matching the schema at @findings.schema.json. Every finding has
exactly five fields — no more:
- `id` — stable identifier of the form `find-NNNN` (zero-padded, sequential within the report)
- `title`
- `url` — a real, fetchable URL present in the report; do NOT invent URLs
- `source_type` ∈ {paper, pr, issue, blog, talk, docs, codebase}
- `technique_summary` (≤ 80 words)

- Only include findings whose URL is present in the report.
- Drop any finding for which you cannot fill all five fields.
- Do not emit prose outside the JSON object.
- Do not emit any field other than the five above.

---

{report_markdown}
````

---

## 7. Stage 3 Mechanics — Mapping + Change Generation + Novel Proposals

### 7.1 Mapping (Stage 3a) — per-finding agent fan-out, alternating Claude Code / Codex

Stage 3a iterates over **findings**, not pairs. For each finding the orchestrator spawns **one** coding-agent subprocess that receives:

- the single finding (`id`, `title`, `url`, `source_type`, `technique_summary` — the full 5-field record)
- the full `candidates.json` (typical Stage 1 output is 5–50 candidates, well under an agent's context budget)
- read-only access to the repo via `--add-dir`

…and returns the subset of candidates the finding actually applies to, with a per-edge `confidence ∈ {low, medium, high}` and reasoning. Since the finding record itself is minimal, the agent is expected to fetch the `url` directly (or read its training-data memory of the cited work) before deciding — the prompt includes an explicit "read the source before deciding which candidates match" instruction (§7.1.2). Agents alternate Claude Code / Codex by finding index (round-robin: `findings[0]→claude`, `findings[1]→codex`, `findings[2]→claude`, …) to debias single-model failure modes — Stage 3b's dual mode (§7.2) takes this further by running both agents per finding in parallel, but Stage 3a stays alternating to keep the mapping pass cheap.

Once all per-finding subprocesses complete, the orchestrator groups edges by candidate and persists `mapping.json` — every edge with `confidence ≥ --map-min-confidence`, no top-K truncation — so Stage 3b can pivot back to finding-centric losslessly.

#### 7.1.1 Mechanics

```
INPUT:  candidates.json (N candidates), findings.json (M findings), repo
OUTPUT: mapping.json (candidate-centric, all edges with confidence ≥ --map-min-confidence)

semaphore     = asyncio.Semaphore(--max-parallel-agents)        # default 4
agents_cycle  = itertools.cycle(["claude_code", "codex"])        # alternating

async def map_one_finding(f, agent_choice):
    async with semaphore:
        invoke agent_choice with:
            - --permission-mode plan (claude) / --sandbox workspace-write (codex)
            - --add-dir <repo-path>                  # read-only repo access
            - stdin: STAGE3A_MAP_PROMPT(f, candidates.json)
            - timeout: --per-finding-wallclock-s     # default 90s
        returns: {finding_id, edges: [{candidate_id, confidence ∈ {"low","medium","high"}, reasoning}]}

tasks = [map_one_finding(f, next(agents_cycle)) for f in findings]
per_finding_results = await asyncio.gather(*tasks, return_exceptions=True)

edges = flatten([r.edges for r in per_finding_results
                 if not isinstance(r, Exception) and r.get("edges")])

# Persist a candidate-centric grouping of every edge above --map-min-confidence.
# Confidence ordering: high > medium > low. Stage 3b pivots back to finding-centric in one line (§7.2).
mapping = group_by_candidate(edges, min_confidence=--map-min-confidence)   # no top-K
# orphans are not persisted; downstream derives them as {c.id for c in candidates} \ mapping.keys()
```

`--mapping-agent {alternating|claude_code|codex}` — default `alternating`. (Stage 3b uses `--stage3b-mode {dual|alternating|claude_code|codex}` instead, default `dual`; see §7.2.)

#### 7.1.2 Per-finding budget, timeouts, and validation

- **Wallclock per finding:** `--per-finding-wallclock-s` (default 90s — bounded task: read one finding + N candidates + verify in code).
- **Cost per finding:** `--per-finding-budget-usd` (default $0.20).
- **Stage-3a parent budget:** `--stage3a-budget-usd` (default $4 — covers ~20 findings × $0.20).
- **Schema validation:** parse each per-finding result with the `mapping_edges` pydantic schema. On parse failure, retry once with a strict-mode reminder appended to the prompt. On second failure: record `{finding_id, edges: [], error: "schema_invalid"}` and continue — **do not** fail the stage.
- **Reference integrity:** drop any edge whose `candidate_id` is not in `candidates.json`; record the drop count in `mapping.json.meta.drops` for diagnostics.
- **Confidence sanity:** confidence must be one of `{"low", "medium", "high"}`; coerce unknown / missing values to `"low"` and warn. Require `reasoning` length ≥ 20 chars (else demote the edge to `"low"` with a warning).
- **Per-finding hang detection:** if a subprocess produces no NDJSON output for `--per-finding-stall-s` (default 60s), kill it and record `{error: "stalled"}` for that finding.

#### 7.1.3 Optional BM25 pre-filter for large candidate sets

If `len(candidates) > --map-prefilter-threshold` (default 100), each per-finding agent receives a BM25-narrowed shortlist of the top-50 candidates instead of the full list, to keep the prompt under context budget. Disabled by default because typical Stage 1 outputs are well under 100. Configured via `--prefilter {none|bm25}` (default `none`). Pure `rank_bm25` over the tokenized `(file, rationale)` text — no embedding endpoint required.

#### 7.1.4 Completeness check (mapping output)

Every candidate ideally has ≥1 edge with `confidence ≥ --map-min-confidence` (default `low`). Orphans (candidates with no qualifying finding) are not persisted on `mapping.json`; Stage 3b derives the set as `{c.id for c in candidates.json} \ {m.candidate_id for m in mapping.json.mappings}`. Because Stage 3b iterates findings, orphans get no proposals by default — they are surfaced in the run summary and listed in `changes.json.stage3b.orphan_candidates[]`. Enable `--orphan-sweep` to run one extra Claude Code session per orphan candidate with the general-heuristics fallback prompt (§7.2). Orphan rate is surfaced as a CLI summary warning.

**Many-to-many shape preserved.** A finding may map to multiple candidates; a candidate may collect edges from multiple findings. `mapping.json` is the candidate-centric grouping of every per-finding edge at or above `--map-min-confidence`, with no top-K cap — lossless, so Stage 3b can pivot back to finding-centric without dropping edges.

#### 7.1.5 Concrete Stage-3a prompt (drop-in)

`prompts/stage3a_map.md`:

```
You map ONE research finding to a list of code optimization candidates. You do
NOT propose changes — that happens later. Your only job is to decide which
candidates this finding applies to, and explain why.

## Inputs
- the finding (below) — a technique surfaced by deep-research (paper / blog /
  PR / issue / talk). Five fields only: `id`, `title`, `url`, `source_type`,
  `technique_summary`. If the summary is insufficient to judge applicability,
  fetch the `url` (or recall the work from training data) before scoring.
- `candidates.json` — the full candidate list from Stage 1. Each candidate has
  `id`, `file`, `line_start`, `line_end`, `rationale`.
- the repository (read-only via --add-dir) — open files, read code, verify
  whether the technique actually applies to each candidate's location.

## Rules
- For each candidate, decide whether THIS finding's technique applies to the
  code at THIS location. A topic match is not enough — the code shape must fit.
  Example reject: a "fused-attention CUDA kernel" finding mapped to a
  pure-Python bytecode candidate, even if both involve attention.
- Read the actual code around (`file`, `line_start..line_end`) before assigning
  `high` confidence. Don't extrapolate from the rationale alone.
- Confidence ∈ {"low", "medium", "high"}:
    high   = you read the code and the technique applies as-is (or with trivial adaptation)
    medium = plausible — the code shape fits but real adaptation work is needed,
             or you couldn't fully verify by reading the source
    low    = weak / speculative; still worth surfacing for a maintainer to judge
    (if it's weaker than `low`, do not emit at all)
- Reasoning ≥ 20 chars and grounded in what you read (cite file:line if useful).
- Emit ONLY candidates the finding actually maps to. Don't enumerate rejections.
- If NO candidate fits, emit `"edges": []` — that's a valid result.

## Finding
{finding_json}

## Output (REQUIRED — strict JSON, no prose outside)
{
  "finding_id": "{finding_id}",
  "edges": [
    {"candidate_id": "cand-XXXX", "confidence": "low|medium|high", "reasoning": "..."}
  ]
}
```

**Prior art consulted:** RepoCoder (https://arxiv.org/abs/2303.12570), CodeRAG-Bench (https://arxiv.org/abs/2406.14497), and Agentless (https://arxiv.org/abs/2407.01489) for retrieval and code-to-spec patterns. Stage 3a and Stage 3b share the same per-finding fan-out shape — same semaphore, same NDJSON tailing, same schema-validate-then-retry envelope. They differ in: (a) Stage 3a alternates Claude Code / Codex per finding (one session each), while Stage 3b runs **both** agents in parallel per finding (two sessions each) and cross-agent-dedupes — same dual pattern as Stage 3c, (b) Stage 3a emits categorical-confidence edges while Stage 3b emits ranked proposals grouped by candidate, and (c) Stage 3b is given the repo + the finding's mapped candidates only (not the full candidate list).

### 7.2 Change generation (Stage 3b)

Per mapped finding, **both** Claude Code and Codex run as separate subprocesses in parallel (`--stage3b-mode dual`, default). Each session is fed:

- the finding (the 5-field record: `id`, `title`, `url`, `source_type`, `technique_summary`); the agent is expected to fetch `url` before proposing changes when the summary is insufficient,
- the **subset of candidates** that finding mapped to with `confidence ≥ --map-min-confidence` — pivoted from `mapping.json.mappings[]` (one-line group-by on `finding_id`, done once at stage start),
- read-only repo access via `--add-dir`.

Each session returns ranked proposals **grouped by candidate**: for each mapped candidate, 1–5 proposals derived from this one finding's technique. The proposal's supporting finding is implicit (the session's own `finding_id`) — Stage 3b never mixes findings within a session. The two agents' outputs for the same `finding_id` are then merged and cross-agent-deduped into one `records[]` entry.

**Why dual, why per finding.** Per finding (vs per candidate) lets the model load one technique's mental model and reason about it across multiple matching code locations — a single research paper applied carefully to N sites beats N shallower applications of N findings to one site each. Dual (vs single-agent) debiases the per-finding read: both models see the same finding and the same candidates, and we observe where they agree on the concrete application, which is a stronger signal than either one alone. Cost is roughly 2× a single-agent pass, capped by `--stage3b-budget-usd`.

**Fan-out shape.**
```
INPUT:   findings.json, mapping.json, repo
OUTPUT:  records[] (one record per finding, with cross-agent-deduped proposals)

semaphore = asyncio.Semaphore(--max-parallel-agents)             # default 4

# Pivot mapping.json once at stage start.
edges_by_finding = defaultdict(list)
for m in mapping.mappings:
    for e in m.findings:
        edges_by_finding[e.finding_id].append({"candidate_id": m.candidate_id, **e})

async def changegen_one(f, agent_choice):
    mapped_candidates = edges_by_finding[f.id]
    async with semaphore:
        invoke agent_choice with:
            - --add-dir <repo-path>                              # read-only
            - stdin: STAGE3B_CHANGEGEN_PROMPT(f, mapped_candidates)
            - timeout: --per-finding-changegen-wallclock-s       # default 180s
        returns: {finding_id, agent, candidate_proposals[]}

# dual mode (default)
tasks = [changegen_one(f, a) for f in findings_with_edges for a in ("claude_code", "codex")]
# alternating mode
agents_cycle = itertools.cycle(["claude_code", "codex"])
tasks = [changegen_one(f, next(agents_cycle)) for f in findings_with_edges]
# pinned mode
tasks = [changegen_one(f, AGENT) for f in findings_with_edges]   # AGENT ∈ {"claude_code","codex"}

results = await asyncio.gather(*tasks, return_exceptions=True)
records = merge_and_dedupe(results)                              # see "Dedupe" below
```

Other modes via `--stage3b-mode`:
- `alternating` — one agent per finding, round-robin (half the cost; loses agreement signal).
- `claude_code` / `codex` — pin to one agent (smoke tests, ablations).

**Budgets.**
- Wallclock per session: `--per-finding-changegen-wallclock-s` (default 180s — longer than Stage 3a because the session generates proposals for ≤K candidates, not just classifies).
- Cost per session: `--per-finding-changegen-budget-usd` (default $0.50).
- Parent stage budget: `--stage3b-budget-usd` (default $16 for dual mode at ~20 findings × 2 agents × $0.50 with headroom; halved automatically for `alternating`, halved again for pinned).
- Stall detection: kill the subprocess if no NDJSON output for `--per-finding-changegen-stall-s` (default 90s); record `{error: "stalled"}` and continue.

**Validation.** Parse each per-session result with the `finding_changes` pydantic schema. On parse failure, retry once with a strict-mode reminder appended. On second failure: record `{finding_id, agent, candidate_proposals: [], error: "schema_invalid"}` and continue — **do not** fail the stage. Drop any `candidate_id` not in the pivoted edge list for that finding (defense against the model inventing candidates). A finding's overall record is `succeeded` if **at least one** of its agent sessions parsed cleanly.

**Cross-agent dedupe (dual mode only).** Two layers:
1. **Same finding × same candidate × same idea.** When both Claude Code and Codex produce a proposal for the same `(finding_id, candidate_id)` with title similarity ≥ 0.7 or LLM-judge agreement on borderline (0.3–0.7), they collapse to **one** proposal with `proposed_by: ["claude_code", "codex"]` and `agreed_with_other_agent: true`. The merged proposal takes the union of `prerequisites`, max-severity `risk`, and median `effort_estimate`. Dropped duplicates are counted in `stage3b.dedupe.rejected_cross_agent_duplicate`.
2. **Same finding × same candidate × distinct ideas.** Both proposals kept, each with `proposed_by: ["claude_code"]` or `["codex"]` and `agreed_with_other_agent: false`. These signal that the finding admits multiple applications at this site; the maintainer sees both.

Cross-agent dedupe is *only* applied within the same `(finding_id, candidate_id)` — proposals derived from different findings stay separate even if they look similar, because the citation differs.

**Orphan sweep (off by default).** When `--orphan-sweep` is set, after the per-finding fan-out completes, run one additional Claude Code session per orphan candidate with the general-heuristics fallback prompt. (Single-agent because orphan sweep is a fallback, not a primary pass; pair-debiasing isn't worth the extra cost on candidates Stage 2 failed to ground.) Each such session emits proposals tagged `from_finding_id: null` and is accounted in `stage3b.orphan_sweep_*` telemetry. Capped by `--orphan-sweep-budget-usd` (default $2). Disabled by default because orphans typically signal weak Stage-2 retrieval, not a Stage-3b gap.

### 7.3 Change-generation prompt (drop-in)

`prompts/stage3b_changegen_per_finding.md`:

````
You are proposing concrete, ranked optimization changes for ONE research
technique applied to a small set of code locations. You do NOT write a patch —
you produce a structured proposal list per location that a human (or a future
v2 patch-writer) can act on.

## The technique (this session's single finding)
- finding_id:   {finding_id}
- title:        {f.title}
- url:          {f.url}
- source_type:  {f.source_type}
- summary:      {f.technique_summary}

The 5 fields above are the entire finding record. If the summary is too thin to
ground concrete proposals, FETCH the `url` (or draw on training-data recall of
the cited work) before writing anything — do not invent specifics not present in
the source.

This is the ONLY finding for this session. Every proposal you emit must be a
direct application of THIS technique — not a generic optimization, not a different
technique you happen to know. If a candidate doesn't admit a faithful application
of this technique after you read the code, emit zero proposals for it and say so
in `skip_reason`.

## Candidates the mapping pass tied to this finding
For each candidate the mapping stage rated at confidence ≥ {map_min_confidence}
(read the code via the repo before proposing anything):

{for each c in mapped_candidates:}
- candidate_id:       {c.id}
- file:               {c.file}
- line range:         {c.line_start}-{c.line_end}
- mapping_confidence: {c.confidence}   # one of "low", "medium", "high"
- mapping_reasoning (from Stage 3a — may be wrong, verify):
  """
  {c.mapping_reasoning}
  """
- rationale (from Stage 1 candidate discovery):
  """
  {c.candidate_rationale}
  """

## Repository access
You have read-only access to the repo via `--add-dir`. Open the files for each
candidate above and verify the technique applies before writing proposals; if
the code shape doesn't fit, drop the candidate (emit `proposals: []` with a
`skip_reason`).

## Task — for each mapped_candidate
Produce 0 to 5 ranked proposals. Each proposal must:
- be a faithful application of THIS finding's technique to THIS code location
- be implementable as a localized change (single file or small set of files
  near this location); reject ideas that require a full system redesign
- give an `expected_impact` estimate with an honest `evidence_strength` ∈
  {high, medium, low} (high requires the cited source to report concrete
  benchmarks AND for those benchmarks to translate plausibly to this codebase —
  not just to the paper's setup; if you didn't open the URL, cap at medium)
- list `prerequisites` and `risk` ∈ {low, medium, high}
- estimate `effort_estimate`: XS (<1h), S (≤1d), M (≤1w), L (>1w)

Order proposals within each candidate by
  (expected_impact_magnitude × evidence_strength / effort). Best first.

Across candidates, order `candidate_proposals[]` by the strongest proposal in
each (best candidate first) — this helps the maintainer reading the report.

## Output (REQUIRED — strict JSON, no prose outside)
{
  "finding_id": "{finding_id}",
  "candidate_proposals": [
    {
      "candidate_id": "cand-XXXX",
      "proposals": [
        {
          "rank": 1,
          "title": "...",
          "description": "...",
          "expected_impact": {"metric": "...", "estimate": "...", "evidence_strength": "high|medium|low"},
          "prerequisites": ["..."],
          "effort_estimate": "XS|S|M|L",
          "risk": "low|medium|high"
        }
      ],
      "skip_reason": null
    },
    {
      "candidate_id": "cand-YYYY",
      "proposals": [],
      "skip_reason": "Read scheduler.py:142-211; the technique assumes a fixed-capacity request pool but this code path supports unbounded growth — not a faithful application."
    }
  ]
}

Match the full schema at @finding_changes.schema.json.
````

**Orphan-sweep prompt** (`prompts/stage3b_changegen_orphan_sweep.md`, only used when `--orphan-sweep` is set): same structure but with no `finding`; one orphan candidate per session; the model is told explicitly to apply general performance heuristics for the candidate's code location (read the file at `line_start..line_end` and infer the bottleneck type) and to mark proposals with `from_finding_id: null` and `evidence_strength: low` unless it can cite a concrete prior art URL. (This pass overlaps in spirit with Stage 3c, but stays per-candidate-single-agent and only runs against orphans; Stage 3c is the broader, dual-agent pass that runs against every candidate.)

### 7.4 Novel proposals (Stage 3c) — per-candidate, NOT-in-findings

Stage 3b is grounded research: every proposal cites one finding (and runs dual-agent per finding for cross-agent agreement). Stage 3c is the complement — for each candidate, run a coding agent and ask it to propose optimization changes that are **not** already covered by any Stage-2 finding. Findings are passed in as a *negative* list: "do not re-propose anything that overlaps with these techniques."

**Why a separate pass.** Stage 3b is bounded by Stage 2's recall — a technique not surfaced by deep research will never appear in `records[]` no matter how obvious it is from the code. Stage 3c puts the agent's own performance priors back in the loop without contaminating the research-grounded record. Sources of novel ideas typical for this pass: language/runtime micro-optimizations (logging-on-hot-path, redundant `dict.get` chains, `str.format` vs `%s`), framework-version-specific tricks (FSDP vs DDP knobs not in the finding's paper, `torch.compile` modes), repo-specific patterns the agent reads in the surrounding code (an existing helper that's faster, a config flag that's silently expensive). These rarely have publishable papers and are exactly what deep research misses.

**Fan-out shape.** Per candidate, one Claude Code subprocess and one Codex subprocess run in parallel (`--stage3c-mode dual`, default). Each session is fed:
- the candidate (`id`, `file`, `line_start`, `line_end`, `rationale`),
- the **full** `findings.json` as a **negative list** — explicitly framed as "techniques to NOT propose, because Stage 3b already handles them,"
- read-only repo access via `--add-dir`.

Optional cheaper mode `--stage3c-mode alternating`: one agent per candidate, round-robin by candidate index; loses the cross-agent agreement signal but halves cost.

```
INPUT:  candidates.json, findings.json, repo
OUTPUT: novel_records[] (one record per (candidate, agent) pair)

semaphore = asyncio.Semaphore(--max-parallel-agents)             # default 4

async def novel_one(cand, agent_choice):
    async with semaphore:
        invoke agent_choice with:
            - --add-dir <repo-path>                              # read-only
            - stdin: STAGE3C_NOVEL_PROMPT(cand, findings)        # findings = negative list
            - timeout: --per-candidate-novel-wallclock-s         # default 150s
        returns: {candidate_id, agent, proposals[]}

# dual mode (default)
tasks = [novel_one(c, a) for c in candidates for a in ("claude_code", "codex")]
# alternating mode
agents_cycle = itertools.cycle(["claude_code", "codex"])
tasks = [novel_one(c, next(agents_cycle)) for c in candidates]

results = await asyncio.gather(*tasks, return_exceptions=True)
novel_records = dedupe_and_merge(results, findings)              # see "Dedupe" below
```

**Budgets.**
- Wallclock per (candidate, agent) session: `--per-candidate-novel-wallclock-s` (default 150s).
- Cost per session: `--per-candidate-novel-budget-usd` (default $0.30).
- Parent stage budget: `--stage3c-budget-usd` (default $10 for dual mode at ~12 candidates × 2 agents × $0.30 with headroom; halved automatically for alternating mode).
- Stall: `--per-candidate-novel-stall-s` (default 75s).

**Dedupe (three layers).** The merge step is where Stage 3c earns its keep:

1. **Against findings (mandatory).** Each emitted proposal is scored for overlap with every Stage-2 finding by string-similarity over the `(proposal.title + proposal.description)` blob vs the `(finding.title + finding.technique_summary)` blob — fast first pass — then any borderline case (similarity 0.3–0.7) is sent to a single LLM-judge call via the LiteLLM proxy: "Does proposal X re-derive the technique described in finding Y?" Reject the proposal if yes, recording it in `stage3c.dedupe.rejected_overlap_with_findings`. The model also self-reports via `novelty_check.overlaps_finding_ids[]` in its output; the self-report is used as a prefilter but never trusted alone.
2. **Cross-agent (dual mode only).** When both Claude Code and Codex produce a proposal for the same `candidate_id` with title similarity ≥ 0.7 (or LLM-judge agreement on borderline), they are kept as **one** proposal with `agreed_with_other_agent: true`. The other is dropped and counted in `stage3c.dedupe.rejected_cross_agent_duplicate`. Agreement is a positive quality signal that survives into the final ranking.
3. **Against Stage 3b within the same candidate.** Before merging into `by_candidate[]`, run the same overlap check against research-grounded proposals already emitted for that candidate. Reject any novel proposal that overlaps a research-grounded one (the research-grounded one wins because it cites a source).

**Validation.** Parse each session's result with the `novel_changes` pydantic schema. Drop any proposal whose `novelty_check.overlaps_finding_ids[]` self-reports a finding the candidate was mapped to in `mapping.json` *unless* the dedupe LLM-judge disagrees with the self-report. On parse failure: retry once with strict reminder; on second failure record `{candidate_id, agent, proposals: [], error: "schema_invalid"}` and continue.

**Why dual is the default.** This is the *novel* pass — proposals are unbacked, so calibration is harder than for Stage 3b. Running two independent agents lets `agreed_with_other_agent: true` serve as a poor-man's evidence_strength bump; without agreement, a single agent's confident-sounding novel suggestion is the prime hallucination target. M3 ablation compares dual vs alternating on proposal accept-rate (LLM-judge or maintainer-rated).

#### 7.4.1 Stage 3c prompt (drop-in)

`prompts/stage3c_novel.md`:

````
You are proposing optimization changes for ONE code location. Another pass of the
pipeline already covers proposals derived from a curated list of research findings
(papers / blogs / PRs / issues). Your job is to propose changes the research pass
will NOT produce — language- / runtime- / framework- / repo-specific tricks that
sit BELOW the abstraction level of published techniques, or that the research pass
simply did not surface.

## Code location
- candidate_id: {candidate_id}
- file: {file}
- line range: {line_start}-{line_end}
- rationale from candidate discovery:
  """
  {candidate_rationale}
  """
- excerpt (read the surrounding code in the repo via --add-dir before proposing):
  ```{lang}
  {code_excerpt}
  ```

## Findings — DO NOT RE-PROPOSE THESE (negative list)
The following research findings are already being applied to this codebase by a
separate pass. You MUST NOT propose any change that is materially the same as
one of these techniques. If the most obvious change at this location is one of
these, emit zero proposals here — that's a valid result.

{for each finding f in findings:}
- finding_id:   {f.id}
- title:        {f.title}
- url:          {f.url}
- summary:      {f.technique_summary}

(If a summary is too thin to judge overlap, fetch the `url` — you must not
re-derive a technique just because its summary in this list was terse.)

## What "novel" means here
- A different abstraction level (e.g., `logger.debug` arg formatting, redundant
  list-comprehension allocations, `dict.setdefault` vs `defaultdict`, branch
  hoisting out of a hot loop, `__slots__` on a per-step dataclass) — usually NOT
  in a paper.
- A repo-specific observation: an existing helper you found in the codebase that
  is faster than the current call site; a config flag that quietly turns on an
  expensive feature; a redundant copy between two layers.
- A framework- / runtime-version specific trick that the negative-list findings
  don't mention (e.g., `torch.compile(mode="reduce-overhead")`, FSDP `use_orig_params=False`).

A proposal is NOT novel if any negative-list finding's `technique_summary` (or
the source it points to via `url`) already covers it, even with different
wording. When in doubt, do not emit. Better zero proposals than a duplicate of a
research-grounded one.

## Task
Produce 0 to 5 ranked proposals that are novel by the rule above. Each proposal must:
- be implementable as a localized change at this location
- name in `novelty_check.overlaps_finding_ids[]` any finding you considered close
  and explain in `novelty_check.why_distinct` why you still think it's distinct
  (the merge step will second-guess you with an LLM judge)
- give `expected_impact` with honest `evidence_strength` ∈ {high, medium, low}.
  Without a citation, `evidence_strength: high` is rarely justified; default to
  `medium` for repo-specific reads you verified in the code, `low` for general
  heuristics.
- list `prerequisites`, `risk` ∈ {low, medium, high}, and `effort_estimate`
  ∈ {XS, S, M, L}.

Order by (expected_impact_magnitude × evidence_strength / effort).

## Output (REQUIRED — strict JSON, no prose outside)
{
  "candidate_id": "{candidate_id}",
  "agent": "claude_code | codex",
  "negative_finding_ids": [/* the finding ids you read as the negative list */],
  "proposals": [
    {
      "rank": 1,
      "source": "agent_novel",
      "from_finding_id": null,
      "title": "...",
      "description": "...",
      "expected_impact": {"metric": "...", "estimate": "...", "evidence_strength": "high|medium|low"},
      "prerequisites": ["..."],
      "effort_estimate": "XS|S|M|L",
      "risk": "low|medium|high",
      "novelty_check": {
        "overlaps_finding_ids": ["find-XXXX"],
        "why_distinct": "Even though find-XXXX talks about ring buffers in general, my proposal targets the logging fast-path, not the allocator."
      }
    }
  ]
}

Match the full schema at @novel_changes.schema.json.
````

### 7.5 v2: produce a runnable patch?

Deferred to v2. The per-finding Stage 3b prompt could be extended with "and produce a unified diff implementing the rank-1 proposal for each non-skipped candidate" gated by `--gen-patch`, validated with `git apply --check`. Per-finding framing is actually friendlier to patch generation than per-candidate would be — the model already has the technique loaded and just specializes it per file. The cost roughly doubles (output tokens grow significantly) and the failure surface area (compile errors, hallucinated APIs) explodes; KernelBench results suggest 30–50% of LLM-generated kernels fail to compile even with explicit signatures. Plan: add only after M6 closes the eval loop with measured improvement on at least 3 modules.

---

## 8. Component Layout

```
optquest/
├── pyproject.toml                   # uv-managed; pins listed in §1
├── README.md
├── optquest/
│   ├── __init__.py
│   ├── cli.py                       # Typer/Click CLI entrypoint
│   ├── config.py                    # YAML/env loading, budget caps, model pins
│   ├── paths.py                     # ~/.cache/optquest/<repo_slug>/<run_id>/ layout
│   ├── schema/                      # pydantic schemas + JSON Schema export
│   │   ├── candidates.py
│   │   ├── findings.py
│   │   ├── mapping.py
│   │   ├── finding_changes.py       # Stage 3b per-session result schema
│   │   ├── novel_changes.py         # Stage 3c per-(candidate,agent)-session result schema
│   │   └── changes.py               # final changes.json (records[] + novel_records[] + by_candidate[])
│   ├── agents/
│   │   ├── base.py                  # AgentRunner protocol (run, telemetry)
│   │   ├── claude_code.py           # subprocess wrapper (stream-json, plan mode)
│   │   ├── codex.py                 # subprocess wrapper (exec --json --output-last-message)
│   │   ├── env.py                   # _clean_env lifted from CORAL; strips OPENAI_* before
│   │   │                            #   spawning agent subprocesses
│   │   └── stream.py                # NDJSON tailing thread, telemetry extraction
│   ├── proxy/
│   │   ├── client.py                # thin wrapper around openai.OpenAI(base_url=LITELLM_URL)
│   │   ├── embeddings.py            # /v1/embeddings call, batching, retries
│   │   └── coerce.py                # markdown → findings.json coercion call
│   ├── stage1/
│   │   ├── runner.py                # bootstrap + alternating review loop, stop conds
│   │   ├── scope.py                 # modules.json / profile-artifact discovery
│   │   ├── validate.py              # path existence, line-range sanity
│   │   └── prompts/                 # stage1_bootstrap.md, stage1_review.md
│   ├── stage2/
│   │   ├── gpt_researcher_runner.py # configures env, runs GPTResearcher, captures md
│   │   ├── retriever_config.py      # tavily | searxng | custom | duckduckgo selection
│   │   ├── coerce_runner.py         # invokes proxy/coerce.py with stage2 prompt
│   │   ├── validate.py              # URL reachability, schema parse, min-count check
│   │   └── prompts/stage2_research.md, stage2_coerce.md
│   ├── stage3/
│   │   ├── mapping/
│   │   │   ├── per_finding.py       # per-finding subprocess fan-out + semaphore + round-robin
│   │   │   ├── invert.py            # per-finding edges -> candidate-centric mapping (no top-K)
│   │   │   ├── prefilter_bm25.py    # optional pre-filter for >100-candidate sets (off by default)
│   │   │   ├── validate.py          # per-finding schema parse, ref-integrity, confidence coerce
│   │   │   └── prompts/stage3a_map.md
│   │   ├── changegen/
│   │   │   ├── per_finding.py       # per-finding dual/alternating fan-out (Claude Code + Codex)
│   │   │   ├── dedupe.py            # cross-agent dedupe within (finding_id, candidate_id)
│   │   │   ├── orphan_sweep.py      # optional per-orphan-candidate fallback pass (single-agent)
│   │   │   ├── pivot.py             # records[] + novel_records[] -> by_candidate re-pivot
│   │   │   ├── validate.py          # finding_changes schema parse, ref-integrity
│   │   │   └── prompts/             # stage3b_changegen_per_finding.md, stage3b_changegen_orphan_sweep.md
│   │   ├── novel/
│   │   │   ├── per_candidate.py     # per-candidate dual/alternating fan-out (Codex + Claude Code)
│   │   │   ├── dedupe.py            # 3-layer dedupe: vs findings (LLM-judge), cross-agent, vs 3b
│   │   │   ├── validate.py          # novel_changes schema parse + novelty_check sanity
│   │   │   └── prompts/stage3c_novel.md
│   │   └── validate.py
│   ├── orchestrator.py              # asyncio driver: stage1 || stage2; then stage3
│   ├── telemetry.py                 # JSONL cost/wallclock logger, budget enforcement
│   └── reporting.py                 # human-readable run summary, markdown export
├── tests/
│   ├── test_schemas.py
│   ├── test_stage1_loop.py          # mocked agent subprocess
│   ├── test_stage2_gpt_researcher.py # recorded fixture against mocked proxy
│   ├── test_stage3_mapping.py
│   └── test_e2e_smoke.py
└── examples/
    ├── repos.yaml                   # eval set
    └── runs/                        # frozen golden-output diffs
```

---

## 9. CLI Sketch

```bash
# Default: parallel stages, GPT Researcher Stage 2, claude_code↔codex review, K=5.
# Reads OPENAI_API_KEY and OPENAI_BASE_URL from env (LiteLLM proxy).
export OPENAI_API_KEY="…"
export OPENAI_BASE_URL="https://ete-litellm.ai-models.vpc-int.res.ibm.com"
export OPTQUEST_MODEL="<MODEL_NAME>"
export OPTQUEST_EMBEDDING_MODEL="<EMBEDDING_MODEL_NAME>"
export OPTQUEST_RETRIEVER="tavily"            # or searxng | duckduckgo | custom
export TAVILY_API_KEY="…"                     # if RETRIEVER=tavily

optquest run \
  --repo https://github.com/owner/repo \
  --module src/foo/bar \
  --scope-paths 'src/foo/bar/**/*.py' \
  --stage1-budget-usd 5 --stage2-wallclock-s 1800 \
  --mapping-agent alternating \
  --per-finding-wallclock-s 90 --per-finding-budget-usd 0.20 \
  --stage3b-mode dual \
  --per-finding-changegen-wallclock-s 180 \
  --per-finding-changegen-budget-usd 0.50 \
  --stage3c-mode dual \
  --per-candidate-novel-wallclock-s 150 \
  --per-candidate-novel-budget-usd 0.30 \
  --stage3a-budget-usd 4 --stage3b-budget-usd 16 --stage3c-budget-usd 10 \
  --max-parallel-agents 4 \
  --out ~/.cache/optquest/owner__repo/2026-05-13--ab12

# Sequential to ease debugging; pin every agent choice to one model
optquest run --repo . --module pkg/core \
  --sequential \
  --mapping-agent claude_code \
  --stage3b-mode claude_code \
  --stage3c-mode alternating \
  --prefilter none

# Skip Stage 3c entirely (research-grounded only)
optquest run --repo . --module pkg/core --no-stage3c

# Also fill orphan candidates with general-heuristics fallback proposals
optquest run --repo . --module src/foo/bar \
  --orphan-sweep \
  --orphan-sweep-budget-usd 2

# Use a profiling artifact to bound scope; longer research budget
optquest run --repo . --module src/scheduler \
  --profile bench/profile.json \
  --scope-from profile \
  --stage2-wallclock-s 3600

# Resume a partially-completed run (re-uses cached stage outputs that validated)
optquest resume --run-dir ~/.cache/optquest/owner__repo/2026-05-13--ab12

# Just validate an existing run directory's artifacts
optquest validate --run-dir ~/.cache/optquest/owner__repo/2026-05-13--ab12

# Stage 2 alone (useful during retriever bring-up)
optquest stage2 --module . --out tmp/ --retriever duckduckgo
```

---

## 10. Still-Open Questions

1. **Exact CORAL paths.** The CORAL public repo's README places the relevant code at `coral/agent/runtime.py` and `coral/workspace/setup.py` (Architecture section). **Resolution needed before implementation**: confirm the exact tree to lift from.
2. **`claude --add-dir` vs `additionalDirectories` semantics in non-interactive mode.** The docs document `additionalDirectories` in `settings.json` (code.claude.com/docs/en/permission-modes) and shipyard.build/claude-code-cheat-sheet shows `--add-dir` working in CI; but the exact relationship between argv `--add-dir`, `settings.json:permissions.additionalDirectories`, and `--cwd` under `--print` is not crisply documented. **Resolution**: empirical smoke test in milestone M0.
3. **Codex `--output-schema` enforcement strength.** Docs say "Codex validates tool output against it" — unclear whether the validator is hard (rejects + retries internally) or soft (just attaches schema to the system prompt). **Resolution**: M0 test; if soft, we layer pydantic on top.
4. **Stage-2 URL reachability standard.** Some papers sit on arxiv preprints that move; some PR URLs become 404 after force-pushes. Threshold for "reachable" needs to be a policy decision — current plan: HTTP 200 within 10s, ≥1 KB body, no detected paywall block-page heuristics. **Resolution**: an `--allow-unreachable-urls` escape hatch with a warning.
5. **Whether the LiteLLM proxy returns `usage` blocks / cost data on chat completions.** LiteLLM by default does return `usage.prompt_tokens` and `usage.completion_tokens`, but cost (`response_cost`) only when the deployment is configured for it. **Resolution**: probe the proxy at M0; if cost is unavailable, set `cost_unknown: true` and rely on `--stage2-wallclock-s` as the primary budget guard.
6. **Whether `<MODEL_NAME>` (to be filled in) supports `response_format=json_schema`.** OpenAI's gpt-4o-2024-08-06 and later support strict JSON-schema mode; LiteLLM passes this through for compatible upstream models but silently degrades for others. **Resolution**: probe at M0; coercion call falls back to `response_format={"type":"json_object"}` + pydantic validation + retry on failure.
7. **Whether alternating Claude-Code / Codex materially improves quality over a single-agent default, in both Stage 1 (iterative critique) and Stage 3a (per-finding fan-out, alternated across findings).** No public benchmark for "performance candidate discovery" or "finding↔candidate applicability mapping". **Resolution**: ablations in M3 of the eval. Stage 1: Claude-only vs Claude+1-Codex-review vs full alternating. Stage 3a: `--mapping-agent claude_code` vs `--mapping-agent codex` vs `--mapping-agent alternating` — measure orphan rate, edge-precision against a small hand-labeled set, per-finding cost spread.

---

## 11. Milestones

Each milestone is verifiable by a specific command + acceptance check.

**M0 — Smoke test (Day 0–1).** Verify both CLIs and the LiteLLM proxy work end-to-end against trivial inputs. Probe the open questions above.
- Command: `python -m optquest.agents.claude_code --selftest && python -m optquest.agents.codex --selftest && python -m optquest.proxy.client --selftest`
- The proxy selftest does three probes:
  1. `chat.completions.create(model=<MODEL_NAME>, messages=[…])` — verifies LLM works and reports whether `usage` is populated (resolves Q6).
  2. Same call with `response_format={"type":"json_schema", "json_schema":{…}}` — verifies strict-mode support (resolves Q7).
  3. `embeddings.create(model=<EMBEDDING_MODEL_NAME>, input=["test"])` — verifies embeddings endpoint exists.
- Pass: all three CLI selftests write a 1-line JSON to stdout with `{"ok": true, …}`; proxy selftest reports `usage_supported`, `json_schema_supported`, `embeddings_supported` booleans.

**M1 — Schemas + Stage 2 alone (Week 1).** All four pydantic schemas finalized, with `model_json_schema()` export. Stage 2 callable end-to-end on a small module via GPT Researcher.
- Prereqs: a retriever is selected and credentialed (resolves Q5).
- Command: `optquest stage2 --module . --out tmp/`
- Pass: `tmp/findings.json` validates; ≥5 findings; ≥80% of URLs HEAD return 200; markdown→JSON coercion succeeds first-try ≥80% of runs.

**M2 — Stage 1 alone (Week 2).** Bootstrap + 2 review iterations on a known module (e.g., a small Python package).
- Command: `optquest stage1 --repo . --module optquest/stage3/mapping --out tmp/`
- Pass: `tmp/candidates.json` validates; ≥5 candidates; every `file` exists; stop condition triggered within 4 iters.

**M3 — Mapping + change generation + novel proposals (Week 3).** Stage 3a (per-finding fan-out, alternating), Stage 3b (per-finding dual fan-out: Claude Code + Codex per finding, cross-agent dedupe), and Stage 3c (per-candidate dual-agent novel proposals) on artifacts from M1 and M2.
- Command: `optquest stage3 --candidates tmp/candidates.json --findings tmp/findings.json --out tmp/`
- Pass:
  - `tmp/mapping.json` and `tmp/changes.json` validate against their pydantic schemas (changes.json against `changes.py` with `records[]`, `novel_records[]`, and `by_candidate[]` all populated).
  - Stage 3a: ≥ 90% of per-finding subprocesses succeed (no `schema_invalid` / `stalled` / timeout); `meta.drops.schema_invalid + meta.drops.stalled ≤ 10% of findings_total`.
  - Stage 3a round-robin assignment is balanced: |`agents_used.claude_code` − `agents_used.codex`| ≤ 1.
  - Stage 3a parent budget `--stage3a-budget-usd $4` not exceeded; per-finding p95 wallclock ≤ 90s.
  - Final orphan rate ≤ 30%.
  - Stage 3b: ≥ 90% of per-(finding, agent) subprocesses succeed; `stage3b.sessions_failed / sessions_total ≤ 10%`. In `--stage3b-mode dual` (default), every finding has at least one session per agent: `|sessions_by_agent.claude_code − sessions_by_agent.codex| = 0` over the set of findings with ≥1 edge. Every finding's merged record emits ≥1 candidate-proposal block whose `proposals[]` is either non-empty *or* has a non-null `skip_reason` (no silent empties) — *unless* both agent sessions failed, in which case `error` is recorded.
  - Stage 3b cross-agent agreement is real but not unanimous: of the merged proposals, **between 25% and 75%** carry `agreed_with_other_agent: true`. A value outside that band signals one of: (a) the dedupe similarity threshold is mis-tuned (>75% agreed often means too-loose collapse), (b) the two agents are seeing the prompt very differently (<25% agreed often means one agent is malformed / stalling — verify per-agent success rates).
  - Stage 3b parent budget `--stage3b-budget-usd $16` not exceeded in dual mode; per-session p95 wallclock ≤ 180s; mean per-session cost ≤ $0.50 (if cost reporting is available; else mean per-session wallclock ≤ 120s).
  - Stage 3c: ≥ 90% of per-(candidate, agent) subprocesses succeed. In `--stage3c-mode dual` (default), every candidate has at least one session per agent — `|sessions_by_agent.claude_code − sessions_by_agent.codex| = 0`.
  - Stage 3c parent budget `--stage3c-budget-usd $10` not exceeded; per-session p95 wallclock ≤ 150s.
  - Stage 3c dedupe is doing real work: `dedupe.rejected_overlap_with_findings ≥ 1` (at least one model-proposed novel was caught colliding with a finding — if it's zero across a 20-finding run, the LLM-judge is likely broken or being skipped).
  - Stage 3c novelty: of proposals that survived dedupe, ≤ 15% are flagged by a post-hoc audit (LLM-judge run blind over `(novel_proposal, full findings.json)` pairs) as actually overlapping a finding. False-novel rate above 15% fails the milestone — re-tighten the dedupe LLM-judge threshold and re-run.
  - Re-pivot integrity: every proposal in `by_candidate[]` carries a `source` discriminator; `source: "research_grounded"` proposals match a `(finding_id, candidate_id)` pair in `records[]`; `source: "agent_novel"` proposals match an `(agent, candidate_id)` pair in `novel_records[]`. Every `agreed_with_other_agent: true` proposal in 3b's merged records traces back to two underlying sessions in `per_agent_sessions[]`.
  - Ablation 1 (mapping agent): run with `--mapping-agent claude_code` and `--mapping-agent codex` on one eval repo; record orphan rate and edge-set Jaccard vs the `alternating` default (resolves §10 Q7 for mapping).
  - Ablation 2 (Stage 3b dual vs single-agent): on the same eval repo, run `--stage3b-mode claude_code` and `--stage3b-mode codex` (single-agent variants). Compare top-3 proposal quality per finding via LLM-judge against the dual-mode merged output. Compare cost. Feeds the decision on whether dual stays the default beyond M3.
  - Ablation 3 (per-finding vs per-candidate Stage 3b): on the same eval repo, also run a per-candidate variant of Stage 3b (one session per candidate, fed top-K findings) and compare top-3 proposal quality via LLM-judge — feeds the decision on whether to keep per-finding as the fan-out unit beyond M3.
  - Ablation 4 (Stage 3c dual vs alternating): run the same eval repo with `--stage3c-mode alternating`. Compare proposal accept-rate (LLM-judge or quick maintainer-rated sample) between dual and alternating; the cost-quality tradeoff feeds the decision on whether dual stays the default.

**M4 — End-to-end parallel run (Week 4).** Full pipeline with Stage 1 || Stage 2.
- Command: `optquest run --repo <small-test-repo> --module <subpath> --out tmp/`
- Pass: wall-clock < max(stage1, stage2) + 60s; all four artifacts validate; pipeline finishes within configured budget caps.

**M5 — Eval on golden repos (Week 5–6).** Run pipeline on a fixed eval set; compare against ground-truth wins.
- **Eval set**: 3 repos with rich, measured optimization PR histories — e.g., vLLM (paged attention, chunked prefill, prefix caching), PyTorch (FlashAttention integrations, torch.compile inductor passes), llama.cpp (quantization kernels). For each, freeze a `before_sha` and a `golden_winning_prs.csv` of merged optimization PRs landed after that sha.
- **Metric 1 — Candidate recall@10**: fraction of golden PRs whose changed file is present in our top-10 candidates.
- **Metric 2 — Proposal alignment**: BLEU-style or LLM-judge similarity between our top-ranked proposal title/description and the golden PR title/body. Target: ≥40% of golden PRs match a proposal with LLM-judge score ≥ 0.7. The judge runs through the same LiteLLM proxy.
- **Metric 3 — Wallclock & budget**: end-to-end run completes within `--stage1-budget-usd`, `--stage2-wallclock-s`, `--stage3-budget-usd`.
- **Baselines**: (a) single Claude Code call with the bootstrap prompt only, no review loop, no findings; (b) SWE-agent (https://swe-agent.com) or OpenHands (https://github.com/All-Hands-AI/OpenHands) with prompt "find optimization opportunities and propose changes".
- Pass: beats both baselines on Metric 1 by ≥10 points and Metric 2 by ≥10 points on at least 2 of 3 eval repos.

**M6 — Prior-art-anchored eval (Week 7).** Cross-validate proposals against published perf-eval benchmarks.
- **Benchmarks consulted:**
  - **KernelBench** (https://scalingintelligence.stanford.edu/blogs/kernelbench/) — 250 GPU-kernel tasks, ideal for modules whose candidates are kernel launch sites. Used to measure whether our proposals, when hand-applied, generate kernels that compile + pass numerical correctness + beat baseline latency.
  - **PIE / pie4perf** (https://pie4perf.com, https://arxiv.org/abs/2302.07867) — 77K C++ performance-improving edits from competitive-programming submissions on a gem5-simulated CPU. Best fit for CPU-bound algorithmic candidates. PerfCoder, FasterPy, TritonForge, and MaxCode all build on PIE — useful exemplar pipelines to compare against in the writeup.
  - **CodePerf / "code performance benchmarks"**: there is no single canonical "CodePerf" benchmark widely adopted as of 2026-05-13. The closest umbrella is the **Mercury benchmark** (used alongside PIE in FasterPy) — explicitly noted as a deferred enrichment.
  - **EvalPlus's perf track**: EvalPlus (https://evalplus.github.io) is canonical for *correctness* of LLM-generated code (HumanEval+, MBPP+); it does **not** currently publish a perf-focused track. We note this gap and adopt KernelBench + PIE as the perf equivalents instead.
- **Loop**: pick a candidate's proposal, hand-apply, measure delta against the relevant benchmark harness, log to `eval/results.csv`.
- Pass: ≥3 hand-applied proposals from M5 runs yield measured improvement (>5% on the relevant metric, with significance vs noise floor).

**Mfinal — Reference-repo demonstration (Week 8).** Pipeline produces measurably useful output on one full reference repo (e.g., vLLM `vllm/attention`) end-to-end, with a written report (`reporting.py` markdown export) suitable for a maintainer to read in 15 minutes.
- Pass: a maintainer-rated rubric (5 maintainers, ≥3 of them: "I would investigate this") on the top-3 proposals across ≥5 candidates.

---

## 12. Non-Goals

- **Writing the implementation** (this is a plan only).
- **Auto-applying patches** to the target repo. Stage 3 emits proposals; v2 may emit patches; this plan does neither apply nor PR.
- **Multi-repo or multi-module batch mode.** One repo × one module per run.
- **Replacing the pipeline with a single existing agent** (SWE-agent, OpenHands, Aider). Evaluated as a baseline in M5; not adopted.
- **Building a UI.** The CORAL `ui` is skipped; a plain markdown report is the deliverable. Re-evaluate in v2.
- **Cross-language IR-level analysis.** We rely on the agent's textual understanding of code + optional profiler artifacts; we do not build AST or LLVM-IR passes.
- **Multi-provider Stage 2.** Out of scope — GPT Researcher only.
- **Image/vision input to Stage 2.** Stage 2 is text-only; vision could be added in v2 for flame-graph screenshots.

---

## 13. Possible Future Optimizations

1. **Session resume across iterations** (Claude `--resume <uuid>`; Codex `codex exec resume`). Cuts Stage 1 tokens by 20–40%. Adds: session-id loss handling, drift-detection on resume.
2. **Patch generation in Stage 3b.** Append a "now produce a unified diff for proposal #1" step gated by `--gen-patch`. Validate by `git apply --check` against the pinned sha.
3. **Auto-benchmark closing the loop.** For each generated patch, attempt compile + run + benchmark in a worktree (CORAL's worktree pattern earns its keep here). Auto-rank proposals by measured Δ.
4. **OpenCode as a third agent** in the alternating review. CORAL already supports it; adding it just needs another `AgentRunner` subclass.
5. **Module summarization cache.** Stage-2's `module_summary` is recomputed each run; persist it keyed by `(repo, sha, module_path)`.
6. **Cross-repo learning.** Maintain a persistent `findings_library.parquet` of validated findings across runs — Stage 2 then becomes "retrieve from library, top-up from web". Cuts Stage-2 wallclock by 50–80% after the library warms.
7. **LLM-as-judge calibration.** Run the eval (M5) periodically and refit the LLM-judge weighting in the mapping rerank against the human rubric — current 0.7/0.3 blend is a guess.
8. **MCP integration.** Both Claude Code and Codex speak MCP; expose an `optquest-mcp` server so other agents (Claude Desktop, OpenAI Apps) can invoke the pipeline as a tool. GPT Researcher already provides `gptr-mcp` (https://github.com/assafelovic/gptr-mcp) as a reference implementation.
9. **Profiling-artifact-driven scope.** Tighter integration with `py-spy`, `nsys`, `pprof` — parse stack traces, weight candidates by self-time. PerfCoder, FasterPy, TritonForge, and MaxCode (cited in §11 M6) all suggest profiling-aware retrieval beats blind retrieval.
10. **Replace the deep-research call with a learned researcher.** Once we have a few hundred (candidate, finding) pairs from runs, fine-tune a small model (served via the same LiteLLM proxy) to do retrieval directly — cheaper than running GPT Researcher every time.
11. **A `coral ui`-style dashboard** that streams the alternating-review loop live with per-iteration diffs of `candidates.json` — re-uses CORAL's web/ for free.
12. **Vision input** — feed flame-graph PNGs from `py-spy` or NVTX traces into Stage 1 or Stage 2 if the proxy's model is multimodal.
