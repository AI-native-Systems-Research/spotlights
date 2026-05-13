# Implementation Plan: Evidence-Backed Code Optimization Pipeline

> **Status:** Plan only — no implementation code.
> **Pipeline name (placeholder):** `optquest`. Replace freely.
> **Fast-changing items carry a "last verified: 2026-05-13" tag.** The CLIs have shipped multiple breaking flag changes in the past 12 months; re-verify before implementation if more than ~30 days have passed.
> **Stage 2 provider — pinned by user directive: GPT Researcher only**, configured against an OpenAI-compatible LiteLLM proxy. See §6 for the exact configuration; all multi-provider tiering removed.

---

## 1. Goal

Build a Python pipeline that, given `(repo_url_or_path, module_path)`, produces `changes.json` — a list of optimization proposals tied to concrete code locations in the target module, where each proposal is backed by (a) at least one finding from external research (paper / blog / PR / issue / talk) and (b) explicit reasoning emitted by a coding agent (Claude Code or Codex). The pipeline runs Stage 1 (candidate discovery via Claude↔Codex alternating review) and Stage 2 (single GPT Researcher call) in **parallel**, then Stage 3 (mapping + per-candidate change generation) consumes both.

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
| Stage 1 / Stage 3b agent models | Whatever Claude Code / Codex CLIs default to; pin explicitly via wrapper config (`--model` for `claude`, `-c model='"…"'` for `codex`) | Decoupled from the rest of the model surface; user controls via their proxy/account |
| **Stage 2 — GPT Researcher** | `gpt-researcher >= 0.13` (library mode) | **Sole Stage-2 provider — user directive** |
| **Stage 2 — LLM endpoint** | OpenAI-compatible HTTP endpoint at `OPENAI_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com` (LiteLLM proxy). Model name `<MODEL_NAME>` — placeholder in user-provided snippet, to be filled in | User-mandated |
| **Stage 2 — retriever (search backend)** | **OPEN** — Tavily / Serper / Exa / SearXNG / Brave / DuckDuckGo. Must be reachable from inside the VPC. See §11 Q5. | GPT Researcher requires a retriever **separately** from the LLM endpoint |
| Embeddings (Stage 3a) | Default: an embedding model served by the same LiteLLM proxy (e.g., a `text-embedding-3-large`-compatible deployment); self-hosted fallback `BAAI/bge-m3` (1024-d) via `sentence-transformers` | Keeps everything behind one endpoint; voyage-code-3 dropped because it requires Voyage SaaS egress that may not work from the VPC |
| Python | `>= 3.11` (matches GPT Researcher's runtime requirement) | |
| Key libs | `pydantic >= 2.7`, `rank-bm25 >= 0.2.2`, `openai >= 1.40` (the HTTP client for the LiteLLM proxy), `gpt-researcher >= 0.13`, `httpx >= 0.27`, `anyio >= 4`, `sentence-transformers >= 3` (only for bge-m3 fallback) | Anthropic SDK and Voyage SDK removed |

---

## 2. Inputs & Outputs

All four artifacts live under `~/.cache/optquest/<repo_slug>/<run_id>/`. All schemas are checked with `pydantic >= 2.7` models published in `optquest.schema`, which also exports `model_json_schema()` to disk for external validators.

### 2.1 `candidates.json` — Stage 1 output

```json
{
  "schema_version": "1.0",
  "run_id": "2026-05-13T12-00-00Z--ab12cd34",
  "target": {
    "repo": "https://github.com/owner/repo",
    "git_sha": "9f8e7d6c5b4a3210...",
    "module_path": "src/foo/bar",
    "scope_paths": ["src/foo/bar/**/*.py"]
  },
  "iterations": 3,
  "candidates": [
    {
      "id": "cand-0001",
      "file": "src/foo/bar/scheduler.py",
      "symbol": "Scheduler._dispatch_step",
      "line_start": 142,
      "line_end": 211,
      "kind": "function",
      "tags": ["hot-loop", "allocator-heavy", "python-overhead"],
      "rationale": "Called once per decode step; allocates a new list of pending requests every call; iterates O(N) over the request table to filter.",
      "evidence_pointers": [
        {"kind": "profile", "path": "bench/profile.json", "note": "12% of self-time"},
        {"kind": "issue", "url": "https://github.com/owner/repo/issues/1234"}
      ],
      "code_excerpt_sha256": "ab12...",
      "discovered_by": "claude_code",
      "confirmed_by": ["codex"],
      "confidence": 0.78
    }
  ],
  "telemetry": {
    "claude_calls": 4, "codex_calls": 4,
    "tokens_in": 312000, "tokens_out": 41000,
    "cost_usd": 2.81, "wallclock_s": 612
  }
}
```

### 2.2 `findings.json` — Stage 2 output

```json
{
  "schema_version": "1.0",
  "query_context": { "module_path": "src/foo/bar", "module_summary": "..." },
  "findings": [
    {
      "id": "find-0007",
      "title": "PagedAttention: Memory Management for LLM Serving",
      "url": "https://arxiv.org/abs/2309.06180",
      "source_type": "paper",
      "published": "2023-09-12",
      "technique_name": "Paged KV-cache with block table indirection",
      "technique_summary": "Splits KV cache into fixed-size blocks managed by an OS-style page table; eliminates contiguous-allocation fragmentation in batched decoding.",
      "claimed_gains": [
        {"metric": "throughput", "magnitude": "2-4x", "baseline": "FasterTransformer/Orca", "conditions": "OPT-13B, A100, batched serving"}
      ],
      "target_components": [
        {
          "component_hint": "KV cache allocator / block manager",
          "file_or_symbol_hints": ["cache_manager", "block_table", "kv_cache", "allocator"],
          "keywords_for_matching": ["kv cache", "paged", "block table", "fragmentation", "prefix caching"]
        }
      ],
      "suggested_changes": [
        "Introduce fixed-size block allocator backing the KV tensor",
        "Add block-table indirection in attention kernel"
      ],
      "evidence_quality": {
        "primary": true, "peer_reviewed": true, "has_benchmarks": true,
        "reproducible": true, "score": 0.92
      },
      "raw_snippets": ["..."]
    }
  ],
  "telemetry": {
    "provider": "gpt-researcher",
    "llm_endpoint": "https://ete-litellm.ai-models.vpc-int.res.ibm.com",
    "llm_model": "<MODEL_NAME>",
    "retriever": "<tavily|serper|searxng|...>",
    "tokens_in": 7800, "tokens_out": 14200, "search_calls": 23,
    "cost_usd": 0.0, "cost_unknown": true, "wallclock_s": 184
  }
}
```

> **Telemetry note:** the LiteLLM proxy may or may not surface `usage` blocks in chat completions. If it does, we record token counts and (proxy-provided) cost; if not, the orchestrator records `tokens_*` as best-effort from response payloads and sets `cost_unknown: true`. See §11 Q6.

### 2.3 `mapping.json` — Stage 3a output (consumed by Stage 3b)

```json
{
  "schema_version": "1.0",
  "k": 5,
  "method": "hybrid_shortlist+alternating_agent_review",
  "stage3a_2": {
    "iters": 2,
    "stop_reason": "edge_set_stability",
    "agents": ["claude_code", "codex", "claude_code"]
  },
  "mappings": [
    {
      "candidate_id": "cand-0001",
      "ranked_findings": [
        {
          "finding_id": "find-0007",
          "score": 0.86,
          "score_breakdown": {"bm25": 0.42, "dense": 0.91, "llm_judge": 0.88, "agent_final": 0.86},
          "reasoning": "scheduler.py touches block allocation for the KV cache; the finding's paged-attention technique applies directly to the loop at line 142",
          "source_iter": 2,
          "in_shortlist": true
        },
        {"finding_id": "find-0011", "score": 0.71, "source_iter": 0, "in_shortlist": true}
      ]
    }
  ],
  "orphans": []
}
```

Stage 3a-1 emits a sibling `shortlist.json` with the same shape minus the `reasoning` / `source_iter` / `agent_final` fields. `--mapping-review off` produces a `mapping.json` whose `stage3a_2` block is `null` and `method` is `hybrid_shortlist_only`.

### 2.4 `changes.json` — final artifact

```json
{
  "schema_version": "1.0",
  "run_id": "2026-05-13T12-00-00Z--ab12cd34",
  "records": [
    {
      "candidate_id": "cand-0001",
      "location": {
        "file": "src/foo/bar/scheduler.py",
        "symbol": "Scheduler._dispatch_step",
        "line_start": 142, "line_end": 211
      },
      "mapped_findings": ["find-0007", "find-0011"],
      "proposals": [
        {
          "rank": 1,
          "title": "Replace per-step pending-list allocation with a reusable ring buffer",
          "description": "Pre-allocate a fixed-capacity request slot pool; reuse across steps; track head/tail with atomic counters.",
          "expected_impact": {"metric": "decode_step_latency", "estimate": "5-12% reduction", "evidence_strength": "medium"},
          "prerequisites": ["request count bound known at init", "no holes mid-step"],
          "effort_estimate": "S (≤1 day)",
          "supporting_finding_ids": ["find-0011"],
          "risk": "low"
        },
        {
          "rank": 2,
          "title": "Adopt paged KV-block indexing as in PagedAttention",
          "description": "...",
          "supporting_finding_ids": ["find-0007"],
          "effort_estimate": "L (>1 week)",
          "risk": "high"
        }
      ],
      "agent": "claude_code",
      "telemetry": {"tokens_in": 18400, "tokens_out": 3100, "cost_usd": 0.32}
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
                       │  Stage 3a-1: Shortlist           │
                       │  BM25 + proxy embeddings +       │
                       │  optional LLM rerank →           │
                       │  top-10 findings per candidate   │
                       │  → shortlist.json                │
                       └─────────────────┬────────────────┘
                                         ▼
                       ┌──────────────────────────────────┐
                       │  Stage 3a-2: Mapping review loop │
                       │  (CORAL-style loop, like Stage 1)│
                       │  bootstrap:  Claude Code         │
                       │  review #1:  Codex               │
                       │  review #2:  Claude Code         │
                       │  ↓ (until stop cond.)            │
                       │  → mapping.json                  │
                       └─────────────────┬────────────────┘
                                         ▼
                       ┌──────────────────────────────────┐
                       │  Stage 3b: Per-candidate change  │
                       │  generation (Claude OR Codex,    │
                       │  one subprocess per candidate,   │
                       │  parallel with semaphore)        │
                       │  → changes.json                  │
                       └──────────────────────────────────┘
```

---

## 4. CORAL: What to Lift, What to Skip

The user's spec references `coral/agent/builtin/claude_code.py`, `coral/agent/builtin/codex.py`, and `coral/workspace/repo.py`; the current public repo (last verified 2026-05-13, https://github.com/Human-Agent-Society/CORAL, `main` at 10 commits, README dated 2026-03-18) places these at `coral/agent/runtime.py` and `coral/workspace/setup.py` (Architecture section of README). The deltas are noted below.

| CORAL artifact | Decision | Reason |
|---|---|---|
| `coral/agent/runtime.py` (subprocess wrapper for Claude / Codex / OpenCode) | **Lift** — the Popen pattern, log-tailing thread, JSONL parsing of `stream-json` / `--json` | Single best-tested non-interactive wrapper; saves a week. Note the path differs from the spec's `builtin/claude_code.py`; surface this delta to the user before implementation (§11 Q1). |
| `coral/agent/manager.py` lifecycle (spawn, heartbeat, reboot on max_turns) | **Lift selectively** — keep spawn + max_turns reboot; **skip** heartbeat interrupts and `/loop`-style prompts | Our pipeline is finite, not evolutionary. |
| Agent class registry (`coral/agent/runtime.py` selects by `runtime: claude_code|codex|opencode` from YAML) | **Lift** as a thin `AgentRunner` protocol with two implementations | Same `runtime: …` config UX; lets us add `opencode` later. |
| `_clean_env` style env scrubbing (referenced in CORAL workspace setup) | **Lift** | Prevents `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and telemetry envs from leaking into agent subprocesses unintentionally. **Important interaction with §6:** the orchestrator process holds `OPENAI_API_KEY` / `OPENAI_BASE_URL` for the GPT Researcher call; the agent subprocesses (Claude Code, Codex) must receive their own credentials and *not* inherit the LiteLLM-proxy envs (Codex would otherwise try to call the LiteLLM endpoint instead of OpenAI). |
| Worktrees (`coral/workspace/setup.py` clones a git worktree per agent) | **Skip** for Stages 1 and 3 — both read-mostly; mount the repo read-only (or workspace-write of a separate artifacts dir). **Optionally use** if we ever auto-apply patches in v2. | Worktrees add ~3–5s/agent and a git checkpoint surface area we don't need. |
| `.coral/public/` shared knowledge hub, attempts, notes, skills | **Skip** | Only useful for evolutionary improvement loops — orthogonal to a one-shot candidate-discovery pass. The user explicitly flagged these as skip candidates and we agree. |
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
- **`--sandbox read-only` + `--add-dir` interaction (the user's flagged uncertainty):** the Codex docs (https://www.simplified.guide/codex/writable-directory-add, https://github.com/openai/codex/issues/2797, https://developers.openai.com/codex/cli/features) consistently describe `--add-dir` as "expose more **writable** roots", and the canonical example is `codex exec -s workspace-write --add-dir /tmp/codex-writable …`. Read-only mode has no writable roots by definition (https://developers.openai.com/codex/concepts/sandboxing). **Conclusion**: `--sandbox read-only --add-dir X` is *not* a supported "read everything, write only to X" combination. **Recommended workaround**: use `--sandbox workspace-write` with `-C <artifacts_dir>` as the workspace root and add the target repo via `--add-dir <repo>`. Codex's `sandbox_workspace_write.exclude_*` keys prevent unintended writes to system tmp. If strict read-only of the repo is mandatory, run the target repo on a read-only bind mount and let workspace-write apply only to the artifacts directory.
- Output channels: `--json` writes NDJSON state events to stdout; `--output-last-message <path>` writes the final assistant message to a file; `--output-schema` validates the final response against a JSON Schema. Use all three.
- **`--output-schema` enforcement strength is open** — see §11 Q3.
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

1. **Normalized-JSON equality**: `canonical_json(candidates_iN) == canonical_json(candidates_i{N-1})` after sorting by `(file, symbol, line_start)` and stripping whitespace/timestamps.
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
- **OpenHands** (https://github.com/All-Hands-AI/OpenHands) — evaluated as a §10 baseline. Its general "agent does everything in one loop" is exactly what we *don't* want for the cost-disciplined separation of research from coding.
- **KernelBench's prompting style** (https://scalingintelligence.stanford.edu/blogs/kernelbench/) — for GPU-kernel candidates, their explicit "replace this PyTorch op with a custom kernel" framing maps to the `tags=[kernel-launch, fusion-opportunity]` cluster.
- **PIE (pie4perf)** and **FasterPy** — both prove that performance-aware retrieval (give the model exemplars of past optimization edits) significantly beats blind prompting. We borrow this by mandating that Stage-2 findings include `suggested_changes[]` in a form the Stage-3b prompt can show as exemplar guidance.

### 5.5 Validation per iteration

- pydantic schema parse → reject on failure (retry once with a strict-mode reminder appended to the prompt).
- Path existence: every `candidates[i].file` resolves to an existing file at the pinned `git_sha`; bad paths → mark candidate `invalid:true` and continue (don't fail the run).
- Line range sanity: `1 ≤ line_start ≤ line_end ≤ file_line_count`.
- Excerpt SHA: orchestrator computes `code_excerpt_sha256` itself to detect drift across iterations.

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
Emit ONE JSON object matching the schema at @candidates.schema.json. Specifically:
- 5 ≤ len(candidates) ≤ 40
- every `file` must be a path that exists in this checkout
- `rationale` ≤ 240 chars, falsifiable (cite the loop / the alloc / the sync)
- `tags` chosen from the controlled vocabulary listed in the schema
- `evidence_pointers` is optional but encouraged
- emit `meta.delta = {"added": [...all ids...], "removed":[], "modified":[]}`

Do NOT explain outside the JSON object.
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
  does not contain the claimed symbol.
- Merge candidates that point at the same hot path with overlapping line ranges.
- For each kept candidate, you MAY tighten `rationale` and `tags`.
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

**Single provider, pinned by user directive.** All previous multi-tier comparisons are removed.

### 6.1 Provider: GPT Researcher

- Repo: https://github.com/assafelovic/gpt-researcher (Apache-2.0)
- Library import: `from gpt_researcher import GPTResearcher`
- Pin: `gpt-researcher >= 0.13`
- Why: importable directly from Python, emits a structured report we coerce to our findings schema, supports arbitrary OpenAI-compatible LLM endpoints via env vars, separates LLM from retriever, MIT-friendly license.

### 6.2 LLM configuration — LiteLLM proxy

GPT Researcher honors `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the process environment (and equivalent config-file keys). The user-provided client snippet:

```python
import openai
client = openai.OpenAI(
    api_key="some key",
    base_url="https://ete-litellm.ai-models.vpc-int.res.ibm.com",
)

response = client.chat.completions.create(
    model="<MODEL_NAME>",         # placeholder — must be filled in
    messages=[{"role": "user", "content": "Your prompt here"}],
)
```

…is exactly the contract GPT Researcher uses internally. The wrapper sets:

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

**Image input is not used by Stage 2.** The `encode_image` helper in the user-provided snippet is for vision-capable models; Stage 2 is text-only literature/engineering search, so we ignore that path. The same client config works either way.

### 6.3 Retriever (search backend) — default `arxiv`, Tavily recommended

**Resolved 2026-05-13.** Stage 2 has two retriever modes:

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

### 6.4 Embeddings (consumed by Stage 3a, but procured via the same proxy)

If the LiteLLM proxy exposes an embedding endpoint (most LiteLLM deployments do — `/v1/embeddings`), use it directly: `client.embeddings.create(model="<EMBEDDING_MODEL_NAME>", input=[...])`. If not, fall back to `sentence-transformers` running `BAAI/bge-m3` locally. Voyage AI is dropped because it requires SaaS egress.

### 6.5 Single-call vs deep-research mode

GPT Researcher's `report_type` options (https://docs.gptr.dev/docs/gpt-researcher/getting-started/getting-started-with-docker, source: `gpt_researcher/utils/enum.py`):

- `research_report` — single retrieval pass, single synthesis; fastest, cheapest.
- `detailed_report` — section-by-section, deeper.
- `deep_research` — multi-agent, tree-of-thought style search, deepest. Best for our use case.

Recommended pin: `report_type="deep_research"`. Expect 5–15 minutes wall-clock per call.

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

If the proxy's `<MODEL_NAME>` does not support `response_format=json_schema`, the wrapper falls back to plain JSON-mode (`response_format={"type":"json_object"}`) and then validates with pydantic, retrying once on parse failure with a strict reminder. See §11 Q7.

### 6.7 Per-call budget cap

Even with a single provider, runaway cost is possible. Enforce:

- `--stage2-wallclock-s` (default 1800 s = 30 min) — hard timeout on the GPT Researcher call.
- `--stage2-max-iterations` — pass to GPT Researcher `config_path` to bound the deep-research tree.
- `--stage2-budget-usd` — best-effort; only enforceable if the LiteLLM proxy returns `usage` blocks (§11 Q6).

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
include: title, URL, source type (paper/PR/issue/blog/talk/docs/codebase), publication date,
the technique it proposes, the measured or claimed gains (with baseline and conditions),
the components or symbols it would apply to (use concrete keywords that would appear in
source code), suggested changes, and your assessment of evidence quality.
````

### 6.10 Stage 2 coercion prompt (drop-in, fed to second LLM call)

````
You are converting a research report into a strict JSON findings list.

## Input
A markdown report on optimization techniques relevant to a code module. The report is
appended below the `---` separator.

## Output
A single JSON object matching the schema at @findings.schema.json. Every finding must
populate:
- title, url, source_type ∈ {paper, pr, issue, blog, talk, docs, codebase}
- technique_name, technique_summary (≤ 80 words)
- claimed_gains[].(metric|magnitude|baseline|conditions)
- target_components[].(component_hint|file_or_symbol_hints[]|keywords_for_matching[])
- suggested_changes[] (≤ 5 bullets, each ≤ 25 words)
- evidence_quality.(primary|peer_reviewed|has_benchmarks|reproducible|score∈[0,1])

The `keywords_for_matching` array is what the downstream mapper will hybrid-search
against candidate code+rationale. Choose 5-15 specific tokens that would actually
appear in source code or in a code-review rationale (e.g., "kv_cache", "block_table",
"chunked_prefill", NOT "AI", "performance", "speedup").

- Only include findings whose URL is present in the report. Do NOT invent URLs.
- Drop any finding for which you cannot fill the required fields.
- Do not emit prose outside the JSON object.

---

{report_markdown}
````

---

## 7. Stage 3 Mechanics — Mapping + Change Generation

### 7.1 Mapping (Stage 3a) — two-pass: hybrid-retrieval shortlist + alternating-agent review

Stage 3a runs in two passes:

- **Stage 3a-1 (programmatic shortlist).** BM25 + embeddings + an optional LLM rerank produce a per-candidate shortlist of the top-N findings (default N=10). Fast (seconds per repo), no coding-agent invocation.
- **Stage 3a-2 (alternating Claude Code ↔ Codex review).** A coding agent reads the shortlist, the full candidates / findings files, and the actual repo code (via `--add-dir`), and emits `mapping.json` with a per-edge score and reasoning. The opposite agent then reviews and revises. Loop continues until a stop condition triggers — same six-condition shape as Stage 1 (§5.2).

**Why split.** Pure embeddings/BM25 conflate topical similarity with applicability: "this finding talks about quantization and this candidate is about quantization" is not the same as "this finding's specific technique applies to this code shape". Agents reading the actual code can make that call; running them across N×M pairs without a shortlist would blow context and cost. Conversely, agents without retrieval would need to read every finding for every candidate. The shortlist narrows the search space; the agent loop adjudicates within it. The alternating Claude↔Codex pattern from Stage 1 carries over: same debiasing motivation, same convergence shape.

#### 7.1.1 Stage 3a-1: Hybrid-retrieval shortlist

```
INPUT: candidates: List[Candidate], findings: List[Finding]
OUTPUT: shortlist: {candidate_id -> [(finding_id, hybrid_score)]} top-N (N=10 default)

1. Build candidate document:
     cand_doc(c) = c.rationale + " " + c.file + " " + c.symbol + " " +
                   " ".join(c.tags) + " " + code_excerpt(c)[:2000]

2. Build finding document:
     find_doc(f) = f.technique_name + " " + f.technique_summary + " " +
                   " ".join(f.target_components[*].component_hint) + " " +
                   " ".join(f.target_components[*].file_or_symbol_hints) + " " +
                   " ".join(f.target_components[*].keywords_for_matching)

3. Sparse: BM25 (rank_bm25) over the tokenized symbol+keyword space.
     bm25_score(c, f) ∈ [0, ~]

4. Dense: embeddings via the same LiteLLM proxy (§6.4).
     # POST {OPENAI_BASE_URL}/v1/embeddings, model=<EMBEDDING_MODEL_NAME>.
     # Batch candidates and findings separately to amortize cost.
     # Self-hosted fallback: BAAI/bge-m3 via sentence-transformers.
     dense_score(c, f) = cosine(emb(cand_doc(c)), emb(find_doc(f)))

5. Linear blend:
     hybrid(c, f) = 0.4 * normalize(bm25_score) + 0.6 * dense_score
   (weights tunable; sweep on a held-out repo)

6. For each c, keep top 10 findings by hybrid score.

7. Optional LLM re-rank (default: ON when proxy is responsive, OFF otherwise):
     Same proxy / same <MODEL_NAME> with the RERANK_PROMPT(c, top10_findings).
     Output: re-ordered list with scalar scores ∈ [0,1].
     Final score = 0.7 * llm_judge + 0.3 * hybrid.

8. Emit `shortlist.json` (top-N per candidate, N=10 default) as input to Stage 3a-2.
```

The shortlist is an **intermediate artifact**, not the final mapping. Stage 3a-2 may drop, demote, or reorder edges — but cannot add findings outside the shortlist (that's what 3a-1 is for; widening the search at the agent layer would defeat the cost split).

#### 7.1.2 Stage 3a-2: Alternating-agent review loop

```
i=0: claude_code bootstrap (--permission-mode plan, --add-dir <repo-path>)
     prompt: STAGE3A_BOOTSTRAP_PROMPT
     reads:  candidates.json, findings.json, shortlist.json, repo (read-only)
     emits:  mapping_v0.json
             # per edge: {candidate_id, finding_id, score ∈ [0,1], reasoning, source_iter}

i=1: codex review (--sandbox workspace-write, workspace=artifacts, --add-dir <repo-path>)
     prompt: STAGE3A_REVIEW_PROMPT(mapping_v0.json, role="adversarial reviewer")
     emits:  mapping_v1.json + meta.delta = {added:[], removed:[], modified:[], rescored:[]}

i=2: claude review
     prompt: STAGE3A_REVIEW_PROMPT(mapping_v1.json, role="adversarial reviewer")
     emits:  mapping_v2.json

... until stop condition ...
```

The reviewer is instructed to: **demote** edges where the finding's technique doesn't actually apply to the code shape (e.g., a CUDA-kernel finding mapped to pure-Python candidate); **promote** shortlist edges the bootstrap missed; **rescore** edges where the reasoning is weak or contradicted by the code; **mark for inspection** candidates whose top edge fell below `--map-min-score` (default 0.45). Reviewers may **not** introduce findings outside the shortlist.

`--mapping-review {alternating|claude_code|codex|off}` — default `alternating`. Setting `off` skips Stage 3a-2 entirely; `mapping.json` then comes straight from the hybrid shortlist (the prior behavior). Useful for cost-floored runs and for ablation in M3.

#### 7.1.3 Stop conditions (ALL evaluated each iteration; ANY triggers exit)

1. **Canonical-JSON equality**: `canonical_json(mapping_iN) == canonical_json(mapping_i{N-1})` after sorting by `(candidate_id, finding_id)` and rounding scores to 2 dp.
2. **Edge-set stability**: |symmetric-difference of edges| / |total edges| ≤ 5%.
3. **Score stability**: mean per-edge `|score_iN - score_i{N-1}|` ≤ 0.05.
4. **"No changes" predicate**: reviewer's `meta.delta` is empty across all four lists.
5. **Max-iter cap**: `--stage3a-max-iters` (default 3 — mapping converges faster than candidate discovery; the search space is bounded by the shortlist).
6. **Budget guard**: cumulative Stage-3a-2 `cost_usd` ≥ `--stage3a-budget-usd` (default $2). The combined `--stage3-budget-usd` (default $6 = $2 mapping + $4 changegen) is a parent cap.
7. **Hard-failure latch**: invalid JSON twice in a row after retry → abort 3a-2, emit last good mapping (or fall back to the 3a-1 shortlist if iter 0 failed).

#### 7.1.4 Per-iteration validation

- pydantic schema parse on `mapping.json` → reject on failure (retry once with a strict-mode reminder).
- Every `candidate_id` and `finding_id` referenced must exist in the input files; unknown ids → invalid iter.
- Score ∈ [0, 1]; reasoning ≥ 20 characters.
- Edges not in the shortlist are dropped with a warning (do not fail the iter).

#### 7.1.5 Completeness check (mapping output)

Every candidate must have ≥1 edge with score ≥ `--map-min-score` (default 0.45). Otherwise the candidate lands in `mapping.json`'s `orphans[]` and proceeds to Stage 3b with `mapped_findings: []` — the change-generation prompt then falls back to general performance heuristics.

**Many-to-many shape:** a finding may appear under multiple candidates (expected and desirable; not deduped). For Stage 3b inputs we materialize a **candidate-centric** view (top-K findings per candidate, K=5 default, ≤ N=10).

#### 7.1.6 Concrete Stage-3a-2 prompts (drop-in)

**Bootstrap prompt** (`prompts/stage3a_bootstrap.md`):

```
You map performance optimization CANDIDATES (specific code locations) to research
FINDINGS (techniques from papers / blogs / PRs / issues).

## Inputs you have
- `candidates.json` — list of candidate code locations with rationale and tags
- `findings.json`   — list of research findings with techniques and applicability hints
- `shortlist.json`  — top-10 findings per candidate from a hybrid-retrieval pre-pass
                      (BM25 + embeddings). Treat as your candidate set — do NOT
                      introduce findings outside it.
- the repository (read-only via --add-dir) — open files, read code, verify fits

## Rules
- For each candidate, pick the findings (from its shortlist) whose technique
  ACTUALLY applies to the code shape at that location. A finding can match in
  topic but fail in shape (e.g., "fused-attention kernel" → a pure-Python
  bytecode-level candidate). Demote those.
- Use score ∈ [0, 1]: 0.85+ = high confidence the technique applies, 0.6 =
  plausible, 0.45 = weak / speculative, <0.45 = drop.
- Read the actual code around each candidate's (file, symbol, line range)
  before assigning a score above 0.6.
- Reasoning ≥ 20 chars and grounded in what you read.

## Output (REQUIRED — strict JSON, matches @mapping.schema.json)
{
  "edges": [
    {"candidate_id": "...", "finding_id": "...", "score": 0.0–1.0,
     "reasoning": "...", "source_iter": 0}
  ],
  "orphans": ["candidate_id", ...],
  "meta": {"delta": {"added": [...], "removed": [...], "modified": [...],
                     "rescored": [...]}}
}
No prose outside the JSON.
```

**Review prompt** (`prompts/stage3a_review.md`):

```
You are an adversarial reviewer of a candidate↔finding mapping produced by
another agent.

## Inputs
- `candidates.json`, `findings.json`, `shortlist.json` (same as bootstrap)
- `mapping.json` — the previous iteration's edges
- the repository (read-only)

## Rules
- Demote (or drop) edges where the finding's technique doesn't actually apply
  to the code at the candidate's location. Open the code and verify.
- Promote shortlist edges the previous iteration missed if you can justify ≥0.6.
- Rescore edges whose reasoning is weak, contradicted by the code, or
  speculative.
- You may NOT add findings outside `shortlist.json`.
- Keep edits minimal — if the previous mapping is correct, return it unchanged
  with `meta.delta` lists empty.

## Output (REQUIRED — strict JSON, matches @mapping.schema.json)
Same shape as the bootstrap. `meta.delta` must accurately describe the diff
from the input mapping; the orchestrator uses it for the stop conditions.
```

**Embedding model choice rationale (unchanged):** the LiteLLM proxy is the single sanctioned egress for Stage 2 and Stage 3a-1. If it serves an embedding endpoint, use it; otherwise fall back to `BAAI/bge-m3` locally (1024-d, multilingual). voyage-code-3 dropped to honor the single-endpoint constraint.

**Prior art consulted:** RepoCoder (https://arxiv.org/abs/2303.12570) for retrieval over code repos; CodeRAG-Bench (https://arxiv.org/abs/2406.14497) for code-retrieval evaluation; SWE-bench solver families (Agentless https://arxiv.org/abs/2407.01489, SWE-agent) for "code-to-spec" matching patterns; CodeXEmbed (https://arxiv.org/abs/2411.12644) as a counterpoint that argues large open-source code-retrieval models are catching up. The Stage 3a-2 review loop is a direct port of Stage 1's bootstrap+critique pattern; the same prior art motivates it.

### 7.2 Change generation (Stage 3b)

One subprocess invocation **per candidate** (with `mapped_findings` ≤ K), parallelized via `asyncio.Semaphore(n=--max-parallel-agents)` (default 4). Choice of agent: round-robin between Claude Code and Codex by default to debias; CLI flag `--changegen-agent claude_code|codex|alternating` overrides.

Per-call cost cap: `--per-candidate-budget-usd` (default $0.50).

### 7.3 Change-generation prompt (drop-in)

````
You are proposing concrete, ranked optimization changes for a single code location.
You do NOT write a patch. You produce a structured proposal list that a human (or a
future v2 patch-writer) can act on.

## Code location
- file: {file}
- symbol: {symbol}
- line range: {line_start}-{line_end}
- excerpt (read-only):
  ```{lang}
  {code_excerpt}
  ```
- rationale from the candidate-discovery pass:
  """
  {candidate_rationale}
  """
- tags: {tags}

## Mapped research findings (top-{K})
{for each finding f in mapped_findings:}
- id: {f.id}
  title: {f.title}
  url: {f.url}
  technique: {f.technique_name}
  summary: {f.technique_summary}
  claimed_gains: {f.claimed_gains}
  suggested_changes: {f.suggested_changes}

(If mapped_findings is empty, fall back to general performance heuristics for the
 tags above and say so in each proposal's `supporting_finding_ids: []`.)

## Task
Produce 1 to 5 ranked proposals. Each proposal must:
- be implementable as a localized change (single file or small set of files near
  this location); reject ideas that require a full system redesign
- name the supporting finding ids you used
- give an expected_impact estimate with an honest `evidence_strength` ∈ {high,
  medium, low}
- list prerequisites and risks
- estimate effort: XS (<1h), S (≤1d), M (≤1w), L (>1w)

Order by (expected_impact_magnitude × evidence_strength / effort). Best proposal
first.

## Output (REQUIRED — strict JSON)
Match the schema at @proposals.schema.json. No prose outside the JSON.
````

### 7.4 v2: produce a runnable patch?

Deferred to v2. The Stage 3b prompt could be extended with "and produce a unified diff implementing proposal #1" gated by `--gen-patch`, validated with `git apply --check`. The cost roughly doubles (output tokens grow significantly) and the failure surface area (compile errors, hallucinated APIs) explodes; KernelBench results suggest 30–50% of LLM-generated kernels fail to compile even with explicit signatures. Plan: add only after M6 closes the eval loop with measured improvement on at least 3 modules.

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
│   │   └── changes.py
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
│   │   │   ├── bm25.py
│   │   │   ├── dense.py             # proxy /v1/embeddings | bge-m3 fallback
│   │   │   ├── rerank.py            # LLM-judge re-rank via the same proxy
│   │   │   ├── shortlist.py         # 3a-1: hybrid blend, top-N=10, emit shortlist.json
│   │   │   ├── agent_loop.py        # 3a-2: bootstrap + alternating review, stop conds
│   │   │   ├── validate.py          # per-iter schema parse, edge-in-shortlist check
│   │   │   └── prompts/             # stage3a_bootstrap.md, stage3a_review.md
│   │   ├── changegen/
│   │   │   ├── runner.py            # per-candidate subprocess fan-out + semaphore
│   │   │   └── prompts/stage3_changegen.md
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
  --stage3a-budget-usd 2 --stage3a-max-iters 3 --mapping-review alternating \
  --stage3-budget-usd 6 \
  --max-parallel-agents 4 \
  --out ~/.cache/optquest/owner__repo/2026-05-13--ab12

# Sequential to ease debugging; skip the mapping-review loop (use shortlist directly)
optquest run --repo . --module pkg/core \
  --sequential \
  --mapping-review off \
  --changegen-agent codex \
  --no-llm-rerank

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

## 10. Decisions Made

| # | Decision | Rationale | Rejected alternative |
|---|---|---|---|
| 1 | Subprocess agents (`claude -p`, `codex exec`) instead of SDKs | The Anthropic Agent SDK and OpenAI Agents SDK both work, but the user explicitly asked for subprocess CLIs and the CORAL pattern is built around them. Subprocess is also the only path that gives identical UX across Claude Code and Codex. | Anthropic Agent SDK + OpenAI Agents SDK — lost because they fragment the integration and bypass CORAL reuse. |
| 2 | Prompts on stdin / via `@file`, never as argv | Linux MAX_ARG_STRLEN = 128 KB per argv (confirmed in multiple kernel-source citations and a real bug report on Claude Code SDK when CLAUDE.md exceeded 128 KB — https://github.com/AndyMik90/Auto-Claude/issues/1414). | Inline argv — fails at scale. |
| 3 | Claude Code uses `--permission-mode plan` in Stage 1 | Plan mode is documented (code.claude.com/docs/en/permission-modes) as "research and propose, do not edit". Exactly the Stage-1 semantic. | `--permission-mode acceptEdits` — allows edits we don't want. `--dangerously-skip-permissions` — overkill. |
| 4 | Codex uses `--sandbox workspace-write` with workspace = artifacts dir, repo via `--add-dir`, not `--sandbox read-only --add-dir <writable>` | Docs are consistent that `--add-dir` expands writable roots in workspace-write mode; read-only has no writable roots. The combination "read-only with one writable scratch dir" is not natively supported. | The user's intuitive combination — not actually permitted. |
| 5 | Stages 1 & 2 parallelized with **asyncio** + `anyio.to_thread.run_sync` for blocking-subprocess wait | asyncio gives us one event loop for fan-out, semaphores, and budget pumping. Subprocess.Popen still works inside an asyncio worker thread; we don't need `asyncio.create_subprocess_exec` everywhere because the agents are slow enough that thread-spawn cost is rounding. | Bare threads + `concurrent.futures` — fine but awkward to mix with semaphore-bounded Stage 3b. Pure `multiprocessing` — overkill, the work is IO-bound at the orchestrator level. |
| **6** | **Stage 2 = GPT Researcher only, configured against a LiteLLM proxy (OPENAI_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com)** | **User directive.** Plan is no longer cost-tiered. All credentials/egress flow through one endpoint; multi-provider rejected. | OpenAI Deep Research API, Anthropic + web_search, Perplexity Sonar — all require additional egress paths the user did not authorize. |
| 6a | Set GPT Researcher's LLM via OPENAI_API_KEY + OPENAI_BASE_URL env vars (plus FAST_LLM / SMART_LLM / STRATEGIC_LLM = `openai:<MODEL_NAME>`) | This is GPT Researcher's documented OpenAI-compatible config pattern (https://docs.gptr.dev/docs/gpt-researcher/llms/llms#openai); avoids monkey-patching the openai client. | Wrapping `openai.OpenAI(base_url=…)` directly inside a custom retriever — works but reimplements the planner. |
| 6b | Force `report_type="deep_research"` with a hard `--stage2-wallclock-s` cap | `research_report` returns too thin a survey for downstream mapping; `detailed_report` is intermediate. The wallclock cap bounds open-ended exploration. | Always `detailed_report` — produces fewer findings; mapping orphans more often. |
| 6c | Mandatory markdown→JSON coercion call to the same proxy | GPT Researcher emits markdown, mapper needs JSON; one extra LLM call is the simplest reliable bridge. | Regex/heuristic markdown parsing — brittle, costs us most of the structured fields. |
| 6d | Default retriever `arxiv` (free, no key); auto-upgrades to `tavily,arxiv` if `TAVILY_API_KEY` is set | arxiv API is free and key-less, so the pipeline runs out of the box. Tavily (paid) is opt-in for blog / GitHub / vendor-doc coverage. The orchestrator switches modes by inspecting the env at startup. See §6.3. | Hard-pinning `tavily,arxiv` (blocks installs without a paid key); arxiv-only with no upgrade path (gives up the blog/issue coverage the user explicitly asked for). |
| 7 | Stage 3a-1 shortlist: BM25 + proxy embeddings (or bge-m3 fallback), LLM-rerank optional; top-N=10 per candidate | Shortlist narrows the N×M search space before the agent loop. Hybrid retrieval is cheap and topical; agents adjudicate within it. voyage-code-3 dropped (separate SaaS); BM25 still important — embeddings drift on rare code tokens. | voyage-code-3 — disallowed by the single-endpoint constraint. text-embedding-3-large directly from openai.com — also disallowed. Agent-only mapping with no retrieval — would blow context for N×M pair evaluation. |
| 8 | One subprocess per candidate in Stage 3b | Maximizes parallelism, isolates failures, makes per-candidate cost telemetry trivial. | One big batched prompt — argv-size pressure, harder to schema-validate, single failure kills the run. |
| 9 | JSON Schema validation (pydantic) instead of CORAL graders | The pipeline has no scalar score; correctness is "shape + path-existence + URL-reachability". | CORAL `TaskGrader` — overkill. |
| 10 | No session-resume in MVP | Resume halves the "two failures multiply" reliability story; saves <30% tokens; v2 work. | Resume from iter 1 → iter 2 — viable, deferred. |
| 11 | Pipeline runs read-only against target repo; all writes go to `~/.cache/optquest/...` | The user's "target-repo mutation guard" requirement is non-negotiable. | Worktrees — overkill for read-only Stage 1; we adopt them only if v2 auto-applies patches. |
| 12 | **Baseline considered for the "one big agent" alternative**: SWE-agent / OpenHands directly | SWE-agent (https://swe-agent.com) and OpenHands (https://github.com/All-Hands-AI/OpenHands) are general code-agent harnesses optimized for SWE-bench. They lack a built-in deep-research stage, and shoving "do research and propose changes" into one agent surrenders the cost discipline of separating cheap deep-research from expensive coding-agent loops. We use them as **eval baselines only** (see §12 milestones). | "Just run OpenHands on the repo" — strictly worse on the candidates-with-citations metric we care about. |
| 13 | Stage-3a-1 optional LLM rerank uses the same `<MODEL_NAME>` via the same proxy | Same egress, same auth; per-pair rerank is cheap. ON by default when proxy is responsive; the agent loop in 3a-2 is the primary judgment layer, so the rerank is mainly a shortlist-quality booster. | Always-off rerank — wider shortlist quality variance into the agent loop. A *different* model for rerank — would mean another credential. |
| 14 | Findings schema mandates `keywords_for_matching` | Drives BM25 quality without LLM-driven keyword extraction in Stage 3a. The coercion prompt enforces it. | LLM keyword extraction at map time — extra LLM call we don't need if the coercion step does its job. |
| 15 | Strip `OPENAI_BASE_URL` and `OPENAI_API_KEY` from the env passed to Codex / Claude Code subprocesses | The LiteLLM proxy credentials are for Stage 2 / Stage 3a inside the orchestrator process only. Codex would otherwise try to call the proxy as if it were `api.openai.com` and break, or worse, succeed in unexpected ways. The lifted `_clean_env` is the cleanest mechanism. | Sharing the same env — produces silent misrouting bugs. |
| 16 | No automatic compile+benchmark verification in MVP (proposed in v2) | Compile+benchmark requires GPU-capable runners, a per-repo benchmark harness, and per-proposal patch synthesis — three failure modes that compound. Defer until M6's hand-applied eval proves the proposals are worth this investment. | Auto-bench everything from M1 — premature. |
| 17 | Stage 3a-2 mapping uses the same alternating Claude-Code↔Codex review pattern as Stage 1 | Mapping is a judgment task (does the finding's technique apply to this code shape?), not just a similarity calculation. The Stage-1 bootstrap+critique pattern debiases single-model failure modes here for the same reason it does in candidate discovery. Hybrid retrieval (3a-1) seeds a bounded shortlist so the agents don't drown in N×M context. | Pure-programmatic mapping (3a-1 alone) — what we had before; misses applicability judgments and produces more orphans. Single-agent mapping with no review — loses the debiasing. Agent-only with no shortlist — context blow-up. |

---

## 11. Still-Open Questions

1. **Exact CORAL paths.** The user's spec referenced `coral/agent/builtin/claude_code.py`, `coral/agent/builtin/codex.py`, and `coral/workspace/repo.py`. The public repo (as of 2026-05-13) lists `coral/agent/runtime.py` and `coral/workspace/setup.py` in its README's Architecture section. Either the user has a private/older fork, the README is stale, or the spec is approximate. **Resolution needed before implementation**: confirm with the user which tree to lift from.
2. **`claude --add-dir` vs `additionalDirectories` semantics in non-interactive mode.** The docs document `additionalDirectories` in `settings.json` (code.claude.com/docs/en/permission-modes) and shipyard.build/claude-code-cheat-sheet shows `--add-dir` working in CI; but the exact relationship between argv `--add-dir`, `settings.json:permissions.additionalDirectories`, and `--cwd` under `--print` is not crisply documented. **Resolution**: empirical smoke test in milestone M0.
3. **Codex `--output-schema` enforcement strength.** Docs say "Codex validates tool output against it" — unclear whether the validator is hard (rejects + retries internally) or soft (just attaches schema to the system prompt). **Resolution**: M0 test; if soft, we layer pydantic on top.
4. **Stage-2 URL reachability standard.** Some papers sit on arxiv preprints that move; some PR URLs become 404 after force-pushes. Threshold for "reachable" needs to be a policy decision — current plan: HTTP 200 within 10s, ≥1 KB body, no detected paywall block-page heuristics. **Resolution**: an `--allow-unreachable-urls` escape hatch with a warning.
5. **Stage 2 retriever choice for the user's environment.** **RESOLVED 2026-05-13.** Stage 2 runs on a node with public internet egress. Default `RETRIEVER=arxiv` (free, key-less, papers only); the orchestrator auto-upgrades to `RETRIEVER=tavily,arxiv` when `TAVILY_API_KEY` is set in the env (adds blogs / GitHub issues / vendor docs / articles). No install-time secret required for the default path. See §6.3.
6. **Whether the LiteLLM proxy returns `usage` blocks / cost data on chat completions.** LiteLLM by default does return `usage.prompt_tokens` and `usage.completion_tokens`, but cost (`response_cost`) only when the deployment is configured for it. The model name placeholder (`<MODEL_NAME>` in the user snippet) makes it impossible to predict ahead of time. **Resolution**: probe the proxy at M0; if cost is unavailable, set `cost_unknown: true` and rely on `--stage2-wallclock-s` as the primary budget guard.
7. **Whether `<MODEL_NAME>` (to be filled in) supports `response_format=json_schema`.** OpenAI's gpt-4o-2024-08-06 and later support strict JSON-schema mode; LiteLLM passes this through for compatible upstream models but silently degrades for others. **Resolution**: probe at M0; coercion call falls back to `response_format={"type":"json_object"}` + pydantic validation + retry on failure.
8. **Whether the alternating-review-loop materially improves quality vs a single Claude bootstrap, in BOTH Stage 1 (candidate discovery) and Stage 3a-2 (mapping).** Genuine uncertainty — no public benchmark for "performance candidate discovery" or "candidate↔finding mapping". **Resolution**: ablations in M3 of the eval. Stage 1: Claude-only vs Claude+1-Codex-review vs full alternating. Stage 3a: `--mapping-review off` (shortlist only) vs Claude-only bootstrap vs full alternating. The `off` mode preserves the previous purely-programmatic mapping as the ablation baseline.

---

## 12. Milestones

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

**M3 — Mapping + change generation (Week 3).** Stage 3a-1, 3a-2, and 3b on artifacts from M1 and M2.
- Command: `optquest stage3 --candidates tmp/candidates.json --findings tmp/findings.json --out tmp/`
- Pass:
  - `tmp/shortlist.json`, `tmp/mapping.json`, `tmp/changes.json` all validate.
  - Stage 3a-2 stop condition (canonical equality, edge stability, score stability, or no-change predicate) triggers within `--stage3a-max-iters` (default 3) on ≥2 of 3 eval repos.
  - Final orphan rate ≤ 30% (lower than 3a-1-only baseline by ≥10 pts — measured via `--mapping-review off` ablation; resolves §11 Q8 for mapping).
  - Every change record has ≥1 proposal; mean per-candidate cost ≤ $0.50 (if cost reporting is available; else mean per-candidate wallclock ≤ 60s).
  - Stage-3a-2 budget cap `--stage3a-budget-usd $2` not exceeded.

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

## 13. Non-Goals

- **Writing the implementation** (this is a plan only).
- **Auto-applying patches** to the target repo. Stage 3 emits proposals; v2 may emit patches; this plan does neither apply nor PR.
- **Multi-repo or multi-module batch mode.** One repo × one module per run.
- **Replacing the pipeline with a single existing agent** (SWE-agent, OpenHands, Aider). Evaluated as a baseline in M5 (§12, decision #12); not adopted.
- **Building a UI.** The CORAL `ui` is skipped; a plain markdown report is the deliverable. Re-evaluate in v2.
- **Cross-language IR-level analysis.** We rely on the agent's textual understanding of code + optional profiler artifacts; we do not build AST or LLVM-IR passes.
- **Multi-provider Stage 2.** Explicitly out of scope by user directive — GPT Researcher only.
- **Image/vision input to Stage 2.** The user-provided snippet shows an `encode_image` helper; Stage 2 in this pipeline is text-only. Vision could be added in v2 for screenshots of flame graphs.

---

## 14. Possible Future Optimizations

1. **Session resume across iterations** (Claude `--resume <uuid>`; Codex `codex exec resume`). Cuts Stage 1 tokens by 20–40%. Adds: session-id loss handling, drift-detection on resume.
2. **Patch generation in Stage 3b.** Append a "now produce a unified diff for proposal #1" step gated by `--gen-patch`. Validate by `git apply --check` against the pinned sha.
3. **Auto-benchmark closing the loop.** For each generated patch, attempt compile + run + benchmark in a worktree (CORAL's worktree pattern earns its keep here). Auto-rank proposals by measured Δ.
4. **OpenCode as a third agent** in the alternating review. CORAL already supports it; adding it just needs another `AgentRunner` subclass.
5. **Module summarization cache.** Stage-2's `module_summary` is recomputed each run; persist it keyed by `(repo, sha, module_path)`.
6. **Cross-repo learning.** Maintain a persistent `findings_library.parquet` of validated findings across runs — Stage 2 then becomes "retrieve from library, top-up from web". Cuts Stage-2 wallclock by 50–80% after the library warms.
7. **LLM-as-judge calibration.** Run the eval (M5) periodically and refit the LLM-judge weighting in the mapping rerank against the human rubric — current 0.7/0.3 blend is a guess.
8. **MCP integration.** Both Claude Code and Codex speak MCP; expose an `optquest-mcp` server so other agents (Claude Desktop, OpenAI Apps) can invoke the pipeline as a tool. GPT Researcher already provides `gptr-mcp` (https://github.com/assafelovic/gptr-mcp) as a reference implementation.
9. **Profiling-artifact-driven scope.** Tighter integration with `py-spy`, `nsys`, `pprof` — parse stack traces, weight candidates by self-time. PerfCoder, FasterPy, TritonForge, and MaxCode (cited in §12 M6) all suggest profiling-aware retrieval beats blind retrieval.
10. **Replace the deep-research call with a learned researcher.** Once we have a few hundred (candidate, finding) pairs from runs, fine-tune a small model (served via the same LiteLLM proxy) to do retrieval directly — cheaper than running GPT Researcher every time.
11. **A `coral ui`-style dashboard** that streams the alternating-review loop live with per-iteration diffs of `candidates.json` — re-uses CORAL's web/ for free.
12. **Vision input** (the `encode_image` path in the user-provided snippet) — feed flame-graph PNGs from `py-spy` or NVTX traces into Stage 1 or Stage 2 if the proxy's model is multimodal.
