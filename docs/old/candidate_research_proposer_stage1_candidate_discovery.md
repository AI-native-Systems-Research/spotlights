# candidate_research_proposer — Stage 1: Candidate Discovery

Stage 1 produces `candidates.json` — a list of falsifiable, high-yield optimization candidate locations in the target module — through an alternating Claude Code ↔ Codex review loop bootstrapped from a single Claude Code session in plan mode.

## 1. Python API

Stage 1 is invoked as a pure function. Input is one pydantic object; output is one pydantic object. There is no CLI for this stage; flags belong to the agent subprocesses we drive (Claude Code, Codex), not to the stage.

```python
from spotlights_engine.stage1 import run_stage1, Stage1Config, Stage1Result

result: Stage1Result = run_stage1(Stage1Config(
    repo_path=...,
    git_sha=...,
    module_qualified_name="v1/engine/core",
    modules_json=Path(".../modules.json"),
    artifacts_dir=Path(".../<run_id>"),
))
# result.candidates       — pydantic Candidates (mirrors candidates.json)
# result.iterations       — per-iteration telemetry
# result.total_cost_usd   — sum across iterations
# result.stop_reason      — why the loop ended
```

### Stage1Config (input)

```python
class Stage1Config(BaseModel):
    # Target
    repo_path: Path                        # local checkout root; must exist
    git_sha: str                           # informational; emitted with telemetry
    module_qualified_name: str             # ProjectTree.walk() key, e.g. "v1/engine/core"
    modules_json: Path | None = None       # if set, `Module` is resolved via ProjectTree.from_json(...).resolve(qn)

    # Scope bounding (see §4)
    scope_paths: list[str] = Field(default_factory=list)   # globs, repo-root-relative
    profile_paths: list[Path] = Field(default_factory=list)  # *.prof, *.pstats, *.nsys-rep, bench/profile.json

    # Loop control
    max_iterations: int = 6
    min_candidates: int = 5
    max_candidates: int = 40

    # Budgets (Stage 1 only — agent subprocess flags below)
    budget_usd: float = 5.0
    per_iteration_wallclock_s: int = 600
    per_iteration_budget_usd: float = 1.0

    # Artifacts
    artifacts_dir: Path                    # `candidates.json` and the JSONL telemetry land here
```

### Stage1Result (output)

```python
class IterationTelemetry(BaseModel):
    iteration: int
    agent: Literal["claude_code", "codex"]
    session_id: str
    duration_s: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    delta: CandidateDelta                  # {added, removed, modified: list[candidate_id]}
    candidate_count: int
    schema_retries: int                    # 0 or 1; >1 → hard-failure latch
    dropped_invalid_paths: int
    dropped_invalid_ranges: int

class Stage1Result(BaseModel):
    candidates: Candidates                 # mirrors candidates.json (see architecture doc §"Output JSONs")
    iterations: list[IterationTelemetry]
    total_cost_usd: float
    total_duration_s: float
    stop_reason: Literal[
        "converged",         # canonical-JSON equality with previous iteration
        "no_changes",        # agent's emitted meta.delta is empty
        "cycle",             # JSON hash seen before in this run
        "max_iterations",
        "budget_exhausted",
        "hard_failure",      # two consecutive schema-invalid outputs
    ]
```

`run_stage1` also persists `candidates.json` and a per-iteration `stage1.jsonl` telemetry file under `config.artifacts_dir`; the in-memory `Stage1Result` and the on-disk artifacts are equivalent.

## 2. Agent subprocess invocation (last verified 2026-05-13)

The pipeline shells out to the Claude Code and Codex CLIs because they are the canonical non-interactive entry points to those agents. The flags below are agent CLI flags, not Stage 1's API surface.

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
- Cost telemetry comes from the terminal `result` event's `total_cost_usd`, `usage.input_tokens`, `usage.output_tokens`, `duration_ms`, `session_id` — surfaced via `IterationTelemetry`.

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
- **`--output-schema` enforcement strength is open** — see architecture doc, Open Question 3.
- Session id: every `codex exec` prints `session id: <uuid>` near the top of the formatted output and emits a session-init event in the JSON stream. We extract it from the JSON stream for v2 resume.
- Argv limit: same Linux 128 KB MAX_ARG_STRLEN applies. **Workaround**: `codex exec -` reads the prompt from stdin (the dash is explicit and documented at developers.openai.com/codex/noninteractive).
- Cost telemetry: tokens-in/out and tool calls (including `web_search`) appear in the JSON event stream; wallclock is measured by the orchestrator and recorded on `IterationTelemetry`.
- **Credential isolation:** Codex must not see `OPENAI_BASE_URL` from the orchestrator's environment (that variable is intended for the LiteLLM proxy used by GPT Researcher in Stage 2). Strip it from the agent subprocess env via the lifted `_clean_env` and pass Codex's own auth via its config file or `-c` flags.

## 3. Bootstrap + alternating review loop

```
i=0: claude_code bootstrap (--permission-mode plan)
     prompt: BOOTSTRAP_PROMPT
     reads:  Stage1Config.repo_path, Stage1Config.modules_json (optional),
             Stage1Config.profile_paths (optional)
     emits:  candidates_v0.json (schema-validated)

i=1: codex review
     prompt: REVIEW_PROMPT(candidates_v0.json, role="adversarial reviewer")
     emits:  candidates_v1.json

i=2: claude review
     prompt: REVIEW_PROMPT(candidates_v1.json, role="adversarial reviewer")
     emits:  candidates_v2.json

... until stop condition ...
```

Stop conditions (ALL evaluated each iteration; ANY trigger ends the loop and sets `Stage1Result.stop_reason`):

1. **Normalized-JSON equality** (`"converged"`): `canonical_json(candidates_iN) == canonical_json(candidates_i{N-1})` after sorting by `(file, line_start, line_end)` and stripping whitespace/timestamps.
2. **"No changes" predicate** (`"no_changes"`): the agent's emitted `meta.delta` field is `{"added":[],"removed":[],"modified":[]}`.
3. **Cycle detection** (`"cycle"`): hash of canonical JSON has appeared before in this run.
4. **Max-iter cap** (`"max_iterations"`): `iteration >= Stage1Config.max_iterations` (default 6).
5. **Budget guard** (`"budget_exhausted"`): cumulative Stage-1 `cost_usd` ≥ `Stage1Config.budget_usd` (default 5.0).
6. **Hard-failure latch** (`"hard_failure"`): if any iteration produces invalid JSON twice in a row after retry, abort Stage 1 and emit the last good `candidates.json` on `Stage1Result.candidates`.

## 4. Scope bounding

Large modules need pruning. Strategy stack, tried in order:

1. `Stage1Config.scope_paths` — explicit globs, highest precedence.
2. `Stage1Config.modules_json` — if provided, prefer files inside the resolved `Module.path`; if a sibling pipeline produced a `{file: importance_score}` map, prioritize files with highest importance_score.
3. `Stage1Config.profile_paths` — profiling artifacts (`bench/profile.json`, `*.prof`, `*.pstats`, `*.nsys-rep`) listed in the bootstrap prompt as hot-path hints.
4. Guided enumeration: agent runs `ls`/`grep`/`fd` inside the module and proposes a scope before generating candidates (a planning sub-step inside the bootstrap prompt).

## 5. Prior art consulted for Stage 1 prompt design

- **Agentless** (https://arxiv.org/abs/2407.01489) — its file-localization → relevant-code-locations decomposition matches our two-stage scope-then-list pattern. Adopted: emit a "scope" sub-output before the candidate list (folded into the bootstrap prompt's "What to read first" step).
- **SWE-agent / SWE-bench solver families** (https://swe-agent.com) — their ACI primitives (open, search, scroll) inform what *not* to do: we don't replicate the agentic environment; the agent uses the underlying CLI's built-in file tools instead.
- **OpenHands** (https://github.com/All-Hands-AI/OpenHands) — evaluated as an M5 baseline. Its general "agent does everything in one loop" is exactly what we *don't* want for the cost-disciplined separation of research from coding.
- **KernelBench's prompting style** (https://scalingintelligence.stanford.edu/blogs/kernelbench/) — for GPU-kernel candidates, their explicit "replace this PyTorch op with a custom kernel" framing carries over: when a candidate's `rationale` flags a kernel launch site or fusion opportunity, the Stage-3b prompt should be augmented with KernelBench-style "rewrite this op" exemplars.
- **PIE (pie4perf)** and **FasterPy** — both prove that performance-aware retrieval (give the model exemplars of past optimization edits) significantly beats blind prompting. We borrow the *pattern* (Stage 2 retrieves real techniques rather than letting Stage 3b invent them blind) but keep the finding record minimal: just the URL and a one-paragraph `technique_summary`. The Stage-3b agent fetches the source itself when it needs implementation detail.

## 6. Validation per iteration

- pydantic schema parse → reject on failure (retry once with a strict-mode reminder appended to the prompt; a second failure trips the hard-failure latch and ends the loop with `stop_reason="hard_failure"`).
- Path existence: every `candidates[i].file` resolves to an existing file in the working checkout; bad paths → drop the candidate and continue (don't fail the run); count drops on `IterationTelemetry.dropped_invalid_paths`.
- Line range sanity: `1 ≤ line_start ≤ line_end ≤ file_line_count`; out-of-range candidates dropped the same way, counted on `IterationTelemetry.dropped_invalid_ranges`.
- Range size: `len(candidates)` must lie in `[Stage1Config.min_candidates, Stage1Config.max_candidates]`; out-of-range → retry once with strict reminder, otherwise treat as schema-invalid.

## 7. Concrete Stage-1 prompts (drop-in)

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
Constraint: `{min_candidates} ≤ len(candidates) ≤ {max_candidates}`. Do NOT explain
outside the JSON object.
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

## 8. CORAL: What to Lift, What to Skip

Source repo: https://github.com/Human-Agent-Society/CORAL (last verified 2026-05-13). Paths below reference its `main` README's Architecture section; confirm the exact tree to lift from before implementation (see Open Questions Q1).

| CORAL artifact | Decision | Reason |
|---|---|---|
| `coral/agent/runtime.py` (subprocess wrapper for Claude / Codex / OpenCode) | **Lift** — the Popen pattern, log-tailing thread, JSONL parsing of `stream-json` / `--json` | Single best-tested non-interactive wrapper; saves a week. |
| `coral/agent/manager.py` lifecycle (spawn, heartbeat, reboot on max_turns) | **Lift selectively** — keep spawn + max_turns reboot; **skip** heartbeat interrupts and `/loop`-style prompts | Our pipeline is finite, not evolutionary. |
| Agent class registry (`coral/agent/runtime.py` selects by `runtime: claude_code|codex|opencode` from YAML) | **Lift** as a thin `AgentRunner` protocol with two implementations | Same `runtime: …` config UX; lets us add `opencode` later. |
| `_clean_env` style env scrubbing (referenced in CORAL workspace setup) | **Lift** | Prevents `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, and telemetry envs from leaking into agent subprocesses unintentionally. **Important interaction:** the orchestrator process holds `OPENAI_API_KEY` / `OPENAI_BASE_URL` for the GPT Researcher call; the agent subprocesses (Claude Code, Codex) must receive their own credentials and *not* inherit the LiteLLM-proxy envs (Codex would otherwise try to call the LiteLLM endpoint instead of OpenAI). |
| Worktrees (`coral/workspace/setup.py` clones a git worktree per agent) | **Skip** for Stages 1 and 3 — both read-mostly; mount the repo read-only (or workspace-write of a separate artifacts dir). **Optionally use** if we ever auto-apply patches in v2. | Worktrees add ~3–5s/agent and a git checkpoint surface area we don't need. |
| `.coral/public/` shared knowledge hub, attempts, notes, skills | **Skip** | Only useful for evolutionary improvement loops — orthogonal to a one-shot candidate-discovery pass. |
| `TaskGrader` / `function_grader` (`coral/grader/`) | **Skip**. Replace with JSON-schema validation (pydantic) per stage. | We don't have a benchmark score; we have schema correctness + URL reachability + path existence. |
| `coral eval` post-commit hook | **Skip** | No commits. |
| `coral ui` web dashboard | **Skip** for MVP; revisit in v2 — the dashboard's leaderboard concept maps cleanly to "ranked changes per candidate". |
| Session resume across iterations (Claude `--resume <uuid>`, Codex `codex exec resume`) | **Skip in MVP**; document as a v2 optimization. The first iteration's full context fits well under the 200K context window of typical frontier models; subsequent reviews can re-prime with a compact handoff. | Resume reduces tokens but doubles the failure modes (session-id loss, model drift across resumes). |
