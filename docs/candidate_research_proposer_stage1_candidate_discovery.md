# candidate_research_proposer — Stage 1: Candidate Discovery

Stage 1 returns a `Candidates` pydantic model — a list of falsifiable, high-yield optimization candidate locations inside the target module — through a Claude Code bootstrap followed by alternating Claude Code ↔ Codex review iterations. The model is also serialized to `candidates.json` as the on-disk artifact consumed by Stage 3a. The pattern is lifted from [docs/old/modules_plan.md](old/modules_plan.md) and narrowed to a single module: bootstrap drafts, reviewers prune/sharpen/add, stop on convergence or cycle.

## 1. Python API

Package: `spotlights_engine.candidates_discovery`. Stage 1 is invoked as a pure function. Input is a resolved `Module` record from [src/spotlights_engine/schemas/modules.py](../src/spotlights_engine/schemas/modules.py); output is a `Candidates` pydantic model that also serializes to the `candidates.json` shown in [candidate_research_proposer_architecture.md](candidate_research_proposer_architecture.md#candidatesjson--stage-1-output). The driver does the `ProjectTree.from_json(modules_json).resolve(qn)` lookup once and passes the `Module` in.

```python
from spotlights_engine.candidates_discovery import discover, DiscoveryConfig, DiscoveryResult
from spotlights_engine.schemas.modules import Module

result: DiscoveryResult = discover(DiscoveryConfig(
    repo_path=Path("/abs/path/to/repo"),
    module_qualified_name="foo/bar",
    module=Module(...),                  # resolved from modules.json by the driver
    artifacts_dir=Path(".../<run_id>/candidates_discovery/"),
))
# result.candidates       — pydantic Candidates (mirrors candidates.json)
# result.iterations       — per-iteration telemetry
# result.stop_reason      — why the loop ended
```

### `DiscoveryConfig` (input)

```python
class DiscoveryConfig(BaseModel):
    repo_path: Path                        # local checkout root; must exist
    module_qualified_name: str             # ProjectTree.walk() key, e.g. "foo/bar"
    module: Module                         # resolved Module record (see schemas/modules.py)
    artifacts_dir: Path                    # candidates.json + per-iter scratch land here

    max_review_iterations: int = 4         # 4 reviews + 1 bootstrap = 5 total runs
    min_candidates: int = 5
    max_candidates: int = 40
    budget_usd: float = 5.0
    per_iteration_wallclock_s: int = 600
```

### `Candidates` and `DiscoveryResult` (output)

`Candidates` is the on-disk schema:

```python
class Candidate(BaseModel):
    id: str                                # "cand-0001", minted by the bootstrap agent
    file: str                              # repo-root-relative; MUST be under module.path
    line_start: int                        # 1-indexed inclusive
    line_end: int                          # 1-indexed inclusive, ≥ line_start
    rationale: str                         # ≤ 240 chars, falsifiable

class Candidates(BaseModel):
    module_qualified_name: str
    candidates: list[Candidate]
```

`DiscoveryResult` wraps `Candidates` with the loop telemetry:

```python
class IterationTelemetry(BaseModel):
    n: int                                 # 0 = bootstrap, 1..N = reviews
    agent: Literal["claude_code", "codex"]
    duration_s: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    candidate_count: int
    added: list[str]                       # candidate ids added vs prev iter
    removed: list[str]                     # candidate ids removed vs prev iter
    modified: list[str]                    # candidate ids whose range/rationale changed
    dropped_invalid_paths: int
    dropped_invalid_ranges: int

class DiscoveryResult(BaseModel):
    candidates: Candidates
    iterations: list[IterationTelemetry]
    total_cost_usd: float
    total_duration_s: float
    stop_reason: Literal["converged", "cycle", "max_iterations", "budget_exceeded", "hard_failure"]
```

`discover` also persists `candidates.json` (the canonical artifact consumed by Stage 3a) and a per-iteration `iterations.jsonl` telemetry file under `config.artifacts_dir`; the in-memory result and on-disk artifacts are equivalent.

## 2. Bootstrap + alternating review loop

```
i=0: claude_code bootstrap          → candidates_v0.json
i=1: codex      review (adversarial) → candidates_v1.json
i=2: claude_code review              → candidates_v2.json
i=3: codex      review               → candidates_v3.json
...                                    until stop condition fires
```

`agents[(N-1) % 2]` picks the runtime for review iteration N; the default rotation is `["codex", "claude_code", "codex", "claude_code"]` so the bootstrap is reviewed first by the *other* agent. Each review reads the prior iteration's full `candidates.json`, prunes false positives, merges duplicates, sharpens rationales, and may add at most 5 new candidates.

## 3. Run directory layout

Stage 1 owns its `artifacts_dir`, conventionally `~/.cache/optquest/<repo_slug>/<run_id>/candidates_discovery/` per the architecture doc. The driver creates `<run_id>/` and passes `<run_id>/candidates_discovery/` in.

```
<artifacts_dir>/
  iter_0_bootstrap/
    candidates.json        # validated candidates from claude_code
    prompt.md              # the exact wrapped prompt sent
    raw_stdout.log
    raw_stderr.log
    cost.json
  iter_1_codex/
    candidates.json
    diff_from_prev.md      # human-readable structural diff vs iter_0
    prompt.md
    raw_stdout.log
    raw_stderr.log
    cost.json
  iter_2_claude_code/...
  candidates.json          # copy of the last successfully validated iter — the Stage 1 contract output
  iterations.jsonl         # one IterationTelemetry per line
```

`candidates.json` at the top of `artifacts_dir` is the only path Stage 3a is contracted to read; the per-iter directories are for debugging.

## 4. Subprocess invocation

The orchestrator never inlines the prompt in argv (Linux `MAX_ARG_STRLEN` = 128 KB). Both agents receive the prompt on stdin and write `candidates.json` to their iteration directory.

**Claude Code** ([cli-reference](https://code.claude.com/docs/en/cli-reference), [permission-modes](https://code.claude.com/docs/en/permission-modes)):

```
claude -p \
  --output-format stream-json --verbose \
  --input-format stream-json \
  --permission-mode plan \
  --max-turns 30 \
  --add-dir <iter_dir> \
  < prompt.md
```

`--permission-mode plan` keeps the agent read-only against the target repo; `--add-dir <iter_dir>` is the single writable root for `candidates.json`. The terminal `{"type":"result",…,"total_cost_usd":…,"session_id":…}` NDJSON event drives `IterationTelemetry`.

**Codex** ([cli reference](https://developers.openai.com/codex/cli/reference), [noninteractive](https://developers.openai.com/codex/noninteractive)):

```
codex exec - \
  --json \
  --output-last-message <iter_dir>/last_message.txt \
  --output-schema candidates.schema.json \
  --sandbox workspace-write \
  -C <iter_dir> \
  --add-dir <repo_path> \
  -c model='"gpt-5.5"' \
  -c model_reasoning_effort='"high"' \
  < prompt.md
```

`codex exec -` reads stdin. `--sandbox read-only` plus `--add-dir` is *not* a supported "read everywhere, write here" combo (per the Codex docs); we make `<iter_dir>` the workspace and add the repo read-only via `--add-dir <repo_path>`. `--output-schema` validates the model's final message against `candidates.schema.json` (exported by `optquest.schema`).

**Env scrubbing.** Both subprocesses run under a CORAL-style `_clean_env`: strip `OPENAI_BASE_URL` (reserved for Stage 2's LiteLLM proxy) and any orchestrator-only telemetry vars before exec. Each agent uses its own credentials.

## 5. Prompt wrapper contract

The orchestrator wraps every prompt with a fixed preamble pinning the I/O contract:

```
<preamble>
You are running as a non-interactive subprocess. Your current working directory
is the target repository — do NOT modify it; read only.
- Write valid JSON only (no markdown fence) to: <ITER_DIR>/candidates.json
- Make your final assistant message exactly "DONE" when the file is written. No commentary.
</preamble>

<the verbatim bootstrap or review prompt below>
```

For review iterations the orchestrator additionally inlines the prior iteration's `candidates.json` content into the review prompt (the `@candidates_prev.json` reference in §9) before wrapping.

## 6. Validation per iteration

Run after each subprocess exits:

1. **Schema parse** — `Candidates.model_validate_json()`. On failure, retry once with a strict-mode reminder appended to the prompt; second failure trips the hard-failure latch.
2. **Module containment** — every `candidate.file` must lie under `Module.path` (the resolved record's filesystem path). Violators are dropped; count goes to `IterationTelemetry.dropped_invalid_paths`.
3. **Path existence** — every `candidate.file` must resolve to an existing file in `repo_path`. Same drop-and-count behavior.
4. **Line range sanity** — `1 ≤ line_start ≤ line_end ≤ file_line_count`. Same drop-and-count, on `dropped_invalid_ranges`.
5. **Size guard** — `min_candidates ≤ len(candidates) ≤ max_candidates`. Out of range → retry once with the size constraint reiterated; second failure trips the hard-failure latch.
6. **Target-repo mutation guard** — diff the repo's `git status --porcelain=v1 -z` (or a lightweight manifest if not a git repo) before and after each subprocess; any change outside `<iter_dir>` is a protocol violation and ends the loop with `stop_reason="hard_failure"`.

## 7. Stop conditions

Evaluated after each iteration N; the first to fire wins, recorded on `DiscoveryResult.stop_reason`. The final `candidates.json` is always the **last successfully validated** iteration.

- **Converged** — `canonical_json(iter_N.candidates) == canonical_json(iter_{N-1}.candidates)`, where canonicalization sorts `candidates[]` by `(file, line_start, line_end, id)` and uses `json.dumps(sort_keys=True, separators=(",",":"))`.
- **Cycle detected** — the canonical hash of iter N matches any prior iter in `[max(0, N-4), N-1]`. Catches Claude↔Codex ping-pong.
- **Max iterations** — `N >= max_review_iterations`.
- **Budget exceeded** — `sum(cost_usd) + projected_next_iter_cost > budget_usd`, projecting from the mean of prior iters.
- **Hard failure** — two consecutive iterations failed validation, or the mutation guard tripped.

## 8. Bootstrap prompt

`prompts/bootstrap.md`:

````
You are a senior performance engineer auditing ONE module of a (likely Python,
possibly with C++/CUDA) codebase for *high-yield optimization opportunities*.
You do NOT write fixes. You enumerate candidate locations, each with a
falsifiable rationale.

## Module under audit
- module_qualified_name: {module_qualified_name}
- module.name:           {module.name}
- module.path:           {module.path}                   # all candidate files MUST lie under this
- module.description:    {module.description}
- module.depends_on:     {module.depends_on}
- module.main_files:     {module.main_files}             # path + role; start here
- submodules:            {module.submodules names}       # nested modules under this one

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
1. Read every file in `module.main_files` (they are the entry points by design).
2. Walk `module.path` and read the rest of the module's source.
3. Cross-reference `depends_on` only to understand call shapes — do NOT propose
   candidates outside `module.path`.

## Output (REQUIRED — strict JSON, no markdown fence)
Emit ONE JSON object matching @candidates.schema.json:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [
    {
      "id": "cand-0001",
      "file": "<repo-root-relative path under {module.path}>",
      "line_start": <1-indexed inclusive>,
      "line_end":   <1-indexed inclusive, ≥ line_start>,
      "rationale":  "<≤240 chars; cite the loop / the alloc / the sync>"
    }
  ]
}

Constraints:
- {min_candidates} ≤ len(candidates) ≤ {max_candidates}
- Every `file` MUST be under `{module.path}` and exist in the checkout.
- `id` values must be unique within this list; use `cand-NNNN` zero-padded.
- Do NOT explain outside the JSON object. End the assistant message with "DONE"
  after writing the file.
````

## 9. Review prompt

`prompts/review.md`:

````
You are reviewing another agent's candidate list for the same module. Your job
is adversarial: prune false positives, merge duplicates, sharpen rationales, and
add at most 5 *new* candidates the previous pass missed.

## Inputs
- previous candidates: @candidates_prev.json     # inlined by the orchestrator
- module_qualified_name: {module_qualified_name}
- module.path: {module.path}                     # every candidate's `file` MUST live here

## Rules
- DO NOT inflate the list. If the previous pass was good, return it nearly unchanged.
- Remove any candidate whose `file` does not exist, whose `file` is outside
  `module.path`, or whose `line_start..line_end` is out of range for that file.
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others).
- For each kept candidate you MAY tighten `rationale` or adjust `line_start` /
  `line_end`. Do NOT change `id` for kept candidates.
- For each NEW candidate, the rationale must cite a specific code construct
  (function name, loop, alloc call, sync primitive) — not a vague category.
- Mint new ids continuing from the previous pass's highest `cand-NNNN`.

## Output (REQUIRED — strict JSON, no markdown fence)
Same schema as the previous pass:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [ ... ]
}

If you genuinely have no changes, return the previous list unchanged. Do NOT pad.
End the assistant message with "DONE" after writing the file.
````
