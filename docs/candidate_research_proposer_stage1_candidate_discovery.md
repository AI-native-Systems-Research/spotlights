# candidate_research_proposer — Stage 1: Candidate Discovery

Stage 1 produces a `Candidates` pydantic model — a list of falsifiable, high-yield optimization candidate locations inside the target module — through a Claude Code bootstrap followed by a fixed number of alternating Claude Code ↔ Codex review iterations. The Python API returns a `DiscoveryResult` wrapper whose `candidates` field is the canonical Stage-1 artifact.

## 1. Python API

Package: `spotlights_engine.candidate_discovery`. Stage 1 is invoked as a pure function. Input is the module's `ProjectTree.walk()` qualified name plus the resolved `Module` record from [src/spotlights_engine/schemas/modules.py](../src/spotlights_engine/schemas/modules.py); output is a `DiscoveryResult` whose `candidates` field also serializes to the `candidates.json` shown in [candidate_research_proposer_architecture.md](candidate_research_proposer_architecture.md#candidatesjson--stage-1-output). The driver does the `ProjectTree.from_json(modules_json).resolve(qn)` lookup once, raises a setup error if it returns `None`, and passes both `qn` and `Module` in.

```python
from pathlib import Path

from spotlights_engine.candidate_discovery import discover, DiscoveryConfig, DiscoveryResult
from spotlights_engine.schemas.modules import Module

result: DiscoveryResult = discover(DiscoveryConfig(
    repo_path=Path("/abs/path/to/repo"),
    module_qualified_name="v1/engine/core",
    module=Module(...),                  # resolved from modules.json by the driver
    artifacts_dir=Path(".../<run_id>/"),  # final artifact lands at <artifacts_dir>/candidates.json
))
```

### `DiscoveryConfig` (input)

```python
class DiscoveryConfig(BaseModel):
    repo_path: Path                        # local checkout root; must exist
    module_qualified_name: str             # ProjectTree.walk() key, e.g. "v1/engine/core"
    module: Module                         # resolved Module record (see schemas/modules.py)
    artifacts_dir: Path                    # shared run dir; final artifact is <artifacts_dir>/candidates.json

    num_review_iterations: int = Field(default=3, ge=0)  # N review passes after bootstrap; no early stop
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = "gpt-5.5"
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "high"
```

### `Candidates` and `DiscoveryResult` (output)

`Candidates` is the on-disk schema. The schema classes live in `spotlights_engine.schemas.candidate`; the discovery package imports them rather than defining a private copy. All shape constraints are enforced in pydantic so the exported `model_json_schema()` carries them into both subprocess validators:

```python
class Candidate(BaseModel):
    id: str = Field(pattern=r"^cand-\d{4}$")  # minted by bootstrap; monotone across reviews
    file: str                                 # repo-root-relative, non-absolute; MUST resolve under module.path
    line_start: int = Field(ge=1)             # 1-indexed inclusive
    line_end: int = Field(ge=1)               # 1-indexed inclusive
    rationale: str = Field(min_length=1, max_length=240)  # falsifiable

    @model_validator(mode="after")
    def _check_range(self) -> "Candidate":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Candidates(BaseModel):
    module_qualified_name: str
    candidates: list[Candidate] = Field(min_length=1)
```

`DiscoveryResult` wraps `Candidates` with the loop telemetry:

```python
class IterationTelemetry(BaseModel):
    n: int                                 # 0 = bootstrap, 1..N = reviews
    agent: Literal["claude_code", "codex"]
    session_id: str | None = None
    duration_s: float
    cost_usd: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    candidate_count: int                   # post-drop validated count
    schema_retries: int = 0
    dropped_outside_module: int = 0        # candidate.file resolves outside module.path
    dropped_missing_file: int = 0          # candidate.file does not exist in repo_path
    dropped_invalid_ranges: int = 0        # line_start/line_end out of file bounds
    added: list[str]                       # ids new vs prev iter; for n=0, the full bootstrap list
    removed: list[str]                     # ids in prev iter but not this one; empty for n=0
    modified: list[str]                    # ids present in both iters with changed line_start, line_end, or rationale; empty for n=0


class DiscoveryResult(BaseModel):
    candidates: Candidates
    iterations: list[IterationTelemetry]
    total_duration_s: float
    total_cost_usd: float | None = None
```

`discover` also persists `<artifacts_dir>/candidates.json` (the canonical artifact consumed by Stage 3a) and a per-iteration `candidate_discovery/iterations.jsonl` telemetry file; the in-memory result and on-disk artifacts are equivalent. The loop always runs `1 + num_review_iterations` iterations and never resumes (a pre-existing `<artifacts_dir>/candidate_discovery/` is a setup error, not a resume point). Failure modes inside an iteration split three ways:

- **schema-parse failure** retries once with a strict-mode reminder appended (§6.2); a second failure raises `DiscoveryValidationError`.
- **protocol checks** — target-repo mutation (§6.1), qualified-name mismatch (§6.3), ID uniqueness/monotonicity (§6.7) — raise immediately.
- **content drops** — module containment (§6.4), path existence (§6.5), line-range sanity (§6.6) — drop the offending candidate and increment a counter; the iteration continues.

## 2. Bootstrap + alternating review loop

```
i=0: claude_code bootstrap            → candidates_v0.json
i=1: codex      review (adversarial)  → candidates_v1.json
i=2: claude_code review                → candidates_v2.json
i=3: codex      review                 → candidates_v3.json   # for num_review_iterations=3
```

The bootstrap is always `claude_code`. Review iteration `N` (1-indexed) uses `agents[(N-1) % 2]` with `agents = ["codex", "claude_code"]`, so the bootstrap is reviewed first by the *other* agent. With the default `num_review_iterations=3` the full rotation is `claude_code → codex → claude_code → codex`. Each review reads the prior iteration's full `candidates.json`, prunes false positives, merges duplicates, sharpens rationales, and may add at most 5 new candidates.

Iteration `num_review_iterations` is the downstream-facing result: its post-drop `candidates.json` is copied to `<artifacts_dir>/candidates.json` (§6.8).

## 3. Run directory layout

Stage 1 receives the shared run dir as `artifacts_dir` — conventionally `~/.cache/optquest/<repo_slug>/<run_id>/` per the architecture doc. The final, Stage-3a-facing artifact is `<artifacts_dir>/candidates.json`; all Stage 1 scratch logs live under `<artifacts_dir>/candidate_discovery/`. The driver creates `<run_id>/` and passes it in; the stage mints `<artifacts_dir>/candidate_discovery/` itself.

```
<artifacts_dir>/
  candidates.json            # final validated candidates from the last iteration
  candidate_discovery/
    candidates.schema.json   # exported from the pydantic schema for subprocess validation
    iterations.jsonl
    iter_0_bootstrap/
      candidates.json        # orchestrator-written, post-drop validated JSON for this iteration
      last_message.json      # exact final-assistant-message JSON string, pre-validation
      prompt.md              # the exact wrapped prompt sent
      raw_stdout.log
      raw_stderr.log
    iter_1_codex/
      candidates.json
      last_message.json
      diff_from_prev.md      # human-readable structural diff vs the previous iteration (present in every review iter, i >= 1)
      prompt.md
      raw_stdout.log
      raw_stderr.log
    iter_2_claude_code/      # same files as iter_1_codex
    iter_3_codex/            # same files as iter_1_codex
```


## 4. Subprocess invocation

Before the first iteration the orchestrator exports the `Candidates` pydantic schema via `model_json_schema()` and writes it to `<artifacts_dir>/candidate_discovery/candidates.schema.json`; both subprocesses then receive it (Claude inline via `--json-schema`, Codex by path via `--output-schema`) so the same shape constraints are enforced model-side as well as orchestrator-side.

The orchestrator never inlines the prompt in argv (Linux `MAX_ARG_STRLEN` = 128 KB). Both agents receive the prompt on stdin and return one JSON object as the final assistant message. In every iteration the orchestrator: (a) captures or directs the CLI to capture the final message verbatim to `<iter_dir>/last_message.json`, (b) runs the §6 validation pipeline, (c) writes the post-drop result to `<iter_dir>/candidates.json`, and (d) on the last iteration only, copies that file to `<artifacts_dir>/candidates.json`. The subprocesses never write any `candidates.json`; Codex is the one exception for `last_message.json`, because `--output-last-message` is its supported non-interactive output path.

**Claude Code** ([cli-reference](https://code.claude.com/docs/en/cli-reference), [permission-modes](https://code.claude.com/docs/en/permission-modes)):

```
SCHEMA_PATH=<artifacts_dir>/candidate_discovery/candidates.schema.json
claude -p \
  --output-format stream-json --verbose \
  --json-schema "$(<"$SCHEMA_PATH")" \
  --permission-mode plan \
  --max-turns ${claude_max_turns} \
  < prompt.md
```

Run Claude with `cwd=<repo_path>`. `--permission-mode plan` keeps the agent read-only against the target repo. The schema is small (a few KB) so inline-via-shell is well under `MAX_ARG_STRLEN`. The terminal `{"type":"result",…,"total_cost_usd":…,"session_id":…}` NDJSON event supplies `session_id`, duration, and cost when available; the orchestrator writes the `result` payload string verbatim to `<iter_dir>/last_message.json` and then runs §6 over it.

**Codex** ([cli reference](https://developers.openai.com/codex/cli/reference), [noninteractive](https://developers.openai.com/codex/noninteractive)):

```
codex exec - \
  --json \
  --output-last-message <iter_dir>/last_message.json \
  --output-schema <artifacts_dir>/candidate_discovery/candidates.schema.json \
  --sandbox read-only \
  -C <repo_path> \
  -c model='"${codex_model}"' \
  -c model_reasoning_effort='"${codex_reasoning_effort}"' \
  < prompt.md
```

`codex exec -` reads stdin. `-C <repo_path>` makes repo-root-relative paths natural; `--sandbox read-only` blocks target-repo writes from model-issued commands. The Codex CLI itself writes `--output-last-message` to the absolute `<iter_dir>/last_message.json` (containing the final assistant message). `--output-schema` validates that message against `candidates.schema.json` before the CLI exits, so a malformed shape is caught both by Codex *and* by the orchestrator. Resource asymmetry is intentional: Claude is capped via `--max-turns` while Codex has no equivalent turn cap and relies on `model_reasoning_effort` to bound depth. Both subprocesses share the same wall-clock kill at `per_iteration_wallclock_s`, enforced by the orchestrator (`subprocess.run(..., timeout=…)` with the iteration counted as a schema-parse failure on timeout, then retried once per §6.2).

`cwd` semantics are asymmetric by CLI design (Claude is run with `cwd=<repo_path>`; Codex with `-C <repo_path>`); both put repo-root-relative paths at the root of each model's working set.

**Env scrubbing.** Both subprocesses run under a CORAL-style `_clean_env` that, in addition to passing through the host PATH and each agent's own credentials, drops:

- `OPENAI_BASE_URL`, `OPENAI_API_BASE` — reserved for Stage 2's LiteLLM proxy; would otherwise reroute Codex.
- `ANTHROPIC_BASE_URL` — same reasoning for Claude.
- `OPTQUEST_*`, `SPOTLIGHTS_*` — orchestrator telemetry namespaces that must not leak into agent context.
- `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY` — proxies pinned by the orchestrator, not the subprocess.

## 5. Prompt wrapper contract

The orchestrator wraps every prompt with a fixed preamble pinning the I/O contract:

```
<preamble>
You are running as a non-interactive subprocess. Your current working directory
is the target repository — do NOT modify it; read only.
- Return exactly one valid JSON object as your final assistant message.
- Do not write files. The orchestrator writes candidates.json after validation.
- No markdown fence, no commentary, no "DONE" sentinel.
</preamble>

<the verbatim bootstrap or review prompt below>
```

Before substitution into either prompt, the orchestrator pre-formats the `Module` fields into agent-friendly strings:

- `{main_files}` → bulleted lines, one per `File`: `- src/foo/x.py (entrypoint)`.
- `{submodule_names}` → comma-joined `m.name for m in module.submodules`, or `"(none)"`.
- `{depends_on}` → comma-joined `module.depends_on` (each entry is another module's qualified name, **not** a filesystem path), or `"(none)"`.

For review iterations the orchestrator additionally inlines the prior iteration's `candidates.json` content into the review prompt (the `@candidates_prev.json` reference in §9) before wrapping and substitutes `{max_seen_candidate_id}` with the highest candidate id accepted in any earlier iteration.

## 6. Validation per iteration

Run after each subprocess exits, in this order. Steps marked **(raise)** abort the run via `DiscoveryValidationError` / `DiscoveryMutationError` (both in `spotlights_engine.candidate_discovery.errors`); steps marked **(drop)** prune the offending candidate and increment a counter on `IterationTelemetry`.

1. **Target-repo mutation guard** **(raises `DiscoveryMutationError`)** — diff the repo's `git status --porcelain=v1 -z` (or a lightweight `mtime+size+xattr` manifest if `repo_path` is not a git work tree) before and after each subprocess, before trusting the emitted candidates or retrying the prompt. Coverage is limited to `repo_path`; the orchestrator does not detect writes outside this subtree, but `--permission-mode plan` and `--sandbox read-only` are the primary defenses there.
2. **Schema parse** — `Candidates.model_validate_json()` over the captured final assistant message. On failure, retry the iteration once with a strict-mode reminder appended to the prompt; the retry increments `schema_retries`. A second failure **(raises)**.
3. **Qualified-name check** **(raises)** — `Candidates.module_qualified_name` must equal `config.module_qualified_name`.
4. **Module containment** **(drop → `dropped_outside_module`)** — build `module_root = (repo_path / config.module.path).resolve(strict=False)`. Every `candidate.file` must be a relative path, and `(repo_path / candidate.file).resolve(strict=False)` must lie under `module_root`.
5. **Path existence** **(drop → `dropped_missing_file`)** — every surviving `candidate.file` must resolve to an existing file in `repo_path`.
6. **Line range sanity** **(drop → `dropped_invalid_ranges`)** — `1 ≤ line_start ≤ line_end ≤ file_line_count`. (`line_end ≥ line_start` is already enforced by the pydantic validator.)
7. **ID integrity** **(raises)** — within the raw parsed iteration, `id` values are unique. Across review iterations (`n ≥ 1`):
   - on the raw parsed list, any `id` carried over from the previous iteration MUST retain the previous iteration's `file` (rationale and line range may change);
   - on the post-drop list, an `id` absent from the previous iteration is treated as newly minted, even if it appeared in an older iteration;
   - every newly minted `id` MUST be strictly greater than the highest id accepted in any earlier iteration under lexicographic order on the zero-padded `cand-NNNN` form.
8. **Persist normalized JSON** **(raises if empty)** — re-validate the surviving list via `Candidates.model_validate(dict(module_qualified_name=..., candidates=survivors))` so the `min_length=1` invariant fires on the *current* iteration if drops emptied it (raises `DiscoveryValidationError`). Then serialize via `model_dump_json(indent=2)` to `<iter_dir>/candidates.json`. On the final iteration (`n == num_review_iterations`), copy that file byte-for-byte to `<artifacts_dir>/candidates.json`.

After step 8 the orchestrator also computes `IterationTelemetry.added` / `removed` / `modified` by joining on `id` against the previous iteration: `added = ids_n \ ids_{n-1}`, `removed = ids_{n-1} \ ids_n`, `modified = { id in ids_n ∩ ids_{n-1} : (line_start, line_end, rationale.strip()) differs }`. For `n=0`, `added` is the full bootstrap list and the other two are empty. For review iterations (`n >= 1`), the same structural comparison is written to `<iter_dir>/diff_from_prev.md`.

## 7. Loop termination

The loop runs exactly `1 + num_review_iterations` iterations (bootstrap at `i=0`, reviews at `i=1..num_review_iterations`) and then returns. The final `candidates.json` is the **last successfully validated** iteration — by construction that is iteration `num_review_iterations`, since validation failures raise. There is no convergence, cycle, or budget logic: the iteration count is the only stop signal.

## 8. Bootstrap prompt

`prompts/bootstrap.md`:

````
You are a senior performance engineer auditing ONE module of a (likely Python,
possibly with C++/CUDA) codebase for *high-yield optimization opportunities*.
You do NOT write fixes. You enumerate candidate locations, each with a
falsifiable rationale.

## Module under audit
- module_qualified_name: {module_qualified_name}
- module.name:           {module_name}
- module.path:           {module_path}                   # all candidate files MUST lie under this
- module.description:    {module_description}
- module.depends_on:     {depends_on}                    # qualified names of sibling modules, NOT filesystem paths
- module.main_files:                                     # start here, in order
{main_files}
- submodules:            {submodule_names}               # nested modules under this one

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
3. `depends_on` lists other modules' qualified names (e.g. `v1/engine/scheduler`),
   not paths. Use them only as call-shape context; do NOT guess dependency paths
   and do NOT propose candidates outside `module.path`.

## Output (REQUIRED — strict JSON, no markdown fence)
Emit ONE JSON object matching the `Candidates` schema enforced by the wrapper:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [
    {
      "id": "cand-0001",
      "file": "<repo-root-relative path under {module_path}>",
      "line_start": <1-indexed inclusive>,
      "line_end":   <1-indexed inclusive, >= line_start>,
      "rationale":  "<<=240 chars; cite the loop / the alloc / the sync>"
    }
  ]
}

Constraints:
- Aim for 5–40 candidates; quality beats quantity. Do not pad to hit a count.
- Every `file` MUST be under `{module_path}` and exist in the checkout.
- `id` values must be unique within this list; use `cand-NNNN` zero-padded,
  starting at `cand-0001` and increasing monotonically.
- Do NOT explain outside the JSON object.
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
- module.path: {module_path}                     # every candidate's `file` MUST live here
- highest id accepted so far: {max_seen_candidate_id}

## Rules
- DO NOT inflate the list. If the previous pass was good, return it nearly unchanged.
- Remove any candidate whose `file` does not exist, whose `file` is outside
  `module.path`, or whose `line_start..line_end` is out of range for that file.
- Merge candidates that point at the same hot path with overlapping line ranges
  (keep one `id`, drop the others).
- For each KEPT candidate you MAY tighten `rationale` and adjust `line_start` /
  `line_end`. You MUST NOT change `id` or `file`. If you believe a kept
  candidate's `file` is wrong, drop the old `id` and mint a new one instead.
- For each NEW candidate, the rationale must cite a specific code construct
  (function name, loop, alloc call, sync primitive) — not a vague category.
- Mint new ids strictly greater than `{max_seen_candidate_id}`,
  zero-padded to four digits.
- Add at most 5 new candidates per review pass.

## Output (REQUIRED — strict JSON, no markdown fence)
Same schema as the previous pass:
{
  "module_qualified_name": "{module_qualified_name}",
  "candidates": [ ... ]
}

If you genuinely have no changes, return the previous list unchanged. Do NOT pad.
Do NOT explain outside the JSON object.
````
