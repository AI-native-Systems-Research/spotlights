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
    artifacts_dir: Path                    # shared run dir outside repo_path; final artifact is <artifacts_dir>/candidates.json
    repo_context_markdown: str | None = Field(default=None, min_length=1, max_length=20_000)

    num_review_iterations: int = Field(default=3, ge=0)  # N review passes after bootstrap; no early stop
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = "gpt-5.5"
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"
```

`repo_context_markdown` is optional repo-level markdown context (test commands, benchmark harnesses, metric conventions, hard constraints) that the orchestrator inlines into every iteration's prompt so the agent does not have to rediscover that surface each run. Sourcing is the caller's problem; the recommended authoring prompt is at [repo_context.txt](repo_context.txt). When supplied, the orchestrator persists the markdown verbatim to `<artifacts_dir>/candidate_discovery/repo_context.md` for run reproducibility. When `None`, the corresponding prompt section renders `_(none provided)_` and no file is written.

### `Candidates` and `DiscoveryResult` (output)

`Candidates` is the on-disk schema. The schema classes live in `spotlights_engine.schemas.candidate`; the discovery package imports them rather than defining a private copy. All shape constraints are enforced in pydantic so the exported `model_json_schema()` carries them into both subprocess validators:

```python
CandidateKind = Literal[
    "function", "method", "loop", "region", "kernel", "config_block", "plugin_seam",
]
EstimatedImpact = Literal["high", "medium", "low"]


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")    # minted by bootstrap; monotone across reviews
    file: str                                   # repo-root-relative, non-absolute; MUST resolve under module.path
    line_start: int = Field(ge=1)               # 1-indexed inclusive
    line_end: int = Field(ge=1)                 # 1-indexed inclusive
    symbol: str = Field(min_length=1, max_length=200)
    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    estimated_impact: EstimatedImpact
    estimated_impact_explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_range(self) -> "Candidate":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self


class Candidates(BaseModel):
    model_config = ConfigDict(extra="forbid")    # emits `additionalProperties: false` for codex --output-schema

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
    added: list[str]                       # ids new vs prev iter; for n=0, all bootstrap ids; sorted
    removed: list[str]                     # ids in prev iter but not this one; empty for n=0
    modified: list[str]                    # ids present in both iters whose structural content changed (see §6); empty for n=0


class DiscoveryResult(BaseModel):
    candidates: Candidates
    iterations: list[IterationTelemetry]
    total_duration_s: float
    total_cost_usd: float | None = None
```

`discover` also persists `<artifacts_dir>/candidates.json` (the canonical artifact consumed by Stage 3a) and a per-iteration `candidate_discovery/iterations.jsonl` telemetry file; the in-memory result and on-disk artifacts are equivalent. The loop always runs `1 + num_review_iterations` iterations and never resumes (a pre-existing `<artifacts_dir>/candidate_discovery/` is a setup error, not a resume point). `artifacts_dir` must resolve outside `repo_path`; otherwise Stage 1's own artifact writes would mutate the target checkout it is supposed to guard. Failure modes inside an iteration split three ways:

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

The bootstrap is always `claude_code`. Review iteration `N` (1-indexed) uses `agents[(N-1) % 2]` with `agents = ["codex", "claude_code"]`, so the bootstrap is reviewed first by the *other* agent. With the default `num_review_iterations=3` the full rotation is `claude_code → codex → claude_code → codex`. Each review reads the prior iteration's full `candidates.json`, prunes false positives, merges duplicates, sharpens rationales and impact explanations, and adds any further candidates that clear the same evolve-target quality bar. No cap on the number of additions — quality remains the only gate.

Iteration `num_review_iterations` is the downstream-facing result: its post-drop `candidates.json` is copied to `<artifacts_dir>/candidates.json` (§6.8).

## 3. Run directory layout

Stage 1 receives the shared run dir as `artifacts_dir` — conventionally `~/.cache/optquest/<repo_slug>/<run_id>/` per the architecture doc, and always outside the target checkout. The final, Stage-3a-facing artifact is `<artifacts_dir>/candidates.json`; all Stage 1 scratch logs live under `<artifacts_dir>/candidate_discovery/`. The driver creates `<run_id>/` and passes it in; the stage mints `<artifacts_dir>/candidate_discovery/` itself.

```
<artifacts_dir>/
  candidates.json            # final validated candidates from the last iteration
  candidate_discovery/
    candidates.schema.json   # exported from the pydantic schema for subprocess validation
    repo_context.md          # verbatim copy of config.repo_context_markdown; present only when supplied
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

Both the bootstrap and the review template carry a dedicated `## Repository context` section with a single `{repo_context}` placeholder. The orchestrator substitutes `config.repo_context_markdown` verbatim (no trim, no reflow) when supplied, and the literal string `_(none provided)_` when it is `None`. The section header is always present so the rendered template shape is stable across runs.

## 6. Validation per iteration

Run after each subprocess exits, in this order. Steps marked **(raise)** abort the run via `DiscoveryValidationError` / `DiscoveryMutationError` (both in `spotlights_engine.candidate_discovery.errors`); steps marked **(drop)** prune the offending candidate and increment a counter on `IterationTelemetry`.

1. **Target-repo mutation guard** **(raises `DiscoveryMutationError`)** — diff the repo's `git status --porcelain=v1 -z` (or a lightweight `mtime+size+xattr` manifest if `repo_path` is not a git work tree) before and after each subprocess, before trusting the emitted candidates or retrying the prompt. Coverage is limited to `repo_path`; the orchestrator does not detect writes outside this subtree, but `--permission-mode plan` and `--sandbox read-only` are the primary defenses there.
2. **Schema parse** — `Candidates.model_validate_json()` over the captured final assistant message. On failure, retry the iteration once with a strict-mode reminder appended to the prompt; the retry increments `schema_retries`. A second failure **(raises)**.
3. **Qualified-name check** **(raises)** — `Candidates.module_qualified_name` must equal `config.module_qualified_name`.
4. **Module containment** **(drop → `dropped_outside_module`)** — build `module_root = (repo_path / config.module.path).resolve(strict=False)`. Every `candidate.file` must be a relative path, and `(repo_path / candidate.file).resolve(strict=False)` must lie under `module_root`.
5. **Path existence** **(drop → `dropped_missing_file`)** — every surviving `candidate.file` must resolve to an existing file in `repo_path`.
6. **Line range sanity** **(drop → `dropped_invalid_ranges`)** — `1 ≤ line_start ≤ line_end ≤ file_line_count`. (`line_end ≥ line_start` is already enforced by the pydantic validator.)
7. **ID integrity** **(raises)** — within the raw parsed iteration, `id` values are unique. Across review iterations (`n ≥ 1`):
   - on the raw parsed list, any `id` carried over from the previous iteration MUST retain the previous iteration's `file` (other fields — line range, `symbol`, `kind`, `description`, `current_approach`, `evolve_rationale`, `estimated_impact`, `estimated_impact_explanation` — may change);
   - on the post-drop list, an `id` absent from the previous iteration is treated as newly minted, even if it appeared in an older iteration;
   - every newly minted `id` MUST be strictly greater than the highest id accepted in any earlier iteration under lexicographic order on the zero-padded `cand-NNNN` form.
8. **Persist normalized JSON** **(raises if empty)** — re-validate the surviving list via `Candidates.model_validate(dict(module_qualified_name=..., candidates=survivors))` so the `min_length=1` invariant fires on the *current* iteration if drops emptied it (raises `DiscoveryValidationError`). Then serialize via `model_dump_json(indent=2)` to `<iter_dir>/candidates.json`. On the final iteration (`n == num_review_iterations`), copy that file byte-for-byte to `<artifacts_dir>/candidates.json`.

After step 8 the orchestrator also computes `IterationTelemetry.added` / `removed` / `modified` by joining on `id` against the previous iteration: `added = ids_n \ ids_{n-1}`, `removed = ids_{n-1} \ ids_n`, and `modified = { id in ids_n ∩ ids_{n-1} : any of (line_start, line_end, kind, estimated_impact, symbol.strip(), description.strip(), current_approach.strip(), evolve_rationale.strip(), estimated_impact_explanation.strip()) differs }`. The emitted ID lists are sorted lexicographically for deterministic telemetry. For `n=0`, `added` is the full bootstrap ID list sorted the same way and the other two are empty. For review iterations (`n >= 1`), the same structural comparison is written to `<iter_dir>/diff_from_prev.md`.

## 7. Loop termination

The loop runs exactly `1 + num_review_iterations` iterations (bootstrap at `i=0`, reviews at `i=1..num_review_iterations`) and then returns. The final `candidates.json` is the **last successfully validated** iteration — by construction that is iteration `num_review_iterations`, since validation failures raise. There is no convergence, cycle, or budget logic: the iteration count is the only stop signal.

## 8. Bootstrap prompt

The bootstrap template lives at [src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md](../src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md) and is the source of truth. It instructs the agent to audit one module for **evolve-optimizable** code locations (OpenEvolve / AlphaEvolve style), enumerating candidates that each name a self-contained behavior, a falsifiable rationale, a correctness oracle, and real headroom. The emitted `Candidate` shape matches the pydantic schema above (`id`, `file`, `line_start`, `line_end`, `symbol`, `kind`, `description`, `current_approach`, `evolve_rationale`, `estimated_impact`, `estimated_impact_explanation`); the prompt also documents the `plugin_seam` kind as a separate, language-neutral target category anchored at the registration site.

The supported placeholder set is:

- `{module_qualified_name}`, `{module_name}`, `{module_path}`, `{module_description}`
- `{depends_on}`, `{main_files}`, `{submodule_names}` — pre-formatted by the orchestrator per §5
- `{repo_context}` — `## Repository context` section body; `config.repo_context_markdown` verbatim or `_(none provided)_`

The `## Repository context` section sits between `## Module under audit` and `## What counts as a good evolve target` so module-scoped information is presented first; in particular, `module.path` ("all candidate files MUST lie under this") is established before the repo-level context, so a conflicting repo-context entry cannot expand the candidate surface.

## 9. Review prompt

The review template lives at [src/spotlights_engine/candidate_discovery/prompts_data/review.md](../src/spotlights_engine/candidate_discovery/prompts_data/review.md) and is the source of truth. It is adversarial: prune false positives, merge duplicates, sharpen rationales and impact explanations, and add any further candidates that clear the same evolve-target quality bar (no fixed cap). For each kept candidate the agent MAY tighten `description`, `current_approach`, `evolve_rationale`, `estimated_impact`, `estimated_impact_explanation`, `symbol`, `kind`, and adjust `line_start` / `line_end`; it MUST NOT change `id` or `file`.

The supported placeholder set is:

- `{prev_candidates_json}` — the prior iteration's `candidates.json` inlined by the orchestrator (replaces the `@candidates_prev.json` Claude-file-attach convention, which is not available in non-interactive subprocess mode)
- `{module_qualified_name}`, `{module_path}`
- `{max_seen_candidate_id}` — the highest id accepted in any earlier iteration; new ids MUST be strictly greater (lex order on the zero-padded `cand-NNNN` form)
- `{repo_context}` — `## Repository context` section body; same substitution rules as the bootstrap

The `## Repository context` section sits between `## Inputs` and `## Rules`, again preserving module-scoping precedence.
