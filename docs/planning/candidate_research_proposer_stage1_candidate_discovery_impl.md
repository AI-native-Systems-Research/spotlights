# candidate_research_proposer — Stage 1: Candidate Discovery — Implementation Plan

Implementation plan for the spec in [candidate_research_proposer_stage1_candidate_discovery.md](candidate_research_proposer_stage1_candidate_discovery.md). The spec is the *contract*; this document is the *build order*. Where the two disagree, the spec wins: flag the drift and update this implementation plan. If the spec itself is wrong, update both documents in the same PR.

The goal of this plan is to take Stage 1 from the current empty stub at [src/spotlights_engine/candidates/](../src/spotlights_engine/candidates/) (only `__init__.py`, which gets deleted) to a working `discover()` entrypoint at [src/spotlights_engine/candidate_discovery/](../src/spotlights_engine/candidate_discovery/) that satisfies every clause of the spec, with a test suite that exercises both the orchestration logic and the §6 validation pipeline without depending on live Claude / Codex CLIs.

---

## 1. Package layout

All new code lives under [src/spotlights_engine/candidate_discovery/](../src/spotlights_engine/candidate_discovery/). One package, flat namespace, small public API (`discover`, `DiscoveryConfig`, `DiscoveryResult`) re-exported from `candidate_discovery/__init__.py`. The `Candidate` / `Candidates` schemas live in [src/spotlights_engine/schemas/candidate.py](../src/spotlights_engine/schemas/candidate.py) (the existing stub gets replaced) because the architecture doc designates `schemas/` as the only contract surface other repos may import.

Note that this stage introduces a *second* public surface beyond `schemas/` — `spotlights_engine.candidate_discovery.discover` (and its `DiscoveryConfig` / `DiscoveryResult` types). That is intentional per spec §1; document this exception in the `candidate_discovery/__init__.py` module docstring so a future reviewer expecting the `schemas`-only cross-repo contract doesn't mistake it for a regression.

```
src/spotlights_engine/
├── schemas/
│   └── candidate.py                         # Candidate, Candidates  (replaces the dataclass stub)
└── candidate_discovery/
    ├── __init__.py                          # re-exports: discover, DiscoveryConfig, DiscoveryResult
    ├── api.py                               # DiscoveryConfig, IterationTelemetry, DiscoveryResult, discover()
    ├── errors.py                            # DiscoveryValidationError, DiscoveryMutationError, DiscoverySetupError
    ├── orchestrator.py                      # the bootstrap+review loop; calls into agent runners + validator
    ├── agents.py                            # AgentRunner ABC, ClaudeRunner, CodexRunner, _clean_env(), kill-on-timeout
    ├── prompts.py                           # wrap(), render_bootstrap(), render_review() + module-field formatters
    ├── validation.py                        # the §6 pipeline; returns (post_drop_list, telemetry_counters)
    ├── repo_guard.py                        # git/manifest-based target-repo mutation detector
    ├── telemetry.py                         # IterationTelemetry construction + diff helpers; iterations.jsonl writer
    ├── layout.py                            # run-dir path helpers: iter_dir(i, agent), artifact paths
    └── prompts_data/
        ├── bootstrap.md                     # verbatim from spec §8
        ├── review.md                        # verbatim from spec §9
        ├── preamble.md                      # verbatim from spec §5
        └── strict_retry.md                  # §6.2 retry reminder
```

The pre-existing empty `src/spotlights_engine/candidates/` stub is removed in M1.

`prompts_data/*.md` are shipped as package data (referenced via `importlib.resources`) so behavior does not depend on the cwd. Update [pyproject.toml](../pyproject.toml) `[tool.hatch.build.targets.wheel]` to include the data dir if hatchling does not pick it up automatically (verify in M3).

Tests mirror the source tree:

```
tests/unit/candidate_discovery/
├── test_schema_candidate.py
├── test_prompts.py
├── test_validation.py
├── test_telemetry.py
├── test_repo_guard.py
├── test_agents.py                            # uses a fake AgentRunner; no real CLIs
└── test_orchestrator.py                      # end-to-end with fake runners + tmp_path repo

tests/integration/candidate_discovery/
└── test_discover_smoke.py                    # opt-in; requires real `claude` and `codex` on PATH (marker: needs_clis)
```

---

## 2. Schemas — `spotlights_engine.schemas.candidate`

Replace the dataclass stub at [src/spotlights_engine/schemas/candidate.py](../src/spotlights_engine/schemas/candidate.py) with the pydantic models in spec §1. Carry the field constraints from the spec verbatim, including `extra="forbid"` on every object so the exported `model_json_schema()` is acceptable to OpenAI strict structured outputs (Codex `--output-schema`):

```python
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator

CandidateKind = Literal[
    "function", "method", "loop", "region", "kernel", "config_block", "plugin_seam",
]
MetricDirection = Literal["minimize", "maximize"]
EstimatedImpact = Literal["high", "medium", "low"]


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    direction: MetricDirection
    target_or_baseline: str | None = Field(...)


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cand-\d{4}$")
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    symbol: str = Field(min_length=1, max_length=200)
    kind: CandidateKind
    description: str = Field(min_length=1)
    current_approach: str = Field(min_length=1)
    evolve_rationale: str = Field(min_length=1)
    metrics: list[Metric] = Field(min_length=1)
    estimated_impact: EstimatedImpact

    @model_validator(mode="after")
    def _check_range(self) -> "Candidate":
        if self.line_end < self.line_start:
            raise ValueError("line_end must be >= line_start")
        return self

class Candidates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    module_qualified_name: str
    candidates: list[Candidate] = Field(min_length=1)
```

Re-export both names from [src/spotlights_engine/schemas/__init__.py](../src/spotlights_engine/schemas/__init__.py) (`__all__ = ["Candidate", "Candidates", ...]`).

Unit tests in `test_schema_candidate.py`:
- `cand-0001` accepted; `cand-1`, `cand-00001`, `candidate-0001` rejected.
- `line_end < line_start` rejected with the validator's message.
- Each required string field rejects the empty string; `symbol` rejects a >200-char value; `Metric.name` rejects a >80-char value.
- `metrics=[]` rejected (the `min_length=1` invariant on `Candidate.metrics`).
- `candidates=[]` rejected (the `min_length=1` invariant on `Candidates.candidates`).
- An unknown `kind` / `direction` / `estimated_impact` value is rejected by the `Literal` constraint.
- An extra top-level field is rejected (`extra="forbid"`).
- `Candidates.model_json_schema()` round-trips through `json.dumps` (used by `--json-schema` and `--output-schema`).

---

## 3. Errors — `spotlights_engine.candidate_discovery.errors`

Three exception classes, all subclasses of `Exception`. The spec calls these out by name in §1 and §6:

```python
class DiscoverySetupError(Exception):       # raised before any iteration starts
    pass

class DiscoveryValidationError(Exception):  # §6.2 second-failure, §6.3, §6.7, §6.8-empty
    pass

class DiscoveryMutationError(Exception):    # §6.1 target-repo mutation
    pass
```

Each carries a structured `context: dict` (iteration `n`, agent name, paths involved) so test asserts and Stage-3 callers can branch on cause without parsing the message string.

`DiscoverySetupError` covers setup failures before an iteration starts: invalid `repo_path`, an `artifacts_dir` that resolves inside `repo_path`, a pre-existing `<artifacts_dir>/candidate_discovery/` directory (§1 "never resumes"), and missing required CLIs (caught in `AgentRunner.__init__` via `shutil.which`).

---

## 4. Public API — `candidate_discovery.api`

```python
class DiscoveryConfig(BaseModel):
    repo_path: Path
    module_qualified_name: str
    module: Module
    artifacts_dir: Path
    repo_context_markdown: str | None = Field(default=None, min_length=1, max_length=20_000)
    num_review_iterations: int = Field(default=3, ge=0)
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = Field(default="gpt-5.5", pattern=r"^[\w.\-/]+$")
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"

class IterationTelemetry(BaseModel): ...     # exact fields from spec §1
class DiscoveryResult(BaseModel):
    candidates: Candidates
    iterations: list[IterationTelemetry]
    total_duration_s: float
    total_cost_usd: float | None = None

def discover(config: DiscoveryConfig) -> DiscoveryResult:
    """Thin wrapper around Orchestrator.run() — sole public entrypoint."""
```

`discover` does three things and delegates everything else:
1. `DiscoverySetupError` if `repo_path` doesn't exist, isn't a directory, `artifacts_dir` resolves inside `repo_path`, or `<artifacts_dir>/candidate_discovery/` already exists.
2. Construct the `Orchestrator`.
3. Return `Orchestrator.run()`.

The Module → qualified-name resolution happens **upstream** in the driver per spec §1; `discover` trusts that `config.module` matches `config.module_qualified_name`. The driver-level resolution is out of scope for this plan.

---

## 5. Orchestrator — `candidate_discovery.orchestrator`

The orchestrator is a class because it carries non-trivial state across iterations:

- `self._prev_raw: Candidates | None` — the previous iteration's *raw parsed* candidates (before drops). Used for the §6.7 carry-over `file` check ("any `id` carried over from the previous iteration MUST retain the previous iteration's `file`"), which the spec specifies on the raw list.
- `self._prev_post_drop: Candidates | None` — the previous iteration's normalized/post-drop candidates. Used for the next review prompt and for telemetry/diff calculation.
- `self._prev_post_drop_ids: set[str]` — the previous iteration's *post-drop* id set. Used for the §6.7 newly-minted detection ("on the post-drop list, an `id` absent from the previous iteration is treated as newly minted").
- `self._max_seen_id: str` — the lexicographically highest id ever accepted (post-drop) across all earlier iterations. Used for the §6.7 monotonicity check ("every newly minted `id` MUST be strictly greater than the highest id accepted in any earlier iteration"). Initialized to `"cand-0000"` so n=0's `cand-0001` passes.
- An open NDJSON file handle for `iterations.jsonl`, opened in append-line-buffered mode and flushed after each `write()` so a mid-loop crash leaves a partial-but-parseable file.

```python
class Orchestrator:
    def __init__(self, config: DiscoveryConfig): ...
    def run(self) -> DiscoveryResult: ...

    # internal
    def _bootstrap(self) -> _IterOutcome: ...                       # i=0, claude
    def _review(self, n: int, agent: AgentRunner, prev: Candidates) -> _IterOutcome: ...
    def _run_iteration(self, n: int, agent: AgentRunner, prompt: str) -> _IterOutcome: ...
    def _copy_final(self) -> None: ...                              # §6.8: copy iter N candidates.json → <artifacts_dir>/candidates.json
    def _finalize(self) -> DiscoveryResult: ...                     # build DiscoveryResult from accumulated telemetry
```

`_IterOutcome` is an internal dataclass: `(raw: Candidates, candidates: Candidates, telemetry: IterationTelemetry, agent_invocation: AgentInvocation)`. The raw field is needed only to advance the §6.7 cross-iteration state after the telemetry row has been built against the previous post-drop list.

`run()` does exactly what spec §2/§7 prescribe:

```python
def run(self) -> DiscoveryResult:
    claude = ClaudeRunner(self.config)
    codex = CodexRunner(self.config)
    agents = [codex, claude]                       # review rotation: agents[(N-1) % 2]
    self._mint_run_dir()                           # mkdir candidate_discovery/, dump schema

    boot = self._run_iteration(n=0, agent=claude, prompt=self._bootstrap_prompt())
    prev = boot.candidates
    self._record(boot)

    for n in range(1, self.config.num_review_iterations + 1):
        agent = agents[(n - 1) % 2]
        out = self._run_iteration(n, agent, prompt=self._review_prompt(prev))
        prev = out.candidates
        self._record(out)

    self._copy_final()
    return self._finalize()
```

`_run_iteration` is the inner loop; it composes agent invocation, validation, persistence, and the §6.2 single retry. See §6 below for its shape.

Construct the runners before `_mint_run_dir()` so a missing `claude` or `codex` executable raises `DiscoverySetupError` without leaving behind a fresh `<artifacts_dir>/candidate_discovery/` directory that would make the next invocation fail as a false "resume" attempt.

---

## 6. Inner iteration — bind agent ↔ validation ↔ persistence

`_run_iteration(n, agent, prompt)` carries the §6.1–§6.8 pipeline. Pseudocode:

```python
def _run_iteration(self, n, agent, prompt) -> _IterOutcome:
    iter_dir = self.layout.iter_dir(n, agent.name)
    iter_dir.mkdir(parents=True)

    schema_retries = 0
    last_exc: Exception | None = None
    for attempt in (0, 1):
        attempt_prompt = prompt
        if attempt == 1:
            attempt_prompt = prompt + "\n\n" + self._strict_mode_reminder()
        (iter_dir / "prompt.md").write_text(attempt_prompt)

        try:
            with self.repo_guard.observe():                          # §6.1 wraps the subprocess (RepoGuard holds repo_path)
                inv = agent.invoke(prompt=attempt_prompt, iter_dir=iter_dir, schema_path=self.schema_path)
            # repo_guard raises DiscoveryMutationError on diff; that escapes the for-loop.
            payload_json = agent.parse_last_message(iter_dir)       # raises _SchemaParseError
            try:
                parsed = Candidates.model_validate_json(payload_json)
            except ValidationError as e:
                raise _SchemaParseError("schema validation failed") from e
            self._check_qualified_name(parsed)                      # §6.3 — raises
            survivors, drops = self.validator.run(parsed)           # §6.4-6.6
            self._check_id_integrity(parsed, survivors)             # §6.7 — raises
            normalized = self._normalize_and_persist(n, agent.name, survivors, iter_dir)  # §6.8 — re-validates min_length=1, writes candidates.json
            telemetry = self.telemetry_builder.build(
                n=n, agent=agent.name, inv=inv, survivors=normalized,
                drops=drops, prev=self._prev_post_drop, schema_retries=schema_retries)
            return _IterOutcome(raw=parsed, candidates=normalized, telemetry=telemetry, agent_invocation=inv)
        except _SchemaParseError as e:
            last_exc = e
            if attempt == 1:
                break
            schema_retries += 1                                     # one retry will be attempted
            continue
    raise DiscoveryValidationError("schema parse failed twice", cause=last_exc, ...)
```

Notes:
- `_SchemaParseError` is **package-internal** (defined alongside the runners in `candidate_discovery.agents`, leading-underscore so it never reaches the public API). It is raised by `AgentRunner.invoke` (timeout / nonzero exit) *and* by `AgentRunner.parse_last_message` (missing / empty / non-JSON `last_message.json`); the orchestrator is the only consumer and translates each occurrence into either a retry or a `DiscoveryValidationError`. The spec's "iteration counted as a schema-parse failure on timeout, then retried once per §6.2" maps to: `AgentRunner.invoke` raises `_SchemaParseError("timeout")` on `subprocess.TimeoutExpired`, and the same retry branch handles it.
- The mutation guard intentionally wraps `agent.invoke`. If `invoke` raises `_SchemaParseError` *and* the guard's `__exit__` detects a mutation, Python suppresses the `_SchemaParseError` in favor of `DiscoveryMutationError` — which is the right priority (mutation is fatal, not retryable).
- **`_check_id_integrity` (§6.7) decomposes into four ordered sub-checks**, all in the orchestrator (they raise rather than drop):
  1. *Within-iter uniqueness* — `len({c.id for c in parsed.candidates}) == len(parsed.candidates)`. Operates on the raw parsed list.
  2. *Carry-over file unchanged* — for every `id` in `parsed.candidates ∩ self._prev_raw`, the new candidate's `file` equals the prev raw candidate's `file`. Operates on the raw parsed list.
  3. *Newly-minted detection* — `new_ids = {c.id for c in survivors} - self._prev_post_drop_ids`. Operates on the post-drop list.
  4. *Monotonicity* — every id in `new_ids` is `> self._max_seen_id` under lexicographic comparison on the zero-padded `cand-NNNN` form. For n=0, `self._prev_raw is None`, `self._prev_post_drop_ids` is empty, and `self._max_seen_id == "cand-0000"`, so checks 2 and 3 are no-ops and check 4 reduces to "all ids > cand-0000". This intentionally rejects a bootstrap `cand-0000`, which the schema pattern alone would otherwise accept.
- `_normalize_and_persist` re-validates the survivors via `Candidates(module_qualified_name=..., candidates=survivors)` to enforce §6.8's `min_length=1` invariant on the *current* iteration, then writes `candidates.json` via `model_dump_json(indent=2)`. A pydantic `ValidationError` from the re-validation (drops emptied the list) is caught and re-raised as `DiscoveryValidationError(context={"iteration": n, "agent": agent.name, "reason": "post-drop list empty"})` — it escapes the retry loop because the for-loop only catches `_SchemaParseError`.
- `_record(out)` appends and flushes telemetry, then advances cross-iteration state in this order: `self._prev_raw = out.raw`, `self._prev_post_drop = out.candidates`, `self._prev_post_drop_ids = {c.id for c in out.candidates.candidates}`, `self._max_seen_id = max(self._max_seen_id, max(c.id for c in out.candidates.candidates))`.
- Persisted artifacts per iteration: `prompt.md`, `last_message.json`, `raw_stdout.log`, `raw_stderr.log`, `candidates.json`, and for `n >= 1` also `diff_from_prev.md`. A schema-retry iteration produces *one* `prompt.md` (the second-attempt prompt, with the strict-mode reminder concatenated) and `last_message.json` from the second attempt only; the first attempt's stdout/stderr go to `raw_stdout.log` / `raw_stderr.log` followed by a `--- retry separator ---` line and then the second attempt's streams, so both are inspectable without colliding on path.
- **`duration_s` is iteration-wallclock, including any retry.** The orchestrator measures `time.monotonic()` at iteration start and end and uses that for `IterationTelemetry.duration_s`. `AgentInvocation.duration_s` returned by the runner is per-attempt and informational; it does *not* appear in the telemetry directly. This matters because a schema-retry iteration's true duration is roughly 2× a normal one, and downstream cost/perf analysis needs that visibility.
- A raised iteration (any `DiscoveryValidationError` / `DiscoveryMutationError`) leaves no row in `iterations.jsonl`. The exception itself carries the iteration index in `context`.

---

## 7. Agent runners — `candidate_discovery.agents`

A small abstract base plus two concrete runners. Both invoke their CLI on stdin with a wall-clock timeout, capture stdout/stderr, normalize `<iter_dir>/last_message.json` to the raw final assistant message text, and return a uniform `AgentInvocation`. **Each runner owns its own `parse_last_message`** so CLI-specific missing/empty/output-shape checks stay out of the orchestrator.

```python
@dataclass
class AgentInvocation:
    session_id: str | None
    duration_s: float                          # per-attempt; informational only, NOT used for IterationTelemetry
    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None

class AgentRunner(ABC):
    name: str                                  # "claude_code" | "codex"
    def __init__(self, config: DiscoveryConfig): ...
    @abstractmethod
    def invoke(self, prompt: str, iter_dir: Path, schema_path: Path) -> AgentInvocation: ...
    @abstractmethod
    def parse_last_message(self, iter_dir: Path) -> str:
        """Extract the candidates-JSON text from <iter_dir>/last_message.json.

        Returns the raw JSON string for `Candidates.model_validate_json()`. Raises
        `_SchemaParseError` on a missing, empty, or non-JSON candidates payload.
        """
```

Common behavior in a `_run_subprocess` helper:
- Construct argv per spec §4.
- `env = _clean_env(...)` — see §10 below.
- `subprocess.run(argv, input=prompt.encode("utf-8"), capture_output=True, env=env, cwd=cwd, timeout=config.per_iteration_wallclock_s)`.
- Tee stdout/stderr to `<iter_dir>/raw_stdout.log` and `<iter_dir>/raw_stderr.log`; if those files already exist because this is a retry, append a `--- retry separator ---` line before the new streams.
- On `TimeoutExpired`: write any partial stdout/stderr carried by the exception, then raise `_SchemaParseError("timeout")` so the §6.2 retry path engages.
- On any nonzero exit: raise `_SchemaParseError` with the return code plus stderr/stdout tails; the retry path engages even if stderr is empty.

### ClaudeRunner (spec §4)

```python
argv = [
    "claude", "-p",
    "--output-format", "stream-json", "--verbose",
    "--json-schema", schema_path.read_text(),
    "--permission-mode", "plan",
    "--max-turns", str(config.claude_max_turns),
]
cwd = config.repo_path
```

After the subprocess returns:
1. Parse stdout as NDJSON (one JSON value per line, last value `{"type":"result", ...}` per spec §4).
2. Write the `result` payload string **verbatim** to `<iter_dir>/last_message.json`. This is Claude's final assistant message text and must itself be the raw `Candidates` JSON object.
3. Pull `session_id`, `duration_ms`, `total_cost_usd`, `usage.input_tokens`, `usage.output_tokens` if present and return them in `AgentInvocation`.

**`ClaudeRunner.parse_last_message`** reads `<iter_dir>/last_message.json` and returns its contents unchanged. Missing file, empty file, a missing terminal stdout `result` event, or a `result` payload that does not parse as JSON → `_SchemaParseError`.

### CodexRunner (spec §4)

```python
argv = [
    "codex", "exec", "-",
    "--json",
    "--output-last-message", str(iter_dir / "last_message.json"),
    "--output-schema", str(schema_path),
    "--sandbox", "read-only",
    "-C", str(config.repo_path),
    "-c", f'model="{config.codex_model}"',
    "-c", f'model_reasoning_effort="{config.codex_reasoning_effort}"',
]
cwd = config.repo_path                              # `-C` is authoritative for path resolution, but a
                                                    # belt-and-braces cwd prevents any relative path read
                                                    # from escaping the target subtree.
```

Codex writes `last_message.json` itself using the same raw-final-message contract. After return, parse Codex's NDJSON event stream (stdout) to extract `session_id`, duration, and any cost / token counters Codex surfaces; fields not surfaced stay `None`. Spec §4 also notes Codex validates the message against the schema CLI-side, so a malformed shape arrives as a nonzero exit with a schema-violation message — this lands in the same `_SchemaParseError` retry path.

**`CodexRunner.parse_last_message`** reads `<iter_dir>/last_message.json` and returns its contents *unchanged* (the file already holds the raw candidates JSON). Missing file, empty file, zero-byte file (which `--output-schema` can produce on a CLI-side validation failure before the candidates message is written), or content that does not parse as JSON → `_SchemaParseError`.

### CLI presence check

Both runners check `shutil.which(self._executable)` in `__init__` and raise `DiscoverySetupError` if missing. This is the only setup check owned by the runners; `discover()` still owns `repo_path` and run-dir setup checks.

---

## 8. Prompts — `candidate_discovery.prompts`

Three prompt templates (`bootstrap.md`, `review.md`, `preamble.md`) plus the strict retry reminder (`strict_retry.md`) are loaded once at module import via `importlib.resources.files("spotlights_engine.candidate_discovery.prompts_data")`.

**Substitution must not use `str.format` / `format_map`.** Both prompts contain literal JSON object braces (`{` followed by newline+whitespace, see spec §8 and §9), and the standard formatter parser raises on any unbalanced `{`. Use a regex-driven replacer instead, matching only `\{(\w+)\}` (placeholder = `{` + word-chars + `}`, no intervening whitespace), which by construction never matches the JSON braces in the templates:

```python
_PLACEHOLDER = re.compile(r"\{(\w+)\}")

def _substitute(template: str, values: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            raise KeyError(f"unknown placeholder {{{key}}} in template")
        return values[key]
    return _PLACEHOLDER.sub(repl, template)
```

Unknown placeholders are an error, not a silent passthrough — they would silently leak the literal `{name}` into the prompt and confuse the agent. The set of allowed keys per template is fixed (see public functions below); a unit test feeds each template through `_substitute` with the matching key set and asserts no `\{(\w+)\}` survives.

The strict-mode reminder for §6.2 retries also lives here, as `prompts_data/strict_retry.md`:

```
Your previous reply did not parse as JSON matching the candidates schema. Reply with exactly one JSON object — no markdown fence, no commentary, no leading or trailing prose. The orchestrator will reject anything else.
```

Loaded once at import time and exposed as `STRICT_RETRY_REMINDER: str`; the orchestrator concatenates it onto the prompt with a single blank line separator.

Public functions:

```python
def wrap(prompt: str) -> str:
    """Prepend the §5 preamble."""

def render_bootstrap(
    module_qualified_name: str,
    module: Module,
    *,
    repo_context_markdown: str | None = None,
) -> str: ...

def render_review(
    module_qualified_name: str,
    module: Module,
    prev_candidates_json: str,
    max_seen_candidate_id: str,
    *,
    repo_context_markdown: str | None = None,
) -> str: ...
```

The new `repo_context_markdown` parameter on both renderers is keyword-only — both for forward compatibility against further positional additions and so call sites are self-documenting. The `None` default preserves backward compatibility for tests that exercise the renderers without the new field.

Three private formatter helpers preformat the Module fields per spec §5, plus one new helper for the repo-context section:

```python
def _format_main_files(files: list[File]) -> str:
    # one line per File: "- <path> (<role>)"
def _format_submodule_names(submodules: list[Module]) -> str:
    return ", ".join(m.name for m in submodules) or "(none)"
def _format_depends_on(deps: list[str]) -> str:
    return ", ".join(deps) or "(none)"

_REPO_CONTEXT_DEFAULT = "_(none provided)_"

def _format_repo_context(md: str | None) -> str:
    return _REPO_CONTEXT_DEFAULT if md is None else md  # verbatim; no trim, no reflow
```

Both `bootstrap.md` and `review.md` carry a single dedicated block — `## Repository context\n\n{repo_context}` — placed after the module/inputs block and before the rules / quality-bar block (see spec §8 / §9 for the exact insertion points). Because `_substitute` raises on unknown placeholders, both renderers MUST pass `{repo_context}` into the values dict on every call; the `None` default routes through `_format_repo_context`.

Unit tests in `test_prompts.py`:
- Bootstrap with no submodules / no deps renders `(none)`.
- After substitution, no unsubstituted `{word}` placeholder remains in either prompt (regex assertion on the output). Run this with a brace-free `repo_context_markdown` (or the default `None`) so the regression-locked scan can run safely against the whole rendered output.
- The literal JSON-object braces in the templates appear unchanged in the output, byte-for-byte.
- An unknown placeholder in a template (test-only fixture) raises `KeyError` rather than passing through.
- The review prompt includes the full `prev_candidates_json` text and the literal `max_seen_candidate_id`.
- The wrapped prompt always begins with the preamble; the second half is byte-identical to the unwrapped prompt.
- `STRICT_RETRY_REMINDER` loads at import and is non-empty.
- **Repo context: default branch.** `repo_context_markdown=None` substitutes `_(none provided)_`; the `## Repository context` header is present in both rendered templates.
- **Repo context: round-trip.** A small payload (e.g. ``"## Tests\n\n`pytest -q`\n"``) appears byte-for-byte in the rendered output (no escaping, no reflow).
- **Repo context: literal braces preserved.** A payload containing a `{json}`-shaped substring round-trips intact. Conceptually: `re.sub` does not rescan replacement text, so the placeholder scan runs once against the *template*, never against substituted values — user-supplied markdown cannot trigger spurious substitutions or `KeyError`s. Test this separately from the "no placeholders remain" assertion.
- **Repo context: schema bounds.** `DiscoveryConfig(repo_context_markdown="")` and `DiscoveryConfig(repo_context_markdown="x" * 20_001)` both raise `ValidationError`; `None` is accepted and routes through `_(none provided)_`.

---

## 9. Validation — `candidate_discovery.validation`

One class, one entrypoint. The orchestrator calls it after schema parse and qualified-name check (which it owns directly since those *raise*).

```python
@dataclass
class DropCounters:
    dropped_outside_module: int = 0
    dropped_missing_file: int = 0
    dropped_invalid_ranges: int = 0

class Validator:
    def __init__(self, config: DiscoveryConfig): ...
    def run(self, parsed: Candidates) -> tuple[list[Candidate], DropCounters]:
        """Apply §6.4, §6.5, §6.6 in that order; return survivors and counters."""
```

Each rule is a private method returning `(kept, dropped_count)`. The order matters: containment first (cheap, no file read), then existence (one `os.stat`), then line-range sanity (one `len(open(...).readlines())` per surviving file). Cache `file_line_count` by `(file_path, mtime_ns)` within a single validator call so two candidates in the same file don't re-read it.

`_check_id_integrity` lives on the orchestrator (not the validator) because it needs the cross-iteration state (`self._prev_raw`, `self._prev_post_drop_ids`, `self._max_seen_id`) and it *raises* rather than drops.

Unit tests in `test_validation.py` (using `tmp_path` to build a minimal `repo_path`):
- Drops outside `module.path` and bumps the counter.
- Drops absolute paths (`/etc/passwd`) — they fail the relative-path check before the resolve.
- Drops symlink escape (`module/foo.py` → `../../outside.py`); the resolve happens *strict=False* per spec but containment is checked on the resolved path.
- Drops a file that doesn't exist.
- Drops `line_end > file_line_count`.
- Empty post-drop list triggers the §6.8 re-validate → `DiscoveryValidationError`.
- Two-candidate same-file case reads the file once (assert via `unittest.mock.patch` on `Path.read_text`).

---

## 10. Env scrubbing — `_clean_env`

Per spec §4 "Env scrubbing". Local to `candidate_discovery.agents`. Pattern follows [CORAL](../../CORAL/coral/workspace/repo.py) but with the Stage-1-specific drops:

```python
_DROP_EXACT = {"OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
               "VIRTUAL_ENV"}
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")

def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env
```

Unit tests in `test_agents.py` patch `os.environ` with all four kinds of leak (each `_DROP_EXACT` entry, each `_DROP_PREFIX` family) and assert they're stripped while `PATH`, `HOME`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY` survive.

---

## 11. Repo guard — `candidate_discovery.repo_guard`

Detects target-repo mutation per §6.1. Two implementations behind a context-manager interface:

```python
class RepoGuard:
    def __init__(self, repo_path: Path): ...
    @contextmanager
    def observe(self) -> Iterator[None]:
        before = self._snapshot()
        try:
            yield
        finally:
            after = self._snapshot()
            if before != after:
                raise DiscoveryMutationError(diff=self._diff(before, after))
```

`_snapshot()` picks one of two strategies based on `(repo_path / ".git").is_dir()`:

1. **Git path**: capture `git -C <repo_path> -c core.quotepath=off status --porcelain=v1 -z`
   *and* the lightweight manifest described below, excluding `.git`. The status bytes give useful
   git/index context; the manifest closes the dirty-repo blind spot where an already-modified
   tracked file or an already-untracked file can change while `git status` remains byte-identical.
   `core.quotepath=off` avoids locale-driven encoding surprises.
2. **Non-git path**: walk `repo_path` (skipping `.git`, `__pycache__`, `.venv`), build a dict `{relpath: (st_mtime_ns, st_size, xattr_digest)}` and hash it. `xattr_digest` is computed from sorted extended-attribute names and values via `os.listxattr` / `os.getxattr` where the platform supports them; if xattrs are unavailable for a path, record a stable sentinel rather than failing the run.

Spec coverage limit: "the orchestrator does not detect writes outside this subtree" — make this explicit in the docstring; `--permission-mode plan` and `--sandbox read-only` are the primary defenses there.

Unit tests in `test_repo_guard.py`:
- Git mode: clean repo → no raise; create a file → raise.
- Git mode: stage-but-not-commit also raises.
- Git mode: changing an already-dirty tracked file or an already-untracked file also raises.
- Non-git mode: touch an existing file → raise; rename a file → raise; create a new file → raise; changing an xattr raises on platforms that expose xattr APIs.
- The `.git` metadata path is excluded from the non-git walk (cover with a repo-like fixture where `.git` is a plain file or the non-git strategy is forced; a real `.git/` directory takes the git-status branch).

---

## 12. Telemetry — `candidate_discovery.telemetry`

`IterationTelemetry` lives in `api.py` (it's part of the public schema), but its *construction* — including the added/removed/modified diff — lives here.

```python
class TelemetryBuilder:
    def build(self, n, agent, inv: AgentInvocation, survivors: Candidates,
              drops: DropCounters, prev: Candidates | None, schema_retries: int) -> IterationTelemetry: ...

def render_diff_markdown(prev: Candidates, current: Candidates) -> str:
    """Used by orchestrator to write iter_dir/diff_from_prev.md for n >= 1."""
```

Diff rules (spec §6 final paragraph) are exact:
- `added = ids_n \ ids_{n-1}`
- `removed = ids_{n-1} \ ids_n`
- `modified = { id ∈ ids_n ∩ ids_{n-1} : any of (line_start, line_end, kind, estimated_impact, symbol.strip(), description.strip(), current_approach.strip(), evolve_rationale.strip(), normalized metrics tuple) differs }` — the per-field strip rule preserves the historical "whitespace-only edits are not material" intent across all text fields, and the metrics tuple is `sorted((name.strip(), direction, target_or_baseline is None, target_or_baseline.strip() if str else ""))` so reorderings and equivalent-value rewordings do not register as drift.

Serialize `added`, `removed`, and `modified` in lexicographic `cand-NNNN` order so telemetry JSON and `diff_from_prev.md` are deterministic. For `n=0`, `added` is all survivor ids in that same order, `removed = []`, and `modified = []`.

The orchestrator appends one NDJSON line per iteration to `<artifacts_dir>/candidate_discovery/iterations.jsonl` — `IterationTelemetry.model_dump_json()` plus a newline, flushed immediately. The same in-memory list goes into `DiscoveryResult.iterations`, so the on-disk file and the in-memory result are equivalent (spec §1 invariant).

`DiscoveryResult.total_duration_s` is `time.monotonic()` end-to-end over `Orchestrator.run()` (includes orchestrator overhead, not just the sum of agent calls). `DiscoveryResult.total_cost_usd` is `sum(t.cost_usd for t in iterations if t.cost_usd is not None)` if at least one iteration reported a cost, else `None` — i.e., partial visibility is preserved, but a fully-unreported run surfaces as `None` rather than `0.0` so downstream cost dashboards don't confuse "free" with "unknown".

`render_diff_markdown(prev, current)` emits three fixed sections; an empty section is rendered as `_(none)_` so the file shape is stable across iterations. Each line carries the candidate id, file, line range, `[kind]`, `symbol`, and `evolve_rationale` so a reviewer can scan one iteration's drift without opening the underlying JSON:

```markdown
## Added
- cand-0007 — src/foo/x.py:142-211 [region] hot_loop — Hoist allocator out of inner loop

## Removed
- cand-0003 — src/foo/y.py:88-94 [function] cpu_copy — (was: redundant tensor.cpu() on hot path)

## Modified
- cand-0001 — src/foo/scheduler.py:142-211 → 142-203 [function] schedule — evolve_rationale tightened
```

For `n=0`, every survivor is "Added" and the other two sections are `_(none)_`.

Unit tests in `test_telemetry.py`:
- Round-trip a few `IterationTelemetry` instances through `model_dump_json` ↔ `model_validate_json` and assert equality.
- Diff: rename → (removed, added); change of `line_end` → modified; whitespace-only change of `evolve_rationale` (or any other text field covered by `.strip()`) → not modified.

---

## 13. Run-dir layout — `candidate_discovery.layout`

A tiny module so path conventions are in one place:

```python
def candidate_discovery_root(artifacts_dir: Path) -> Path:
    return artifacts_dir / "candidate_discovery"

def schema_path(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "candidates.schema.json"

def repo_context_path(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "repo_context.md"

def iter_dir(artifacts_dir: Path, n: int, agent: str) -> Path:
    tag = "bootstrap" if n == 0 else agent       # iter_0_bootstrap, iter_1_codex, ...
    return candidate_discovery_root(artifacts_dir) / f"iter_{n}_{tag}"

def iterations_jsonl(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "iterations.jsonl"

def final_artifact(artifacts_dir: Path) -> Path:
    return artifacts_dir / "candidates.json"
```

Spec §3 directory tree is the authoritative reference; this module is the one place that knows the layout. `repo_context_path` is the single source of truth for the persisted run-time copy of `config.repo_context_markdown` (see §5 orchestrator wiring); the file is written only when the caller supplied repo context.

---

## 14. End-to-end orchestrator test — fake agents

The integration of all components above is exercised by `tests/unit/candidate_discovery/test_orchestrator.py` using a `FakeAgentRunner` that bypasses subprocess entirely.

`FakeAgentRunner.invoke(prompt, iter_dir, schema_path)`:
- Reads a scripted sequence of canned `last_message.json` bodies (one per iteration).
- Writes the next scripted body to `<iter_dir>/last_message.json`, matching the raw-final-message contract used by both real runners.
- Returns a fixed `AgentInvocation(session_id="fake-0", duration_s=1.0, cost_usd=0.01, input_tokens=100, output_tokens=200)`.

Test scenarios (each gets its own scripted sequence):
1. **Happy path** — 4 iterations, all valid, candidates monotone-grow. Assert: final `candidates.json` byte-equals the last iter's, `iterations.jsonl` has 4 lines, `diff_from_prev.md` exists in iters 1/2/3 only, no discovery exception is raised.
2. **Schema-parse retry succeeds** — iter 1 first call emits malformed JSON, second call (after strict-mode reminder) emits valid JSON. Assert `schema_retries == 1` on iter 1's telemetry.
3. **Schema-parse retry fails twice** — `DiscoveryValidationError` raised; the loop is not entered for n=2.
4. **Mutation guard fires** — fake runner touches a file inside `repo_path` during `invoke`; `DiscoveryMutationError` raised.
5. **Qualified-name mismatch** — `DiscoveryValidationError` raised on iter 0.
6. **Containment drop** — iter 0 emits one candidate whose `file` resolves outside `module.path`; survives if at least one valid candidate remains; iter telemetry shows `dropped_outside_module=1`.
7. **All candidates dropped** — `DiscoveryValidationError` ("min_length=1") raised on iter 0.
8. **ID monotonicity violation** — iter 2 mints `cand-0003` after iter 1 had `cand-0005`; `DiscoveryValidationError` raised.
9. **Pre-existing run dir** — `<artifacts_dir>/candidate_discovery/` exists at call time; `DiscoverySetupError` raised before any subprocess.
10. **Artifacts inside target repo** — `artifacts_dir` resolves under `repo_path`; `DiscoverySetupError` raised before any subprocess or run-dir creation.
11. **Final artifact copy** — iter `num_review_iterations`'s `candidates.json` is byte-identical to `<artifacts_dir>/candidates.json`.
12. **Repo context persisted when supplied** — with `repo_context_markdown="## X\n"`, after `run()` returns, `layout.repo_context_path(artifacts_dir)` exists and its bytes equal the supplied string. The orchestrator writes it once in `_mint_run_dir`.
13. **No repo-context file when `None`** — with `repo_context_markdown=None`, `layout.repo_context_path(artifacts_dir)` does **not** exist after `run()` returns. Absence is itself informative ("this run had no repo context").
14. **Repo context reaches every iteration prompt** — `FakeAgentRunner` already lets the orchestrator write `(iter_dir / "prompt.md").write_text(attempt_prompt, ...)`. Assert that every iteration's `prompt.md` contains the supplied markdown verbatim (proves it threaded through bootstrap *and* each review).

The fake fixture takes ~1 ms per iteration, so the whole orchestrator suite stays under a second.

---

## 15. Integration smoke (opt-in)

`tests/integration/candidate_discovery/test_discover_smoke.py` is marked `@pytest.mark.needs_clis` and skipped by default. It runs `discover()` against a tiny fixture repo with one module of ~80 lines and `num_review_iterations=1, claude_max_turns=4, per_iteration_wallclock_s=180`. The point isn't to validate outputs (Claude/Codex behavior is non-deterministic) — it's to confirm:
- The argv lines in §7 actually launch each CLI.
- Both runners produce `last_message.json` using the raw-final-message contract the orchestrator parses.
- The launched argv includes Claude's `--permission-mode plan` and Codex's `--sandbox read-only`; target-repo mutation detection itself stays covered deterministically by the unit tests.

This test runs in CI only when `STAGE1_NEEDS_CLIS=1`. Locally a developer with both CLIs installed runs `pytest -m needs_clis`.

---

## 16. Implementation order (M1 → M5)

Each milestone leaves the repo in a green-tests state and is independently reviewable.

**M1 — schemas + errors.** Replace [src/spotlights_engine/schemas/candidate.py](../src/spotlights_engine/schemas/candidate.py) with the pydantic models. Remove the empty `src/spotlights_engine/candidates/` stub. Add `candidate_discovery/errors.py`. Tests: `test_schema_candidate.py`. *Done when:* `uv run pytest tests/unit/candidate_discovery/test_schema_candidate.py` passes and `Candidates.model_json_schema()` is exportable.

**M2 — prompts + layout.** Add `prompts.py`, `layout.py`, and the four `prompts_data/*.md` files (`preamble`, `bootstrap`, `review`, `strict_retry`; the first three verbatim from spec §5/§8/§9). Tests: `test_prompts.py`. *Done when:* both `render_bootstrap` and `render_review` produce stable byte-for-byte output for a fixed `Module` fixture (regression-locked via golden files).

**M3 — validation + repo guard.** Add `validation.py` and `repo_guard.py`. Tests: `test_validation.py`, `test_repo_guard.py`. *Done when:* every §6.4–§6.6 drop case and every §6.1 mutation case has a passing test. Confirm hatchling ships `prompts_data/` in the wheel (`uv build && unzip -l dist/*.whl | grep prompts_data`).

**M4 — agent runners + telemetry.** Add `agents.py` (with `_clean_env`, both runners) and `telemetry.py`. Tests: `test_agents.py` (env scrubbing, argv construction via a `_build_argv` helper, `_SchemaParseError` translation from `subprocess.TimeoutExpired` / nonzero exit), `test_telemetry.py`. No live CLI calls — `subprocess.run` is patched.

**M5 — orchestrator + public API.** Add `orchestrator.py`, `api.py`, wire `discover` through `candidate_discovery/__init__.py`. Tests: `test_orchestrator.py` with the 11 scenarios in §14, and the integration smoke at §15 (gated). *Done when:* `uv run pytest tests/unit/candidate_discovery/` is green and a manual `python -c "from spotlights_engine.candidate_discovery import discover; ..."` against a real fixture repo with the smoke marker produces a valid `DiscoveryResult`.

---

## 17. Risks and notes

- **Codex stdout schema.** Spec §4 references a Codex CLI surface that may have drifted by the time this lands. M4 starts with a 15-minute probe (`codex exec - --json --output-last-message /tmp/x.json --output-schema schema.json -C . < /dev/null`) to confirm flag names; treat any drift as a spec issue and update both docs in the same PR.
- **`stream-json` from Claude.** The `result` event has been stable through Claude Code 1.x but the parser must be defensive — anything that isn't NDJSON or that lacks a terminal `result` line is a `_SchemaParseError`, not a crash.
- **`--json-schema` argv-length ceiling.** The exported schema is a few KB; spec §4 calls this out. The implementation sanity-checks `len(schema_str.encode("utf-8")) < 120_000` (Linux `MAX_ARG_STRLEN` is per-argument 131,072 bytes; the 120K threshold leaves headroom for environment growth) and raises `DiscoverySetupError` otherwise. A future schema addition (e.g., adding evidence fields to `Candidate`) could push this past the limit; the test for the check should pin the *current* size so a future growth is loud, not silent.
- **No resume.** Spec §1: "never resumes (a pre-existing `<artifacts_dir>/candidate_discovery/` is a setup error)". The check lives in `discover()` before construction so callers can't accidentally race two `discover()` calls into the same run dir.
- **`gpt-5.5` default.** Spec pins `codex_model="gpt-5.5"`; if the LiteLLM proxy or Codex CLI doesn't surface that exact model id at integration time, the runner emits a clear setup error rather than silently falling back. Adjust the default in the spec, not in code, if it needs to change.
- **Cwd asymmetry.** `Claude` uses `cwd=repo_path` (CLI design — no flag). `Codex` uses `-C repo_path` *and* `cwd=repo_path` (the flag is authoritative; the cwd is defense in depth against relative-path reads escaping the subtree). The runner abstraction must not paper this over — `test_agents.py` asserts both runners receive `cwd=repo_path` and that only Codex carries the `-C` argv pair.
- **Artifact directory location.** Reject `artifacts_dir` if it resolves inside `repo_path`. Otherwise the orchestrator's own `prompt.md`, logs, and `candidates.json` writes would look like target-repo mutations to `RepoGuard`, or worse, require the guard to ignore real files inside the checkout.

- **Test scaffolding.** `tests/integration/` currently has only `__init__.py`. Add `tests/integration/candidate_discovery/` and `tests/unit/candidate_discovery/` for the new suites; include `__init__.py` files only if local fixtures use package-relative imports.
