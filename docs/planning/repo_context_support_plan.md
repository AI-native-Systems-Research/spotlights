# Repo-context support for Stage 1 — Implementation Plan

Add optional repo-level context that the Stage-1 discovery agent receives on
every iteration. Today the agent gets module-level context only (description,
`main_files`, `depends_on`); it has to rediscover the repo's test and
benchmark infrastructure each run, even though the bootstrap prompt at
[prompts_data/bootstrap.md](../../src/spotlights_engine/candidate_discovery/prompts_data/bootstrap.md)
explicitly demands every candidate name a metric, a measurement, and a
correctness oracle. Pre-loading the repo's test commands, benchmark harnesses,
metric conventions, and hard constraints should produce sharper, more
grounded candidates.

The companion authoring prompt for generating `repo_context.md` per target
repo lives at [repo_context.txt](repo_context.txt).

This plan touches one stage only (Stage 1 / `candidate_discovery`). The
Stage-1 spec is
[candidate_research_proposer_stage1_candidate_discovery.md](candidate_research_proposer_stage1_candidate_discovery.md);
its implementation plan is
[candidate_research_proposer_stage1_candidate_discovery_impl.md](candidate_research_proposer_stage1_candidate_discovery_impl.md).
Both must be updated alongside the code change.

Review note from this checkout: the live Stage-1 schema and prompt templates
have already moved beyond the older spec snippets (`Candidate` now has
`description`, `current_approach`, `evolve_rationale`, `metrics`, and
`estimated_impact`; the review prompt no longer has the old "add at most 5"
cap). Treat the live code/templates as the behavior to preserve, and reconcile
the spec + impl-plan drift as part of this change.

---

## 1. Scope

In scope:

- New optional field on `DiscoveryConfig`: `repo_context_markdown: str | None`.
- New `{repo_context}` placeholder rendered into both `bootstrap.md` and
  `review.md`.
- Persisting the supplied markdown into the run dir for reproducibility.
- Tests for substitution, persistence, and the `None` path.
- Spec + impl-plan updates.

Out of scope:

- Auto-generating the markdown. The caller supplies it (typically the
  Stage-3 driver, possibly bootstrapped via the `docs/planning/repo_context.txt`
  prompt run against the target repo).
- Per-module context (already in the prompt).
- Caching / dedup across runs. Each run gets a fresh copy.

---

## 2. API change — `candidate_discovery.api`

Add one optional field to `DiscoveryConfig`
([src/spotlights_engine/candidate_discovery/api.py](../../src/spotlights_engine/candidate_discovery/api.py)):

```python
class DiscoveryConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_path: Path
    module_qualified_name: str
    module: Module
    artifacts_dir: Path
    repo_context_markdown: str | None = Field(
        default=None, min_length=1, max_length=20_000
    )
    num_review_iterations: int = Field(default=3, ge=0)
    per_iteration_wallclock_s: int = Field(default=900, ge=1)
    claude_max_turns: int = Field(default=30, ge=1)
    codex_model: str = Field(default="gpt-5.5", pattern=r"^[\w.\-/]+$")
    codex_reasoning_effort: Literal["minimal", "low", "medium", "high", "xhigh"] = "high"
```

Decisions:

- **String, not `Path`.** Callers already manage the source of truth
  (a `docs/repo_context.md` in the target repo, the cached output of a
  prior generation step, a curated string built from `CLAUDE.md` plus
  `README.md`). Keeping the API a `str` means no hidden filesystem read
  and trivial test setup. Reading a file is a one-liner at the call site.
- **`None` is the default.** This is a strictly additive change. Existing
  callers and tests continue to work unchanged.
- **`min_length=1`.** Reject the empty string. A caller passing `""` is
  a bug — either pass `None` (and get `_(none provided)_` in the prompt)
  or pass real markdown. Without this guard, `""` would render as a blank
  body under the `## Repository context` header.
- **`max_length=20_000`.** Hard cap. Cheap insurance against a caller
  pasting an entire README into every iteration's prompt. The recommended
  size in the authoring prompt is ~5 KB; 20 KB gives 4× headroom before
  it becomes a config problem.

Update `discover()`'s setup checks to validate nothing additional — the
pydantic `max_length` is enough. The schema-bytes guard in
`Orchestrator._mint_run_dir` is unaffected; this string lands in the
*prompt* (stdin), not in `--json-schema` (argv).

Export remains:

```python
__all__ = ["DiscoveryConfig", "DiscoveryResult", "IterationTelemetry", "discover"]
```

---

## 3. Prompt template change — `prompts_data/{bootstrap,review}.md`

Add a single dedicated block to both templates:

- In `bootstrap.md`, place it **after** `## Module under audit` and **before**
  `## What counts as a good evolve target`.
- In `review.md`, place it **after** the `## Inputs` list and **before**
  `## Rules`.

The block uses one new placeholder, `{repo_context}`:

```markdown
## Repository context

{repo_context}
```

Substitution rules:

- When `DiscoveryConfig.repo_context_markdown is None`, substitute the
  literal string `_(none provided)_`. The section header stays so the
  template shape is stable across runs and tests can assert structure.
- When non-`None`, substitute the markdown verbatim. Do not trim, indent,
  or reflow — callers may be relying on specific markdown structure
  (code fences, tables).

Both templates ship as package data via `importlib.resources`, so editing
them is a content change — no code change required to load them.

---

## 4. Prompt assembly change — `candidate_discovery.prompts`

Thread the new value through both renderers in
[src/spotlights_engine/candidate_discovery/prompts.py](../../src/spotlights_engine/candidate_discovery/prompts.py):

```python
_REPO_CONTEXT_DEFAULT = "_(none provided)_"

def _format_repo_context(md: str | None) -> str:
    if md is None:
        return _REPO_CONTEXT_DEFAULT
    return md

def render_bootstrap(
    module_qualified_name: str,
    module: Module,
    *,
    repo_context_markdown: str | None = None,
) -> str:
    values = {
        "module_qualified_name": module_qualified_name,
        "module_name": module.name,
        "module_path": module.path,
        "module_description": module.description or "(none)",
        "depends_on": _format_depends_on(module.depends_on),
        "main_files": _format_main_files(module.main_files),
        "submodule_names": _format_submodule_names(module.submodules),
        "repo_context": _format_repo_context(repo_context_markdown),
    }
    return wrap(_substitute(_BOOTSTRAP, values))

def render_review(
    module_qualified_name: str,
    module: Module,
    prev_candidates_json: str,
    max_seen_candidate_id: str,
    *,
    repo_context_markdown: str | None = None,
) -> str:
    values = {
        "module_qualified_name": module_qualified_name,
        "module_path": module.path,
        "prev_candidates_json": prev_candidates_json,
        "max_seen_candidate_id": max_seen_candidate_id,
        "repo_context": _format_repo_context(repo_context_markdown),
    }
    return wrap(_substitute(_REVIEW, values))
```

The new parameter is keyword-only (note the `*,` separator) on both
renderers. This future-proofs the signature against further positional
additions and makes call sites self-documenting; the `None` default
preserves backward compatibility for any test that calls the renderers
without the new field.

The existing `_substitute` guard ("unknown placeholder is a `KeyError`")
already enforces that `{repo_context}` must be substituted on every call —
no extra code needed.

---

## 5. Orchestrator wiring — `candidate_discovery.orchestrator`

Two call sites in
[orchestrator.py](../../src/spotlights_engine/candidate_discovery/orchestrator.py)
pass the new value through:

```python
boot_prompt = prompts.render_bootstrap(
    self._config.module_qualified_name,
    self._config.module,
    repo_context_markdown=self._config.repo_context_markdown,
)

review_prompt = prompts.render_review(
    self._config.module_qualified_name,
    self._config.module,
    prev_json,
    self._max_seen_id,
    repo_context_markdown=self._config.repo_context_markdown,
)
```

No loop, validation, or agent-invocation semantics change.

### Persistence

Add a helper to `candidate_discovery.layout` so the path lives in one
place (the §13 invariant from the stage-1 impl plan):

```python
def repo_context_path(artifacts_dir: Path) -> Path:
    return candidate_discovery_root(artifacts_dir) / "repo_context.md"
```

In `_mint_run_dir` (orchestrator.py:116), when
`self._config.repo_context_markdown is not None`, write it once via the
helper:

```python
layout.repo_context_path(self._config.artifacts_dir).write_text(
    self._config.repo_context_markdown, encoding="utf-8"
)
```

This:

- Makes runs reproducible — a diff of two runs' artifact dirs reveals
  whether they were fed different context.
- Lets a debugger inspect what the agent actually saw, without having to
  reconstruct the caller's state.

When the value is `None`, do not create the file. Its absence is itself
informative ("this run had no repo context").

---

## 6. Scripts wiring — `scripts/run_*.py`

The four runner scripts in `scripts/` are the only non-test callers of
`discover()` today
([run_epp.py](../../scripts/run_epp.py),
[run_scheduling.py](../../scripts/run_scheduling.py),
[run_scheduling_plugins.py](../../scripts/run_scheduling_plugins.py),
[run_kv_offload.py](../../scripts/run_kv_offload.py)).
They are the path the feature actually ships through, so they must learn to
pass `repo_context_markdown` — otherwise the field exists in the API but is
never set in practice.

Convention: each script reads `docs/repo_context.md` from its `REPO_PATH`
(the authoring prompt at [repo_context.txt](repo_context.txt) writes to that
exact path), and passes the contents through if present. Absent file means
`None` — no error, no warning, just the existing behavior. This keeps the
change a no-op for every target repo that hasn't generated its
`repo_context.md` yet.

Add the same three-line helper to each script, right above the
`DiscoveryConfig(...)` call:

```python
_ctx_path = REPO_PATH / "docs" / "repo_context.md"
_repo_context = _ctx_path.read_text(encoding="utf-8") if _ctx_path.is_file() else None
```

…then thread it into the config:

```python
cfg = DiscoveryConfig(
    repo_path=REPO_PATH,
    ...
    repo_context_markdown=_repo_context,
)
```

Deliberately *not* doing in this PR:

- **No shared helper module.** The four scripts already duplicate
  `_stub_observability_if_missing`, `REPO_PATH`, `ARTIFACTS_DIR`, and
  the `discover()` + print block. Three more lines does not justify a
  `scripts/_runner_common.py` extraction; that refactor is its own PR.
- **No CLI flag.** Path is fixed at `REPO_PATH / "docs" / "repo_context.md"`
  to match the authoring prompt's output. A future caller that wants to
  source from somewhere else can edit the script — the API stays a `str`,
  not a `Path`, exactly so callers can build the string however they like.
- **No fallback to `CLAUDE.md` / `README.md`.** Per §1, sourcing is the
  caller's problem; auto-inferring would silently include irrelevant
  content.

State of the target repos as of this checkout: none of
`/Users/ophir/GoProjects/llm-d-inference-scheduler-main/docs/repo_context.md`,
`/Users/ophir/PycharmProjects/vllm/docs/repo_context.md` exists. Until the
authoring prompt is run against each target, the scripts will pass `None`
and behavior is unchanged.

---

## 7. Tests

All under `tests/unit/candidate_discovery/`. The structure mirrors
[stage1_impl §1](candidate_research_proposer_stage1_candidate_discovery_impl.md).

### `test_prompts.py` — additions

- `repo_context_markdown=None` substitutes `_(none provided)_`; the
  `## Repository context` header is present in both rendered templates.
- A small markdown payload (e.g. `"## Tests\n\n`pytest -q`\n"`) round-trips
  byte-for-byte inside the rendered output (no escaping, no reflow).
- The `{repo_context}` placeholder is fully substituted — no
  `\{(\w+)\}` survives in either rendered template (regression-locked
  by the existing whole-template regex assertion). Run this assertion with
  the default value or a brace-free context payload.
- A payload containing literal `{json}`-shaped braces is preserved verbatim
  in the rendered output. The placeholder regex `\{(\w+)\}` would match
  a bare `{word}` in the final output, but `re.sub` does not rescan
  replacement text — the scan runs once against the *template*, never
  against the substituted value — so user-supplied markdown cannot trigger
  spurious substitutions or `KeyError`s. Test this separately from the
  whole-output "no placeholders remain" assertion.
- `max_length=20_000` is enforced by pydantic on `DiscoveryConfig` (assert
  via `pytest.raises(ValidationError)` on a 20_001-char string).
- `min_length=1` is enforced — an empty string raises `ValidationError`.
  `None` is still accepted and routed through `_(none provided)_`.

### `test_orchestrator.py` — additions to the existing scenarios

Extend the existing `FakeAgentRunner`-driven happy-path test (or add one
scenario):

- **Repo context persistence.** With `repo_context_markdown="## X\n"`,
  after `run()` returns, `<artifacts_dir>/candidate_discovery/repo_context.md`
  exists and its bytes equal the supplied string.
- **No file when `None`.** With `repo_context_markdown=None`, that path
  does **not** exist after `run()` returns.
- **Prompt contains the context.** The `FakeAgentRunner` already writes
  `prompt.md` per iteration via the orchestrator's
  `(iter_dir / "prompt.md").write_text(attempt_prompt, ...)` call
  ([orchestrator.py](../../src/spotlights_engine/candidate_discovery/orchestrator.py)).
  Assert that every iteration's `prompt.md` contains the supplied markdown
  verbatim (proves it threaded through to *every* iteration, not just the
  bootstrap).

No changes to `test_agents.py`, `test_validation.py`, `test_telemetry.py`,
or `test_repo_guard.py`.

---

## 8. Spec and impl-plan updates

This is a contract change to the Stage-1 public API. First reconcile the
existing spec drift called out at the top of this document, then update both
planning docs in the same PR as the code:

- [candidate_research_proposer_stage1_candidate_discovery.md](candidate_research_proposer_stage1_candidate_discovery.md)
  §1: add `repo_context_markdown: str | None = None` to the
  `DiscoveryConfig` block; add one short sentence under §1 describing the
  field's purpose ("optional repo-level markdown context inlined into
  every iteration's prompt; persisted to
  `<artifacts_dir>/candidate_discovery/repo_context.md`"). §3 (run-dir
  layout): add `repo_context.md` to the directory tree alongside
  `iterations.jsonl` and `candidates.schema.json`, with a note that it
  is present only when the caller supplied repo context.
- [candidate_research_proposer_stage1_candidate_discovery_impl.md](candidate_research_proposer_stage1_candidate_discovery_impl.md)
  §4 (Public API): mirror the new field. §8 (Prompts): document the new
  `{repo_context}` placeholder in both bootstrap and review templates;
  add `_format_repo_context` to the formatter helpers; update the
  renderer signatures to take a keyword-only `repo_context_markdown`.
  §13 (Run-dir layout): add `repo_context_path()` to the listed helpers.
  §14 (orchestrator tests): add the three assertions from §7 above.

The live `bootstrap.md` / `review.md` files in `prompts_data/` should remain
mirrored in spec §8 / §9, but this checkout is already drifted. Do not copy
the older spec snippets back over the newer templates; update the spec text to
match the live templates, then add the repository-context block to both.

---

## 9. Implementation order

Small, single-PR change. One ordering that keeps tests green at every step:

1. **Spec + impl-plan edits** — reconcile current drift, then write the new
   repo-context contract in both docs. The PR description quotes the spec diff
   as the source of truth.
2. **Schema field** — add `repo_context_markdown` to `DiscoveryConfig`.
   Tests still green (additive, optional).
3. **Templates** — add the `## Repository context` block to both
   `prompts_data/bootstrap.md` and `prompts_data/review.md`. Existing
   `test_prompts.py` assertions on substitution coverage start failing
   here (good — it proves the regex guard works).
4. **Renderers** — add the `repo_context_markdown` parameter and the
   `_format_repo_context` helper; update both render fns to pass the
   new key. Existing tests now green again.
5. **Orchestrator wiring** — pass the config field through at both call
   sites; persist to `repo_context.md` in `_mint_run_dir` when non-`None`;
   add `repo_context_path` to `layout.py`.
6. **Scripts wiring** — add the `docs/repo_context.md` read + pass-through
   to each of the four `scripts/run_*.py`. With no `repo_context.md` in
   any target repo yet, this step is observably a no-op and is safe to
   land before the first target repo's context is authored.
7. **Tests** — the three additions in §7.

Each step is independently revert-safe.

---

## 10. Risks and notes

- **Token cost per run.** A 5 KB markdown is roughly low-thousands of tokens;
  with one bootstrap + three reviews it is repeated four times. Avoid pricing
  claims in the spec, but flag that a future caller pushing the field toward
  the 20 KB cap multiplies the prompt cost and latency.
- **Argv-length ceiling unaffected.** The schema-byte guard at
  [orchestrator.py](../../src/spotlights_engine/candidate_discovery/orchestrator.py)
  protects the `--json-schema` argv. `repo_context_markdown` lands in the
  prompt body (stdin), not argv — different limit, much larger headroom.
  Do not conflate the two.
- **Sourcing is the caller's problem.** Auto-extraction from `CLAUDE.md`
  or `README.md` is tempting but would silently include irrelevant
  content. The companion authoring prompt at
  [repo_context.txt](repo_context.txt) is the recommended way to
  produce the markdown; it specifies what to include, what to leave out,
  and bounds the size.
- **Staleness.** `repo_context.md` is a snapshot. If the repo's test or
  benchmark surface changes (a new harness lands, a metric is renamed),
  the markdown must be regenerated. This is a caller concern; Stage 1
  does not validate the content.
- **Module-level information is still authoritative.** When the markdown
  and the `Module` record disagree on something module-scoped (e.g. a
  dependency), the `Module` record wins. The renderer puts the Module
  block *before* the Repository context block so the prompt presents
  module info first. The bootstrap already pins `module.path` as the
  scoping rule ("all candidate files MUST lie under this") and tells the
  agent not to propose candidates outside it, so a conflicting repo
  context cannot expand the candidate surface; that implicit precedence
  is what we rely on rather than adding new "module wins" wording to the
  template.
