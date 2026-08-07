# Plan — Two Deep-Research Modes (module vs. candidate)

## Goal

Add a **second, selectable** deep-research path without touching the existing
one. Today's steps 3 (`module_deep_research`) and 4
(`proposal_from_finding_creator`) run a **module-wide** survey followed by a
**cartesian-product** proposal pass. That behavior is valuable and must be
preserved exactly. This plan adds a parallel **per-candidate** path and a flag
that picks which one runs:

1. **`module` mode (default, unchanged).** Step 3 does one module-wide survey;
   step 4 fans out `candidates × all-module-findings`. The existing packages
   and manager branch keep today's runtime behavior; module-mode
   `module_deep_research.json` gains one `"candidate_id": null` key per finding
   (decided in D3 — accepted, not byte-identical).
2. **`candidate` mode (new).** Step 3 runs one survey **per candidate**, keyed
   on that candidate's code site (`current_approach`, `evolve_rationale`,
   `primary_span`); step 4 iterates **each candidate against only its own
   findings** and fills the `Proposal` fields currently unused by the
   deep-research path (`mechanism`, `required_changes`, `expected_effect`,
   `evaluation_metric`).

The per-candidate behavior is exactly what the earlier
[spotlights_deep_research_on_candidates_v2.md](spotlights_deep_research_on_candidates_v2.md)
plan proposed — but that plan **replaced** steps 3 & 4. This plan instead adds
the behavior as **new coexisting packages** selected by a mode flag, so the
module-mode behavior path is never replaced.

**Hard constraint:** the existing `module_deep_research` and
`proposal_from_finding_creator` packages, and the manager code paths that call
them in `module` mode, are not edited for behavior. Every change below is either
(a) purely additive (new packages, new optional schema fields, a new flag), or
(b) a dispatch branch that leaves the existing branch's runtime behavior
identical.

## Current behavior (grounded — re-verified)

- **Step 3 runs once per module.** The manager gate `run_step3` ([orchestrator.py:1139](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1139)) and
  `_do_step3` build one `ModuleDeepResearchInput` ([orchestrator.py:751-760](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L751-L760)) and call `research_module(request, options, segment=segment)` ([orchestrator.py:765-767](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L765-L767)). Note `research_module` in the orchestrator is a **module-level alias for `research_module_with_telemetry`** ([orchestrator.py:106](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L106)) — kept as the historical monkeypatch surface that several tests patch (e.g. `tests/unit/spotlights_manager/test_orchestrator_status.py`), which is why `_do_step3` duck-types the result at [orchestrator.py:769-771](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L769-L771). `research_module_with_telemetry` renders one module-wide prompt ([module_deep_research/api.py:86](../../src/spotlights_engine/module_deep_research/api.py#L86)) via `render_module_deep_research_prompt` ([module_deep_research/prompts.py:70-73](../../src/spotlights_engine/module_deep_research/prompts.py#L70-L73)), fans out runners once ([api.py:87-99](../../src/spotlights_engine/module_deep_research/api.py#L87-L99)), collects usages once ([api.py:100-107](../../src/spotlights_engine/module_deep_research/api.py#L100-L107)), and merges once ([api.py:108-115](../../src/spotlights_engine/module_deep_research/api.py#L108-L115)). Candidates are advisory *hot spots* gated by `include_candidate_hotspots` ([prompts.py:81-89](../../src/spotlights_engine/module_deep_research/prompts.py#L81-L89)). Nothing links a `Finding` back to a `Candidate`.
- **`Finding` has no candidate reference** ([schemas/finding.py:26-36](../../src/spotlights_engine/schemas/finding.py#L26-L36); `model_config = ConfigDict(extra="forbid")`, so an untyped `candidate_id` key would be *rejected* — the field must be declared). Ids are `find-<segment>-NNNN`, minted once in `_renumber_findings` ([module_deep_research/validation.py:177-194](../../src/spotlights_engine/module_deep_research/validation.py#L177-L194)); the module-wide cap is `max_findings_per_module * len(outcomes)` ([orchestration.py:192-199](../../src/spotlights_engine/module_deep_research/orchestration.py#L192-L199)).
- **Step 4 fans out the full cartesian product** `candidates × findings` in `_build_pair_keys` ([proposal_from_finding_creator/api.py:163-171](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L163-L171)), one Claude session per pair, scheduled at [api.py:387-401](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L387-L401). The drop decision already lives inside the agent (empty `proposals` array = drop; per-pair schema `maxItems:1`, [agent_schema.py:32-60](../../src/spotlights_engine/proposal_from_finding_creator/agent_schema.py#L32-L60)). The Claude runner *unwraps* the `{"proposals": [...]}` envelope before returning, so `PairRunResult.structured_output` is the bare list and `parse_pair_payload` validates a `list` ([claude_exec.py:203-216](../../src/spotlights_engine/proposal_from_finding_creator/claude_exec.py#L203-L216), [validation.py:44-51](../../src/spotlights_engine/proposal_from_finding_creator/validation.py#L44-L51)) — the new package must preserve that wrapper/unwrap contract.
- **The agent-facing `DeepResearchProposal`** has only `title`, `detailed_description`, `finding_id`, `proposal_rationale`, `created_by` ([schemas/proposals.py:14-27](../../src/spotlights_engine/schemas/proposals.py#L14-L27); also `extra="forbid"`). `_convert_proposal` maps it to the unified `Proposal` ([api.py:321-335](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L321-L335)), leaving `mechanism`/`required_changes`/`expected_effect`/`evaluation_metric` (all optional on `Proposal`, [schemas/proposal.py:30-33](../../src/spotlights_engine/schemas/proposal.py#L30-L33)) at `None`.
- **Manager persistence.** Step 3 output → `module_deep_research.json`, step 4 → `proposal_from_finding_creator.json` ([persistence.py:184-186,196-198](../../src/spotlights_engine/spotlights_manager/persistence.py#L184-L198)). The resume ladder gates on `state.deep_research is None` / `state.proposal_from_finding is None` ([orchestrator.py:629-638](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L629-L638)); checkpoint states are `DEEP_RESEARCHED` / `FINDING_PROPOSALS_CREATED` ([persistence.py:66-67](../../src/spotlights_engine/spotlights_manager/persistence.py#L66-L67)). Zero findings short-circuits step 4 for the whole module ([orchestrator.py:1264-1282](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1264-L1282), `_synthetic_step4_output_for_zero_findings` at [orchestrator.py:774-788](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L774-L788)).
- **The step-3 codex last-message path is a single file per module**, `module_deep_research.last_message.md` ([persistence.py:188-190](../../src/spotlights_engine/spotlights_manager/persistence.py#L188-L190)), threaded into `CodexExecOptions.output_last_message` by `_build_deep_research_options` ([orchestrator.py:230-250](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L230-L250), used at [orchestrator.py:761-763](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L761-L763)). This is **not** debug-only: the codex runner reads the file back as `final_message` ([codex_exec.py:169-171](../../src/spotlights_engine/module_deep_research/codex_exec.py#L169-L171)) and `merge_outcomes` parses `final_message or stdout` ([orchestration.py:158](../../src/spotlights_engine/module_deep_research/orchestration.py#L158)). One codex invocation per module makes the single path safe **in module mode** — this is why candidate mode needs its own per-candidate directory (D6 below), while module mode's single-file path stays untouched. (Note: when `output_last_message` is `None`, `build_command` mints a *fresh temp file per invocation* ([codex_exec.py:70-78](../../src/spotlights_engine/module_deep_research/codex_exec.py#L70-L78)), so leaving it unset would also be collision-free — but then the artifact isn't persisted under the module dir, which is why D6 mints explicit per-candidate paths.)
- **Step-3 usage writes** stamp every runner record `deep-research:<cli>` and use the whole-step `dr_duration` as the fallback duration ([orchestrator.py:1214-1229](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1214-L1229)). One survey per module means this is correct today. The generic sink is `_write_cli_usage_records(usages: list[tuple[str, str, Any, float | None]])` — `(invocation_id, cli, usage, fallback_duration)` ([orchestrator.py:374-398](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L374-L398)); step 5 already builds `<candidate_id>:<cli>` tuples through it ([orchestrator.py:1519-1535](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1519-L1535)), which is the exact precedent candidate mode should copy.
- **Step-4 pair-count log** computes `|C|×|F|` at [orchestrator.py:1284](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1284); the `debug_first_n_pairs` warning recomputes it at [api.py:378-385](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L378-L385).
- **Renderer** counts relevant findings from `proposal.finding_ref_id ∩ findings.finding_id` ([results_renderer/aggregator.py:203-215](../../src/spotlights_engine/results_renderer/aggregator.py#L203-L215)); this is mode-agnostic.
- **Resume/fingerprint.** `build_input_fingerprint` currently emits `repo_path`, `context_hash`, `max_findings_per_module`, `continue_on_module_failure`, `include_candidate_hotspots`, `enable_claude_search` ([persistence.py:282-298](../../src/spotlights_engine/spotlights_manager/persistence.py#L282-L298)); the call site is [orchestrator.py:1715-1722](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1715-L1722). `SCHEMA_VERSION = 4` ([persistence.py:60](../../src/spotlights_engine/spotlights_manager/persistence.py#L60)); `_ensure_resume_compatible` rejects a changed schema version ([orchestrator.py:455-463](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L455-L463)) or a changed input fingerprint ([orchestrator.py:465-470](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L465-L470)); the config fingerprint is separate and has its own one-shot forward migration ([orchestrator.py:472-499](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L472-L499)).
- **`SpotlightsManagerConfig` lives in `spotlights_manager/api.py`, not `schemas/`** ([api.py:36-58](../../src/spotlights_engine/spotlights_manager/api.py#L36-L58)) and holds the per-step infra config slots (`deep_research: CodexExecOptions | None`, `proposal_from_finding: ProposalFromFindingConfig | None`). Candidate mode's step-4 config must be reachable from here (see New step 4 / config).

## Design decisions

**D1 — Two new packages, existing two left alone.** Add:
- `candidate_deep_research` — mirrors `module_deep_research` but loops over
  candidates, producing a `ModuleDeepResearchOutput` whose findings carry
  `candidate_id`.
- `proposal_from_candidate_finding_creator` — mirrors
  `proposal_from_finding_creator` but groups findings by `candidate_id` and
  fills the four structured `Proposal` fields.

  Names chosen to read as siblings of the existing packages while making the
  scope explicit (`candidate_*`). They are separate top-level packages under
  `src/spotlights_engine/`, so the existing packages' imports, tests, and public
  surfaces are untouched. Where logic is genuinely identical the new packages
  **import and reuse** the existing helpers rather than fork them — the reused
  functions are pure/stateless: for step 3, `parse_agent_output`,
  `select_runners`, and the whole `merge_outcomes`
  ([orchestration.py:127-199](../../src/spotlights_engine/module_deep_research/orchestration.py#L127-L199)) called **once per candidate**. Verified: `merge_outcomes(outcomes, *, max_findings_per_module: int, segment: str)` takes the cap and segment as keyword parameters and allocates `seen: set[str]` **locally per call** ([orchestration.py:148](../../src/spotlights_engine/module_deep_research/orchestration.py#L148)) — `seen` is *not* a parameter, it is call-scoped, which is even better for us: one call per candidate gives per-candidate dedup with no fork and nothing to thread. Verified: `select_runners(*, repo_path, codex_options, runner, runners, enable_claude_search=False)` ([orchestration.py:60-85](../../src/spotlights_engine/module_deep_research/orchestration.py#L60-L85)) takes the `CodexExecOptions` object and `enable_claude_search` as keywords, so a candidate-local options object can be passed straight in. For step 4: `ensure_claude_available`, `run_pair`/`default_run_pair`, `PairRunResult`, `_PairRunner`, and `mint_proposal_ids`. Importing these from the new package cannot change their behavior in the old one. **Do not import `run_runners` unchanged:** verified that `RunnerOutcome` is `agent_name`/`result`/`error` only, with **no duration field** ([orchestration.py:29-35](../../src/spotlights_engine/module_deep_research/orchestration.py#L29-L35)); `_run_one_runner` measures elapsed time but only logs it and discards it ([orchestration.py:210-254](../../src/spotlights_engine/module_deep_research/orchestration.py#L210-L254)). Since D8 needs per-run fallback durations, candidate mode writes a **small local timed fan-out** in the new package (same `ThreadPoolExecutor` shape as `run_runners`, returning `list[tuple[RunnerOutcome, float]]`). Do **not** add `RunnerOutcome.duration_s` to the old package: `RunnerOutcome` is a frozen dataclass constructed positionally/by-keyword in the old code and touching it violates the hard constraint for no gain.
  **Exception — the prompt/schema/validation-bound orchestration in step 4 is copied, not imported.** Verified: `_run_one_pair` calls `build_per_pair_schema_text` ([api.py:235](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L235)), `build_prompt` ([api.py:239](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L239)), and `parse_pair_payload` ([api.py:296](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L296)) as **module-global names** bound by the imports at [api.py:21-38](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L21-L38); `_run_async` calls `_build_pair_keys`/`_run_one_pair`/`_rebuild_candidate` the same way, and `_rebuild_candidate` calls `_convert_proposal`. Importing any of them unchanged would silently run the old prompt/schema and never emit the four new fields (see New step 4). The new package forks `_build_pair_keys`, `_run_one_pair`, `_run_async`, `_convert_proposal`, `_rebuild_candidate`, `_validate_setup`, `_last_messages_dir`, and `_persist_pair_debug` so they bind its own `prompts`/`agent_schema`/`validation` and its own last-messages directory name. (`_last_messages_dir` hardcodes `"proposal_from_finding_creator.last_messages"` at [api.py:174-177](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L174-L177); the new package must use its own subdirectory name so a candidate-mode `--redo` clears the right tree — see Persistence.)

**D2 — Mode selected by `deep_research_mode: Literal["module", "candidate"]` on
`SpotlightsManagerInput`, default `"module"`.** The manager dispatches inside
`_do_step3` and `_do_step4`. Everything in the per-module *state machine*
(`_plan_module` at [orchestrator.py:603](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L603), the resume ladder `_ladder_to_first_missing` at [orchestrator.py:629-638](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L629-L638), checkpoint statuses, sidecar filenames)
is shared and mode-agnostic — see D4. The two *behavioral* dispatch points are
`_do_step3` / `_do_step4`; three further manager sites gain a mode guard whose
`module` branch is textually the current code (the pair-count log at
[orchestrator.py:1284](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1284), the usage-writes at [orchestrator.py:1214-1229](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1214-L1229), and the fingerprint call at [orchestrator.py:1715-1722](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1715-L1722) — all detailed under Manager changes). Everything else, plus the two new packages, is additive.
  `_do_step3`/`_do_step4` already receive `mgr_input: SpotlightsManagerInput`
  ([orchestrator.py:739](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L739), [orchestrator.py:795](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L795)), so the mode is read off `mgr_input.deep_research_mode`
  with **no new parameter threading**. Both call sites
  ([orchestrator.py:1154-1162](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1154-L1162), [orchestrator.py:1291-1300](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1291-L1300)) stay byte-identical.
  **Keep the `research_module = research_module_with_telemetry` alias intact and keep the module-mode branch calling it through that name** — existing tests monkeypatch `orch.research_module`, so inlining the real import there would break them. Give the candidate branch its own module-level alias (e.g. `research_candidates = research_candidates_with_telemetry`) so candidate-mode tests get the same seam.

**D3 — New schema fields are additive and optional; module-mode JSON accepts the
serialized `null` (DECIDED).** `Finding.candidate_id` is
`str | None = None`; module-mode findings never set it. With today's
`P.write_deep_research` (`output.model_dump(mode="json")`, [persistence.py:531-540](../../src/spotlights_engine/spotlights_manager/persistence.py#L531-L540)), adding the field
means module-mode `module_deep_research.json` gains
`"candidate_id": null` on each finding. **Accept this.** It is
resume-compatible in both directions — new code reading an old sidecar defaults
the field to `None`, and old sidecars already round-trip through
`read_module_state`'s `model_validate` ([persistence.py:400-500](../../src/spotlights_engine/spotlights_manager/persistence.py#L400-L500)) — and no runtime
behavior changes; only the sidecar gains a key. **Do not** add a mode-specific
dump path or `exclude_none`: `write_deep_research` is a shared, mode-agnostic
writer, and forking it to serialize differently per mode is more surface area
than the byte-identity is worth. Nothing in the repo asserts byte-identical
`module_deep_research.json` (checked: the only tests touching that file are
`tests/unit/spotlights_manager/test_persistence.py`, which asserts *loading*
semantics, and `tests/integration/spotlights_manager/test_end_to_end_subset.py`,
which asserts existence). The four new `DeepResearchProposal` fields are optional; module mode
never emits them and `_convert_proposal` in the **old** package is not touched,
so module-mode `Proposal`s keep `mechanism=…=None` exactly as today (they are
already serialized as `null` in `proposal_from_finding_creator.json` today,
since `Proposal` already declares all four — [schemas/proposal.py:30-33](../../src/spotlights_engine/schemas/proposal.py#L30-L33) — so step 4's
sidecar shape does not change at all).

**D4 — Candidate mode reuses the existing sidecars, checkpoint states, and resume
ladder.** Only one mode runs per run (they never coexist in a module dir — D5's
fingerprint rule makes a mid-run mode switch a hard `ResumeMismatchError`), and
both modes emit a `ModuleDeepResearchOutput` (step 3) and a
`ProposalFromFindingCreatorOutput` (step 4). So candidate mode writes
`module_deep_research.json` / `proposal_from_finding_creator.json` through the
same `P.write_deep_research` / `P.write_proposal_from_finding`, advances the same
`DEEP_RESEARCHED` / `FINDING_PROPOSALS_CREATED` checkpoints, and resumes through
the same ladder. This means **zero changes to `persistence.read_module_state`,
`_plan_module`, `_ladder_to_first_missing`, or the checkpoint enum.** It also
means **no new `PipelineStep` or `UsageStep` literals**: candidate mode reports
issues under `step="module_deep_research"` /
`step="proposal_from_finding_creator"` and writes usage records into
`module_deep_research.usage` / `proposal_from_finding_creator.usage`, because
both literal unions are closed (`schemas/common.py:15-22`,
`costing/records.py:19-24`) and widening them would ripple through
`ModuleCheckpoint`, `read_usage_records`, and `compute_external_cost`. The new
packages' `StepIssue` builders therefore reuse the **existing** step names.
The one persistence addition is the per-candidate last-message *directory* (D6),
which is purely additive.

*Tension:* accepting serialized `candidate_id: null` changes the
`module_deep_research.json` shape for newly-written module-mode runs, and adding
`deep_research_mode` to the input fingerprint would change the resume key if it
were emitted unconditionally. Neither changes module-mode **runtime behavior**,
but both affect compatibility expectations around existing run dirs. The
fingerprint side is handled by the mode-specific rule in D5; the JSON side is
decided in D3 (accept the `null`).

**D5 — The fingerprint records only the knobs effective for the selected mode.**
In `module` mode, `build_input_fingerprint` emits exactly today's dict:
`repo_path`, `context_hash`, `max_findings_per_module`,
`continue_on_module_failure`, `include_candidate_hotspots`, and
`enable_claude_search`. It omits `deep_research_mode` and
`max_findings_per_candidate`, so a pre-existing module-mode run dir resumes with
no mismatch and no `SCHEMA_VERSION` bump. In `candidate` mode, the fingerprint
emits `repo_path`, `context_hash`, `continue_on_module_failure`,
`enable_claude_search`, `deep_research_mode`, and
`max_findings_per_candidate`, and omits module-only knobs such as
`max_findings_per_module` and
`include_candidate_hotspots` unless the candidate branch actually consumes them.
`enable_claude_search` is shared, not module-only: candidate mode still uses it
to choose Codex-only vs. Codex+Claude step-3 surveys (verified — it flows into
the reused `select_runners`, [orchestration.py:83-85](../../src/spotlights_engine/module_deep_research/orchestration.py#L83-L85)). This prevents false
candidate-mode resume mismatches when an unused module-mode knob changes, while
still making module-mode and candidate-mode fingerprints distinct. Switching
`--deep-research-mode` on resume raises a clean `ResumeMismatchError` from the
input-fingerprint comparison at [orchestrator.py:465-470](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L465-L470) — because the candidate
fingerprint carries the extra `deep_research_mode` key, the two dicts can never
compare equal in either direction.
(Alternative — always emit the key and bump
`SCHEMA_VERSION` — is simpler but invalidates every existing run dir; rejected
because the constraint is that module mode is unchanged, including its resume.)
**Implementation shape:** give `build_input_fingerprint` keyword-only params with
defaults that reproduce today's dict when the caller passes only today's
arguments — i.e. `deep_research_mode: str = "module"`,
`max_findings_per_candidate: int | None = None` — and branch on
`deep_research_mode` for the returned key set. That way any other existing caller
of `build_input_fingerprint` (and any test constructing an expected fingerprint)
keeps working unchanged.

**D6 — Candidate mode gets a per-candidate last-message directory; module mode
keeps its single file.** Candidate mode runs N codex surveys per module through
derived `CodexExecOptions`; a single `output_last_message` path would let one
candidate's survey parse another's JSON (the runner reads the file back as
`final_message`, see Current behavior). Add a new
`deep_research_last_message_dir` property returning
`module_deep_research.last_messages/` (mirroring step 4's existing
`proposal_from_finding_creator.last_messages/`, [persistence.py:200-202](../../src/spotlights_engine/spotlights_manager/persistence.py#L200-L202)); candidate-mode step 3 mints `…/<candidate_id>.md` per survey via
`options.model_copy(update={"output_last_message": …})` (`CodexExecOptions` is a
pydantic model with `output_last_message: Path | str | None`,
[codex_exec.py:23-42](../../src/spotlights_engine/module_deep_research/codex_exec.py#L23-L42), so `model_copy` is the right non-mutating derivation and
`build_command` already `mkdir -p`s the parent, [codex_exec.py:77-78](../../src/spotlights_engine/module_deep_research/codex_exec.py#L77-L78)).
Do **not** hand Codex a
directory as `output_last_message`. Concretely: `_build_deep_research_options`
takes `last_message_path: Path` positionally today ([orchestrator.py:230-234](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L230-L234)) —
widen it to `Path | None` (default keeps the module-mode call identical), have the
candidate branch pass `None` to get base options, and thread the *directory* to
`research_candidates_with_telemetry` as a separate `last_message_dir` argument so
the concrete per-candidate file path is derived inside the candidate loop.
`<candidate_id>` is filesystem-safe by construction: `Candidate.id` matches
`^cand-[A-Za-z0-9._-]+-\d{4}$` ([schemas/candidate.py:75](../../src/spotlights_engine/schemas/candidate.py#L75)), so it contains no path
separators. **Module mode keeps using
`deep_research_last_message_path` unchanged** — the dispatch in `_do_step3` picks
which one to build.

**D7 — Per-candidate cap and per-candidate concurrency, candidate mode only.** Add
`max_findings_per_candidate: int = Field(default=10, ge=0)` to `SpotlightsManagerInput`
and the new step's input. `max_findings_per_module` is left exactly as-is and
continues to drive module mode. The two caps coexist; each mode reads its own.
Per-candidate concurrency is an infra knob. **Do not introduce a new
`CandidateDeepResearchConfig` model or a new `SpotlightsManagerConfig` slot for it** —
that would drag in another `build_config_fingerprint` entry (see New step 4 for why
that is dangerous) for a single integer. Instead make it a keyword-only argument on
`research_candidates_with_telemetry`: `max_parallel_candidates: int = 2`. Candidate
mode's step 3 continues to receive its `CodexExecOptions` through the existing
`cfg.deep_research` slot and `_build_deep_research_options`, exactly as module mode does.
**Default 2** — cost-conservative by design (see Decisions confirmed #6); it composes
multiplicatively with the manager's own `max_parallel_sessions` (default 1). The naming
and semaphore pattern match `AgentProposalsConfig.max_parallel_candidates`
([agent_proposals/api.py:83](../../src/spotlights_engine/agent_proposals/api.py#L83), whose default is 5 because step 5 is one session per
candidate rather than one *per candidate per runner*).

**D8 — Candidate-scoped observability in candidate mode only.** Candidate-mode
step 3 returns per-(candidate, runner) usages and durations so the manager writes
distinct usage records (`<candidate_id>:<cli>`) with per-run fallback durations,
and stamps each `SearchQueryLog` with `candidate_id`. Module mode's
usage-writing tuple builder ([orchestrator.py:1220-1228](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1220-L1228)) is untouched; the candidate branch builds different tuples through the same generic `_write_cli_usage_records`, whose signature already accepts arbitrary
`(invocation_id, cli, usage, fallback_duration)` tuples ([orchestrator.py:374-398](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L374-L398)) and whose
`step`/`role` stay `"module_deep_research"`/`"deep_research"` (D4). The on-disk
record filename is `s<k>.i<NNNN>.<cli>.json` derived from `invocation_index`
([costing/records.py:61-63](../../src/spotlights_engine/costing/records.py#L61-L63)), i.e. from the tuple list's enumeration order — so N
candidates × M runners simply produce N×M records with distinct indices and no
collision; `invocation_id` is the human-readable label only.

## Schema changes (all additive; module-mode runtime unaffected)

1. **`schemas/finding.py`** — add `candidate_id: str | None = None` (pattern
   `^cand-[A-Za-z0-9._-]+-\d{4}$` when present) to `Finding` ([finding.py:26-36](../../src/spotlights_engine/schemas/finding.py#L26-L36)). Optional so old public reports / ad hoc sidecars still load and so **module-mode findings simply leave it `None`**. The candidate path always sets it. Update the module docstring to say the field is populated only in candidate mode.
   Do **not** add the field to `module_deep_research.validation.AgentFinding`
   ([validation.py:32-44](../../src/spotlights_engine/module_deep_research/validation.py#L32-L44)) — the wire schema stays candidate-unaware and the tag is applied
   post-merge (see New step 3). `Finding` is `extra="forbid"`, so declaring the
   field in `schemas/finding.py` is mandatory, not optional polish.
2. **`schemas/proposals.py`** — add `mechanism`, `required_changes`,
   `expected_effect`, `evaluation_metric` (all
   `str | None = Field(default=None, min_length=1)`) to `DeepResearchProposal`
   ([proposals.py:14-27](../../src/spotlights_engine/schemas/proposals.py#L14-L27)). `DeepResearchProposal` is `extra="forbid"`, so this declaration is
   also mandatory: without it `parse_pair_payload` in the new package would reject
   every agent payload carrying the four fields. The **old** step-4 agent schema/prompt/validation do not reference them, so module mode never produces them; the new candidate package's JSON schema makes them required/non-empty when it emits a proposal.
3. **`schemas/pipeline.py`**
   - `SpotlightsManagerInput` ([pipeline.py:151-162](../../src/spotlights_engine/schemas/pipeline.py#L151-L162)): add `deep_research_mode: Literal["module", "candidate"] = "module"` and `max_findings_per_candidate: int = Field(default=10, ge=0)`. Both default such that a caller who sets neither gets today's behavior. (`Literal` is already imported in this module, [pipeline.py:14](../../src/spotlights_engine/schemas/pipeline.py#L14).) `SpotlightsManagerInput` is `extra="forbid"`, so callers passing the new kwargs need these declarations.
   - **New input contract** `CandidateDeepResearchInput` (dedicated, so
     `ModuleDeepResearchInput` at [pipeline.py:52-72](../../src/spotlights_engine/schemas/pipeline.py#L52-L72) is untouched): same fields as `ModuleDeepResearchInput` minus `include_candidate_hotspots`, with `max_findings_per_candidate` in place of `max_findings_per_module` and `candidates` promoted from "hints" to "iteration set" (so it keeps `project_tree`, `module_qualified_name`, `context`, `repo_path`, `candidates`, `enable_claude_search`). `ModuleDeepResearchOutput` is reused as-is (findings now *may* carry `candidate_id`).
   - Step 4's `ProposalFromFindingCreatorInput` ([pipeline.py:88-100](../../src/spotlights_engine/schemas/pipeline.py#L88-L100)) is reused by the new package unchanged (still `candidates` + flat `findings`); the new package groups by `finding.candidate_id` internally. Update its docstring to note both consumers.
   - **Export surface:** add `CandidateDeepResearchInput` to `schemas/pipeline.py`'s
     `__all__` ([pipeline.py:196-209](../../src/spotlights_engine/schemas/pipeline.py#L196-L209)) **and** to the `schemas/__init__.py` re-export block
     ([schemas/__init__.py:45-57](../../src/spotlights_engine/schemas/__init__.py#L45-L57) plus its `__all__` and the layout docstring) — `schemas/` is
     documented as the only cross-repo-importable surface, so a new contract that
     isn't re-exported there is an inconsistency the next reader will trip on.
4. **`schemas/search.py`** — add optional `candidate_id: str | None = None` to
   `SearchQueryLog` ([search.py:26-38](../../src/spotlights_engine/schemas/search.py#L26-L38)). The type is lenient (`extra="ignore"`, no min-lengths) but that only means unknown keys are *dropped*, not stored — so the field must still be declared to survive a round-trip; module mode leaves it `None`.

## Step 3 — existing `module_deep_research` (UNCHANGED)

No edits. `render_module_deep_research_prompt`, `research_module_with_telemetry`,
`merge_outcomes`, `_renumber_findings`, and the `max_findings_per_module` cap all
run exactly as today when `deep_research_mode == "module"`.

## New step 3 — `candidate_deep_research` (candidate mode)

A new package modeled on `module_deep_research`:

- **`prompts.py`** — `render_candidate_deep_research_prompt(request, module, candidate)`: a per-candidate survey prompt focused on one candidate's `primary_file`/`primary_span` (symbol, kind, lines via `utils/schema_compat` accessors), `description`, `current_approach`, `evolve_rationale`, `estimated_impact(_explanation)`, plus module/repo context for grounding. Step 4's `build_prompt` ([proposal_from_finding_creator/prompts.py:45-55](../../src/spotlights_engine/proposal_from_finding_creator/prompts.py#L45-L55)) already renders exactly this candidate block and is the shape to copy. Reuse the existing `AgentModuleDeepResearchOutput` wire schema (bare `find-NNNN` ids) unchanged.
- **`api.py`** — `research_candidates_with_telemetry(request: CandidateDeepResearchInput, codex_options, *, segment, last_message_dir, check=False, runner=None, runners=None)`: resolve the module via `resolve_target_module` — **note there are two functions with that name and different behavior**: [module_deep_research/api.py:37-39](../../src/spotlights_engine/module_deep_research/api.py#L37-L39) delegates straight to `project_tree.resolve`, while [orchestration.py:48-58](../../src/spotlights_engine/module_deep_research/orchestration.py#L48-L58) additionally falls back from dot-qualified to slash-qualified names. `research_module_with_telemetry` uses the strict `api.py` one (it is defined locally and shadows nothing). Import the `api.py` one for behavioral parity with module mode. Then return the same early `module not found` output the module path returns ([module_deep_research/api.py:71-83](../../src/spotlights_engine/module_deep_research/api.py#L71-L83) — note that issue is `recoverable=False`, i.e. it fails the module; mirror that exactly), then loop over `request.candidates`. For each candidate, derive a candidate-local `CodexExecOptions` first (distinct `output_last_message`, D6), build its prompt, call the reused `select_runners` with that candidate-local options object and `enable_claude_search=request.enable_claude_search`, run a candidate-local timed fan-out over those runners, then **call the existing `merge_outcomes` once per candidate** ([orchestration.py:127-199](../../src/spotlights_engine/module_deep_research/orchestration.py#L127-L199)) with `max_findings_per_module=max_findings_per_candidate` and `segment=f"{module_segment}-{candidate_counter:04d}"`. Derive `candidate_counter` from `parse_id(candidate.id)[2]` and verify `parse_id(candidate.id)[1] == segment`; do not split ids on `-`. One call per candidate gives the per-candidate `seen` dedup scope (it is allocated inside `merge_outcomes`, not passed in) and the per-candidate finding segment for free (no fork of `merge_outcomes`), and its internal cap `max_findings_per_module * len(outcomes)` becomes the per-candidate merged cap (`max_findings_per_candidate` times the selected runner outcome count). **`merge_outcomes`/`_renumber_findings` do not set `candidate_id`** (they build bare `Finding`s at [validation.py:184-194](../../src/spotlights_engine/module_deep_research/validation.py#L184-L194) and un-tagged `SearchQueryLog`s at [orchestration.py:177-190](../../src/spotlights_engine/module_deep_research/orchestration.py#L177-L190)), so after each per-candidate merge, stamp `candidate_id=candidate.id` onto the returned `.findings` and `.search_queries` via `model_copy(update=…)` — this is a post-merge pass, not something `merge_outcomes` does. Concatenate the per-candidate outputs into one `ModuleDeepResearchOutput` (findings, `issues`, **and** `search_queries` — all three lists must be concatenated, not just findings). Return per-(candidate, runner) usage records (candidate-tagged) and durations for D8, reusing `_cli_for_agent` ([module_deep_research/api.py:51-57](../../src/spotlights_engine/module_deep_research/api.py#L51-L57)) to map agent name → `"claude"`/`"codex"`, since only those two are legal `UsageCli` values. Bound per-candidate concurrency with a keyword-only `max_parallel_candidates: int = 2` argument (D7 — no new config model), copying the semaphore pattern at [agent_proposals/api.py:529](../../src/spotlights_engine/agent_proposals/api.py#L529). Note this function is **synchronous** like `research_module_with_telemetry` (the manager wraps it in `asyncio.to_thread`), so use a `ThreadPoolExecutor(max_workers=max_parallel_candidates)` rather than an `asyncio.Semaphore` unless the whole entrypoint is made `async` — which would break the `asyncio.to_thread(lambda: …)` call shape the dispatch reuses. Note `_cli_for_agent` returns `str | None` — skip usage attribution for an unrecognized agent name rather than inventing a `UsageCli`. Also expose a thin `research_candidates(...) -> ModuleDeepResearchOutput` sibling mirroring `research_module`, so standalone/architecture-shaped callers exist for both modes.
- **Finding-id scheme (DECIDED).** Mint `find-<module_segment>-<candidate_counter>-NNNN` where `<candidate_counter>` is the zero-padded 4-digit counter parsed out of the candidate id — i.e. the segment handed to `merge_outcomes` is `f"{module_segment}-{candidate_counter:04d}"`. Verified against the real regexes: `find-src_utils-0003-0007` matches `Finding.finding_id`'s `^find-[A-Za-z0-9._-]+-\d{4}$`, and `parse_id` (right-anchored via `_ID_RE = ^(cand|find|prop)-(.+)-(\d{4})$` at [id_helpers.py:60](../../src/spotlights_engine/utils/id_helpers.py#L60), function at [id_helpers.py:67-81](../../src/spotlights_engine/utils/id_helpers.py#L67-L81)) recovers `("find", "src_utils-0003", 7)` — greedy middle stops at the *last* `-\d{4}`, so a slug containing `-` still parses. `prefix_local_id("find-0007", expected_type="find", segment="src_utils-0003")` also works and stays idempotent on resume. Do **not** infer candidate ids back out of finding ids — use `Finding.candidate_id`.
- **Cap.** Effective per-candidate cap = `max_findings_per_candidate * len(outcomes)` (normally the selected runner count); module total is roughly `candidates × max_findings_per_candidate × selected runner outcomes` (cost driver — see Decisions confirmed #6).
- **Failure granularity.** Wrap each candidate's fan-out so an exception degrades only that candidate: append a recoverable `StepIssue` naming the candidate (built with `step="module_deep_research"`, reusing `module_deep_research_issue` at [orchestration.py:38-45](../../src/spotlights_engine/module_deep_research/orchestration.py#L38-L45)) and continue the loop. `merge_outcomes` already converts per-runner failures into recoverable issues, so a single runner crash inside one candidate is handled for free; the wrapper covers prompt/options/derivation errors. Recoverable issues make the module `DEGRADED` rather than `FAILED` via the untouched `_final_status` ([orchestrator.py:196-216](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L196-L216)).
- **Reuse vs fork.** `agent_exec`, `codex_exec`, `claude_exec` runners are imported from `module_deep_research`, not copied.

## Step 4 — existing `proposal_from_finding_creator` (UNCHANGED)

No edits. `_build_pair_keys` (cartesian), `build_prompt`,
`build_per_pair_schema_text`, `parse_pair_payload`, `_convert_proposal`, and
`will_invoke_claude` all run exactly as today in module mode. The four new
`Proposal` fields are simply never populated on this path.

## New step 4 — `proposal_from_candidate_finding_creator` (candidate mode)

A new package modeled on `proposal_from_finding_creator`:

- **`api.py`** — replace the cartesian `_build_pair_keys` with per-candidate grouping: for each candidate `c`, findings = `[f for f in input.findings if f.candidate_id == c.id]`; pairs are `(c, f)` only for that candidate's own findings (`Σ|F_c|` instead of `|C|×|F|`). Reuse the semaphore pattern and post-gather id minting (`mint_proposal_ids`, [schema_compat.py:83-105](../../src/spotlights_engine/utils/schema_compat.py#L83-L105)) unchanged. Retarget `will_invoke_claude` ([proposal_from_finding_creator/api.py:478](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L478), currently `bool(input.candidates.candidates) and bool(input.findings)`) to "at least one pair survives grouping" (a module whose findings all belong to out-of-input candidates yields zero pairs). Treat findings with missing or out-of-input `candidate_id` as recoverable candidate-mode issues before dropping them, so a broken candidate-mode step 3 cannot silently erase all step-4 work. Extend `_convert_proposal` to copy the four new fields onto the unified `Proposal`.
  - **`_run_one_pair` must be *forked*, not imported unchanged (correctness).** The existing `_run_one_pair` ([api.py:218-318](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L218-L318)) binds `build_prompt`, `build_per_pair_schema_text`, and `parse_pair_payload` from the **old** package's module globals (imported at [api.py:21-38](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L21-L38)). If the new package imported `_run_one_pair` unchanged, it would still call the old prompt/schema/validation, so the four new fields would never be prompted, schema-allowed, or parsed. The new package therefore **copies** `_run_one_pair`, `_run_async`, `_rebuild_candidate`, and `_convert_proposal` (the orchestration + prompt/schema-bound helpers) so they bind the new `prompts`/`agent_schema`/`validation` modules. Only genuinely pure/stateless helpers are imported (see Reuse below).
  - The copied `_run_async` must also fix the `debug_first_n_pairs` warning count: report the grouped pre-truncation pair count, not the old `len(candidates) * len(findings)` cartesian count.
  - Also fork `_validate_setup` and `_last_messages_dir` ([api.py:174-177](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L174-L177), which hardcodes `"proposal_from_finding_creator.last_messages"`) plus `_persist_pair_debug`, so the new package writes its debug artifacts under its own directory name and doesn't collide with a module-mode run in the same module dir.
- **`prompts.py`** — strengthen the relevance gate ("is this finding really relevant to THIS candidate? if not, emit `[]`") and add output rules for the four new fields (required only when a proposal is emitted).
- **`agent_schema.py`** — add the four fields to the per-pair proposal object schema. **Preserve the wrapper contract:** the schema's top level is `{"proposals": [<proposal object>, …]}` and `claude_exec._unwrap_proposals` ([claude_exec.py:203-216](../../src/spotlights_engine/proposal_from_finding_creator/claude_exec.py#L203-L216)) / `parse_pair_payload` ([validation.py:44-51](../../src/spotlights_engine/proposal_from_finding_creator/validation.py#L44-L51)) both depend on it. Keep `finding_id` and `created_by` pinned via `const` as today ([agent_schema.py:44-58](../../src/spotlights_engine/proposal_from_finding_creator/agent_schema.py#L44-L58)) — the new fields are added *inside* the proposal object only.
- **`validation.py`** — accept and pass through the four fields. Note the new fields must also be declared on `DeepResearchProposal`, which is `extra="forbid"` ([schemas/proposals.py:20-27](../../src/spotlights_engine/schemas/proposals.py#L20-L27)) — otherwise a conforming agent payload is rejected at validation. See Schema changes.
- **`config.py` / config class (DECIDED).** Define a sibling `ProposalFromCandidateFindingConfig` with the same infra knobs (`repo_path`, `artifacts_dir`, `max_parallel_pairs=5`, `debug_*`, timeouts) but `created_by="proposal_from_candidate_finding_creator"`. Justification for the distinct default: `created_by` lands in `Proposal.author` via `_convert_proposal` ([api.py:331](../../src/spotlights_engine/proposal_from_finding_creator/api.py#L331)) and is only ever *rendered* ([writer.py:480-481](../../src/spotlights_engine/results_renderer/writer.py#L480-L481)) or copied to `EvolveProposal.agent` ([prep_evolve/extract.py:227,238](../../src/spotlights_engine/prep_evolve/extract.py#L227)) — no consumer compares it to a literal, so a new value is safe. `agent_schema`/`prompts`/`validation` pin whatever value the config carries, so nothing else changes. **Also add a `proposal_from_candidate_finding: <NewConfig> | None = None` slot to `SpotlightsManagerConfig` ([spotlights_manager/api.py:36-58](../../src/spotlights_engine/spotlights_manager/api.py#L36-L58))** and a `_build_proposal_from_candidate_finding_config` mirroring [orchestrator.py:253-266](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L253-L266); otherwise candidate mode has no way to receive infra knobs. **Do NOT add a new key to `build_config_fingerprint`** ([persistence.py:301-333](../../src/spotlights_engine/spotlights_manager/persistence.py#L301-L333)) unconditionally — it hashes an `or <Default>()` for every step, so an unconditional new key changes *every* existing run dir's config fingerprint and breaks all module-mode resumes. Emit the new hash only in candidate mode (same effective-key rule as D5).
- **Reuse.** `claude_exec` (`run_pair`/`default_run_pair`, `ensure_claude_available`, `PairRunResult`), `errors`, `mint_proposal_ids`, and `CliUsage` plumbing are imported from the existing package/utilities (all pure or side-effect-free w.r.t. the old mode). The prompt/schema/validation-bound orchestration (`_run_one_pair`, `_run_async`, `_convert_proposal`, `_rebuild_candidate`) and the path/debug helpers (`_validate_setup`, `_last_messages_dir`, `_persist_pair_debug`) are copied, per the notes above.
- **Export surface.** Add the new package's public names to `pipeline.py`'s re-export block and `__all__`, mirroring how `proposal_from_finding_creator` is surfaced today.

## Manager changes (`spotlights_manager/orchestrator.py`)

- **`_do_step3` ([orchestrator.py:735-771](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L735-L771)) becomes a dispatch:**
  - `module` mode → the **exact** current body (build `ModuleDeepResearchInput`, `_build_deep_research_options(cfg, repo, module_paths.deep_research_last_message_path)`, call `research_module`). Unchanged.
  - `candidate` mode → build `CandidateDeepResearchInput`, build base codex options without a concrete single-file `output_last_message`, pass the new `deep_research_last_message_dir` separately (D6), and call the new package's `research_candidates_with_telemetry`. Returns a tuple of the **same arity** — `(ModuleDeepResearchOutput, duration, usages)`, matching `_do_step3`'s declared `tuple[ModuleDeepResearchOutput, float, list[Any]]` at [orchestrator.py:744](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L744) — so the step-3 persist call sites (`P.write_deep_research`, `P.write_deep_research_search_log`) stay shared. **Note the element type of `usages` differs by mode:** module mode returns `list[CliUsage]` (which carries only `cli` + `usage` — no candidate id, no duration, [costing/usage.py:39-45](../../src/spotlights_engine/costing/usage.py#L39-L45)); candidate mode returns a richer parallel structure carrying `(candidate_id, cli, usage, per_run_duration)` so the usage-writing block (below) can attribute per candidate. This is *not* a shared-code violation of the persist calls — it is exactly why the usage-writing block branches on mode (see Usage writes below); the two dispatch branches of `_do_step3` build the two shapes.
- **`_do_step4` ([orchestrator.py:791-830](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L791-L830)) becomes a dispatch:** `module` → `create_proposals_with_telemetry` with `_build_proposal_from_finding_config` (unchanged); `candidate` → the new package's equivalent with `_build_proposal_from_candidate_finding_config`. Both return the same 4-tuple `(output, total_duration_s, per_pair_durations_s, usages_by_pair)` matching the declared return type at [orchestrator.py:800-806](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L800-L806), so the shared persist/checkpoint tail is untouched.
- **No new parameter threading.** Both `_do_step3` and `_do_step4` already receive `mgr_input: SpotlightsManagerInput` ([orchestrator.py:739](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L739), [:795](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L795)), so `mgr_input.deep_research_mode` is readable inside each dispatch with **zero changes at the call sites** ([orchestrator.py:1154-1162](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1154-L1162), [1291-1300](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1291-L1300)) — keep those byte-identical.
- **Keep the module-level alias seam.** `research_module = research_module_with_telemetry` at [orchestrator.py:106](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L106) exists so tests can monkeypatch `orch.research_module` (`tests/unit/spotlights_manager/test_orchestrator_status.py`, `test_accumulated_duration_resume.py` both do). **Do not inline it into the dispatch** — that would break those tests. Add a sibling `research_candidates = research_candidates_with_telemetry` alias and dispatch through the module-global names in both branches, preserving the existing `hasattr(result, "output") and hasattr(result, "usages")` duck-typing at [orchestrator.py:769-771](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L769-L771) so patched fakes returning a bare output still work.
- **Shared, unchanged after dispatch.** The step-3/step-4 persist calls (`P.write_deep_research`, `P.write_deep_research_search_log`, `P.write_proposal_from_finding`), checkpoint transitions, `_synthetic_step4_output_for_zero_findings`, and the `_final_status` logic are mode-agnostic and stay as-is.
- **Pair-count log ([orchestrator.py:1284](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1284), currently `len(candidates.candidates) * len(research_output.findings)`).** Guard the `|C|×|F|` computation behind the mode: module mode keeps the exact line; candidate mode logs `Σ|F_c|`. (Alternatively compute the count inside each `_do_step*` and return it; either way the module-mode string is unchanged.)
- **Usage writes ([orchestrator.py:1214-1229](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1214-L1229)).** Module mode's tuple builder (`deep-research:<cli>`, `dr_duration` fallback) stays. Candidate mode builds `<candidate_id>:<cli>` tuples with per-run durations from D8. Both flow through the same `_write_cli_usage_records`, whose signature is already generic — `usages: list[tuple[str, str, Any, float | None]]` ([orchestrator.py:374-398](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L374-L398)) — so **no signature change is needed**. Reuse `step="module_deep_research"` / `role="deep_research"` (D4: the `UsageStep` union is closed).
- **Fingerprint ([orchestrator.py:1715-1722](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L1715-L1722)).** Pass `deep_research_mode`, both cap/hotspot knobs, and `enable_claude_search` to `build_input_fingerprint`; that helper emits only the keys effective for the selected mode while keeping shared knobs such as `enable_claude_search` in both modes (D5). New keyword params on `build_input_fingerprint` must have defaults that reproduce today's dict exactly when `deep_research_mode="module"`, since [orchestrator.py:465-470](../../src/spotlights_engine/spotlights_manager/orchestrator.py#L465-L470) compares the freshly built dict to the on-disk one with `!=`.

## Persistence (`spotlights_manager/persistence.py`)

- **Sidecars: no new files.** Candidate mode reuses `module_deep_research.json`
  and `proposal_from_finding_creator.json` (D4). `read_module_state`
  ([persistence.py:400-500](../../src/spotlights_engine/spotlights_manager/persistence.py#L400-L500)) validates both against the updated (additive) schemas with no code change.
- **One additive property + one clear tweak (D6).** Add
  `deep_research_last_message_dir` returning `module_deep_research.last_messages/`
  alongside the existing single-file `deep_research_last_message_path` ([persistence.py:189-191](../../src/spotlights_engine/spotlights_manager/persistence.py#L189-L191), kept for
  module mode); mirror the existing `proposal_from_finding_last_message_dir` property ([persistence.py:201-202](../../src/spotlights_engine/spotlights_manager/persistence.py#L201-L202)) exactly. Extend `clear_deep_research_artifacts`
  ([persistence.py:656-668](../../src/spotlights_engine/spotlights_manager/persistence.py#L656-L668)) to also `shutil.rmtree` the directory if present, so a `--redo` of a candidate-mode step 3 doesn't leak prior messages. `shutil` is already imported and `clear_proposal_from_finding_artifacts` ([persistence.py:669-676](../../src/spotlights_engine/spotlights_manager/persistence.py#L669-L676)) is the exact pattern to copy. The existing `unlink()` of the single file is retained (harmless no-op when only the dir exists), so module-mode clearing is unchanged.
- **`build_input_fingerprint` ([persistence.py:282-298](../../src/spotlights_engine/spotlights_manager/persistence.py#L282-L298))** gains keyword-only params for `deep_research_mode`, `max_findings_per_candidate`, and the module-only knobs, then emits the mode-specific effective key set described in D5. Keep `enable_claude_search` in both fingerprints because both modes consume it. Concretely: in `module` mode return **exactly today's six keys with today's values** (no `deep_research_mode` key at all — adding one unconditionally would break every existing run dir); in `candidate` mode return `repo_path`, `context_hash`, `continue_on_module_failure`, `enable_claude_search`, plus `deep_research_mode: "candidate"` and `max_findings_per_candidate`, and **omit** `max_findings_per_module` / `include_candidate_hotspots` (module-only knobs). Mode switching then always mismatches (the key sets differ), which is the desired `ResumeMismatchError`.
- **`SCHEMA_VERSION`** (currently `4`, [persistence.py:60](../../src/spotlights_engine/spotlights_manager/persistence.py#L60)). *Do not bump*: with the
  mode-specific fingerprint rule (D5) and additive optional schema fields, an existing
  version-4 module-mode run dir still validates and still fingerprint-matches, so
  it resumes unchanged. `read_module_state` ([persistence.py:400-500](../../src/spotlights_engine/spotlights_manager/persistence.py#L400-L500)) re-validates sidecars through the same models, and additive optional fields are backward-compatible in both directions (old JSON lacking the keys validates; new JSON with `null` validates).

## Renderer / CLI

- **`results_renderer/aggregator.py`** — `_count_relevant_findings` ([aggregator.py:203-215](../../src/spotlights_engine/results_renderer/aggregator.py#L203-L215)) is unchanged and works for both modes. Optional enhancement: group findings under their candidate when `candidate_id` is present.
- **`results_renderer/writer.py`** — verified: the writer currently renders **none** of `mechanism`/`required_changes`/`expected_effect`/`evaluation_metric` (it renders `author`, `title`, `description`, `rationale` at [writer.py:457-505](../../src/spotlights_engine/results_renderer/writer.py#L457-L505)). So this is a pure addition: render the four fields only when non-`None`, which leaves module-mode output byte-identical. Optional for a first cut — nothing breaks if it's skipped, the fields just don't surface in Markdown.
- **`module_deep_research/search_log.py`** — the renderer is shared. In candidate mode one module's log now contains many surveys; optionally group/label by `candidate_id` and key `_failed_agents` by `(candidate, agent)`. Module-mode output (one survey) is unchanged either way. If this file is edited, keep the module-mode rendering byte-identical (label section only appears when `candidate_id` is set).
- **`cli.py`** — add `--deep-research-mode {module,candidate}` (default `module`) and `--max-findings-per-candidate` (default `None`), consumed in `_build_input` ([cli.py:342-357](../../src/spotlights_engine/cli.py#L342-L357)) with the same `if … is not None` guard style used for `--max-findings-per-module`, so an unset flag never changes `SpotlightsManagerInput` defaults. Leave `--max-findings-per-module` and `--no-candidate-hotspots` exactly as-is (they still drive module mode). The debug/pair flags ([cli.py:201-212](../../src/spotlights_engine/cli.py#L201-L212)) currently build only a `ProposalFromFindingConfig` in `_build_config` ([cli.py:360-396](../../src/spotlights_engine/cli.py#L360-L396)) — in candidate mode they must instead populate the new `proposal_from_candidate_finding` slot, or the flags silently do nothing. Also update the `[4/5] proposal_from_finding_creator …` progress label ([cli.py:451](../../src/spotlights_engine/cli.py#L451)) or accept that candidate mode logs the module-mode step name (acceptable — the step *name* is fixed by D4 anyway).

## Downstream consumers

- **`prep_evolve`** — out of scope for this change, but note the shape: `ProposalRef` ([prep_evolve/spec.py:83-89](../../src/spotlights_engine/prep_evolve/spec.py#L83-L89)) currently carries `origin`/`agent`/`title`/`detailed_description`/`finding_id`/`rationale` and drops the four structured fields. Since they are `None` in module mode there is no regression either way; extend `_proposals` ([prep_evolve/extract.py:215-240](../../src/spotlights_engine/prep_evolve/extract.py#L215-L240)) only if evolution bundles should consume them, and note `_SpecModel` field additions are a spec-version concern (`spec_version = "1"`).
- **`module_knowledge/adapters.py`** — not on the manager's step-3/4 execution path, but **not verified unaffected**: the current adapter still references legacy flat `Candidate` fields (`candidate.symbol`, `candidate.kind`, `candidate.state`, `candidate.file`, etc.) and legacy proposal lists (`candidate.deep_research_proposals`, `candidate.agent_proposals`). If module knowledge is in scope for this implementation, update it to use `primary_file`/`primary_span`, `proposals_from`, unified `Proposal`, and include `Finding.candidate_id` metadata when present while handling `None` in module mode.
- **`signal_pipeline`** — verified independent: it emits `findings=[]` and populates the four `Proposal` fields itself, confirming they map cleanly; no change.
- **`original_schemas/`** — dead, un-imported duplicate; no edits.

## Tests

- **Regression (module mode unchanged) — the most important tests.** A golden test that runs a fixed fake-runner pipeline in `module` mode and asserts: (a) per the D3 decision, module-mode `module_deep_research.json` findings carry `"candidate_id": null` and nothing else changed (no test currently asserts byte identity of that file — only `tests/unit/spotlights_manager/test_persistence.py` and `tests/integration/spotlights_manager/test_end_to_end_subset.py` touch it, and both read structurally); (b) the input fingerprint for a module-mode run is **exactly** today's six-key dict, so a pre-change run dir resumes; (c) candidate-mode fingerprints omit the module-only knobs, still include `enable_claude_search`, and switching modes raises `ResumeMismatchError`; (d) module-mode step 3 still uses the single `module_deep_research.last_message.md` path; (e) the pair-count log line is unchanged; (f) `orch.research_module` monkeypatching still works (the alias survives).
- **Candidate mode, step 3** — per-candidate loop, per-candidate finding segment ids, `candidate_id` stamping, per-candidate last-message files, per-(candidate,runner) usage/duration attribution, and a two-candidate cross-attribution test (distinct fake payloads per candidate must not bleed across the shared options object).
- **Candidate mode, step 4** — grouping by `candidate_id`, empty-subset candidates yield no proposals, `will_invoke_claude` false when zero pairs survive, `debug_first_n_pairs` reports the grouped pair count, and the four new fields flow through `_convert_proposal`.
- **Schema** — `Finding.candidate_id` and `SearchQueryLog.candidate_id` round-trip; `DeepResearchProposal` new fields round-trip; existing `Finding(`/`Candidate` construction sites still pass with the field defaulting to `None`; a `find-<module_segment>-<candidate_counter>-NNNN` id both passes `Finding.finding_id`'s pattern and survives `parse_id` → `prefix_local_id` idempotently.
- **Manager dispatch** — `deep_research_mode` threads CLI → input → `_do_step3`/`_do_step4`; candidate mode reuses the sidecars and checkpoint states; switching mode on resume raises `ResumeMismatchError`.
- **CLI** — `--deep-research-mode` default is `module`; `--max-findings-per-candidate` binds only in candidate mode.
- **Renderer** — relevant-findings count unaffected; new proposal fields render only when present; search-log module-mode output unchanged.

## Decisions confirmed

These are settled. Implement them as written — do not re-open them.

1. **D3 serialization: accept the serialized `null`.** `Finding.candidate_id` and `SearchQueryLog.candidate_id` are plain additive optionals, and `module_deep_research.json` in module mode will contain `"candidate_id": null` on every finding. **Do not** add a mode-specific dump path, **do not** add `exclude_none=True`, and **do not** add a field serializer. Rationale: the writer (`P.write_deep_research`) is shared by both modes, so any omission trick would require forking a shared writer purely for cosmetics; the reader is the same pydantic model, so `null` round-trips identically to absent; and no test asserts byte identity of that file. Module-mode *runtime* behavior is unchanged, which is the invariant that matters.
2. **Fingerprint: mode-specific effective keys, no `SCHEMA_VERSION` bump.** `build_input_fingerprint` emits exactly today's six keys when `deep_research_mode == "module"` (so every existing v4 run dir still resumes) and a candidate-specific key set otherwise. All new params are keyword-only with module-mode-preserving defaults. Same rule for the new step-4 config hash in `build_config_fingerprint`: emit it only in candidate mode. Do **not** take the "always emit the new keys and bump `SCHEMA_VERSION`" route — it invalidates every existing run dir for zero functional gain.
3. **Dedicated `CandidateDeepResearchInput`.** A new model in `schemas/pipeline.py` with its own field list, not optional fields bolted onto `ModuleDeepResearchInput`. Keeps the module contract frozen and makes the module-only knobs (`max_findings_per_module`, `include_candidate_hotspots`) structurally absent from candidate mode rather than "present but ignored".
4. **Reuse boundary (as fixed in D1 and New step 4).** Import the pure/stateless helpers listed in D1; **copy** the timed fan-out (because `RunnerOutcome` has no duration field and it must not gain one), and **copy** the step-4 orchestration + path helpers that bind module globals. New packages import from old packages only — never the reverse — so no cycle is possible.
5. **Finding-id format: `find-<module_segment>-<candidate_counter>-NNNN`.** Verified against the real regexes and `parse_id`, not assumed: `find-src_utils-0003-0007` matches `Finding.finding_id`'s `^find-[A-Za-z0-9._-]+-\d{4}$`, and `parse_id` returns `("find", "src_utils-0003", 7)` because the middle group is greedy and the counter is right-anchored. Pass `segment=f"{module_segment}-{candidate_counter:04d}"` to the existing `merge_outcomes`; no id-helper changes.
6. **Per-candidate concurrency: default 2; `enable_claude_search` stays default-off.** Make it a keyword-only `max_parallel_candidates: int = 2` argument on `research_candidates_with_telemetry` — not a new config model and not a new `SpotlightsManagerConfig` slot, which would force a new `build_config_fingerprint` key (D7). Rationale: candidate mode multiplies step-3 sessions by the candidate count, so the default must be cost-conservative; 2 gives some wall-clock win over serial without an N-fold burst, and it composes with the manager's own `max_parallel_sessions` (default 1) which already gates module-level fan-out. `enable_claude_search` keeps its `False` default in both modes (it is a shared input knob and is fingerprinted in both). See the human-call item below for the budget posture this implies.
7. **Step-3 failure granularity: recoverable per-candidate `StepIssue`.** Per-candidate wrapper that appends a recoverable issue naming the candidate and continues the loop, built with `step="module_deep_research"` (the `PipelineStep` union is closed — D4). One bad candidate degrades the module to `DEGRADED`, never `FAILED`, and other candidates' findings survive.
8. **Search-log candidate grouping: deferred.** Ship candidate mode with the shared `search_log.py` renderer untouched. In candidate mode one module's log simply concatenates many surveys, which is readable but unlabeled. Revisit only after the mode is exercised on a real repo; if added later, gate every candidate label behind `candidate_id is not None` so module-mode output stays byte-identical.

### Still needs a human call

- **Cost/budget posture for candidate mode.** This is a product/spend decision, not a code one. Step 3 goes from `len(runners)` agent sessions per module to `len(candidates) × len(runners)`, and the merged finding set scales by roughly the same factor; step 4's pair count then shrinks to `Σ|F_c|` but off a larger finding base. With ~8-12 candidates per module this is close to an order of magnitude more spend per module than module mode. Decide (a) whether candidate mode is intended for whole-repo runs or only for narrow `--include` scopes, and (b) whether `max_findings_per_candidate` should default lower than `max_findings_per_module` does. The plan works at any setting; the numbers are a budget owner's choice.
- **Is `module_knowledge/adapters.py` in scope?** It is already broken against the current unified schemas (it references legacy flat `Candidate` fields and legacy proposal lists — see Downstream consumers), independently of this change. Candidate mode does not make it worse, but if module knowledge is expected to work after this lands, its repair needs to be scheduled as its own task rather than smuggled into this one.
