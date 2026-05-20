# `signal_discovery_flow.md` — Alignment Audit (Second Pass)

> **Status:** point-in-time audit, 2026-05-20. Quotes reflect source files at that date and may be stale. **Not a current reference** — see [`README.md`](README.md). Outstanding punch-list items at the bottom of this file are deferred, not applied.

*Re-audit of the rewritten flow doc against `proposal/discovery_engine_proposal.md`, `contracts/signal_interface_spec.md`, `plans/bundle_charters.md`, `plans/repo_scaffolding_plan.md`, and `README.md`. Supersedes the first-pass findings.*

## Summary

The rewrite resolves the substantive issues from the first audit: terminology now matches the canonical objects (`Anomaly`, `Candidate`, `Change`), Bundle A is correctly framed as deterministic, the invented `CodeExtractor` / `get_detailed_traces` APIs are gone, the proposal's `get_raw_trace` is wired in, both KV-cache framings appear as separate worked examples, Bundle D is a handoff layer, and Bundles B/E/F are visibly deferred. The remaining issues are minor — one genuine internal inconsistency (the doc claims it does not label per-step ownership but the pseudocode comments do exactly that), one open question that is partly answered elsewhere (`code_overview` access — see Bundle C charter), and a small slip in the Anomaly fixture for Finding II (`type` is not in the spec's enumerated examples). The doc is ready for human review.

---

## 1. Terminology — **Resolved**

Every prior-audit finding in this category lands clean.

- *Suspect → Anomaly:* the new doc uses `Anomaly` throughout (pseudocode `anomalies = bundle_a.process(...)`; demo tables `Anomaly (Bundle A)`; mermaid `AN[/"Anomaly[]"/]`). The word *suspect* does not appear.
- *Candidate schema reconciled:* `# → list[Candidate{locations, observation, evidence, significance}]` matches `discovery_engine_proposal.md` §4.1 verbatim. The invented `hypothesis` field is gone.
- *Signal layer surfaced:* the three Signal objects (`WorkloadProfile`, `TraceSummary`, `Anomaly`) are now produced in step 1 and consumed in step 2, matching `signal_interface_spec.md`'s "objects Bundle C consumes."
- *"Spotlight" / "finding":* "spotlight" is gone. "Finding" is used as the unit-of-output term in the demo headings ("Finding I", "Finding II"), consistent with the proposal §4.3 ("This is a finding.").

## 2. Pipeline shape — **Resolved**

- *Bundle A vs Bundle C split restored.* The pseudocode's first block is explicitly labeled `# ── Bundle A: deterministic signal extraction (no LLM)` and the comment `# No interpretation; heuristics + summarization only.` directly mirrors `signal_interface_spec.md` "No interpretation in the signal layer." Bundle C's LLM call consumes the structured outputs, not raw telemetry.
- *Bundle B explicitly deferred.* Pseudocode line `# retrieval=bundle_b.query     # technique-driven; deferred for MVP` and mermaid node `BB[/"Bundle B retrieval<br/>(deferred for MVP)"/]` rendered with the `deferred` style. Open question 3 frames it as "When does Bundle B's `query()` enter Bundle C's prompt?" — a deferral question, not a missing-design question.
- *Validation and archive scope-noted.* `# Bundle E (validation) and Bundle F (archive) close the loop. / # See proposal §4.3 stages 4–5. Out of scope for this flow doc.` This is the explicit MVP-slice label the prior audit asked for.

## 3. Data schemas — **Resolved**

- `Candidate{locations, observation, evidence, significance}` — exact match to proposal §4.1.
- `Change{candidate_ref, change_type, mechanism, expected_effect, required_changes, evaluation_metric}` — exact match to proposal §4.1. `evaluation_metric` is preserved.
- The `Change` *spec* is now distinct from the execution *result*: `change = llm_prompt("generate_change", ...)` produces the spec; `result = execution_backend.run(change)` produces `{file_edits, rationale, ...}`. The mermaid keeps these as two artifacts (`CHANGE` and `RES`). This is the split the prior audit asked for.
- Anomaly is referenced via `anomaly.evidence_pointer` for `get_raw_trace`, matching the contract's `evidence_pointer` field.

## 4. Module ownership — **Resolved (with a small slip in the doc's framing — see §Internal inconsistencies)**

- *Invented APIs gone.* `CodeExtractor` and `get_detailed_traces(region, window)` are absent. The doc uses the contract's `get_raw_trace(anomaly.evidence_pointer)` (mermaid edge `BC -.->|"get_raw_trace (on demand)"| BA`), matching `signal_interface_spec.md`'s raw-access method.
- *Cross-reference replaces per-step labels:* the preamble says "Per-step ownership lives in those docs; this file is the visual + sequential sketch only — read `bundle_charters.md` for who owns what." This is the first-audit recommendation, taken.
- *Codebase as Bundle C input:* `code=codebase, # Bundle C reads the repo directly` is consistent with `bundle_charters.md` Bundle C ("In: ... Human hints in a format Bundle C defines"), though codebase access itself is not explicitly chartered anywhere — see Open Q2 in the doc, which the rewrite correctly flags as still-open.

## 5. Demo story — **Resolved**

Both KV-cache framings are now present as parallel worked examples ("Finding I — transfer-latency / prefetch (proposal §4.3)" and "Finding II — eviction policy / ARC"). The closing paragraph — "The two findings are independent; the pipeline can produce either, both, or neither, depending on which `Anomaly` types Bundle A's heuristics emit and how Bundle C ranks them" — is exactly the disambiguation the prior audit requested.

## 6. MVP claims and open questions — **Resolved**

The three open questions in the rewrite are the genuinely-open ones:

1. *Candidate granularity* — the prior audit flagged this as the single open question that *is* still open. Kept.
2. *Code-access contract* — new and legitimate. Bundle C reads the codebase, but no contract surface owns code overview / extraction, and the rewrite is honest about this ("there is no canonical `CodeAccess` module in `repo_scaffolding_plan.md`").
3. *Bundle B wiring* — deferral question, correctly framed.

The previously-redundant open questions (suspect schema, stub API contracts, MVP-claim wording) are gone.

## 7. Other (first-audit) divergences — **Resolved**

- `code_overview` placeholder is gone; `codebase` is the named input and is flagged as needing a contract surface (Open Q2 above).
- *Coding agent vs Bundle D handoff:* mermaid renders `EXEC[["Execution backend<br/>(Claude Code in MVP;<br/>OpenEvolve / Shinka / Nous later)"]]` with the dedicated `stub` style, and the pseudocode comment reads "MVP wires Claude Code first; per Bundle D's charter, OpenEvolve / ShinkaEvolve / Nous are pluggable later." This is the abstraction-level fix the prior audit asked for.

---

## New issues introduced

### N1 — Anomaly `type` value `anomalous_distribution` is in the spec, but the demo's framing slightly drifts

> Flow doc, Finding II: `type=anomalous_distribution, components=[kv_cache_manager], magnitude="cache hit-rate degrades when reused contexts evicted under load"`

> `signal_interface_spec.md`: `magnitude: a quantitative measure where meaningful (e.g., "GPU idle for 12% of decode time")`

`anomalous_distribution` is on the spec's enumerated example list — fine. But the `magnitude` is a prose description ("cache hit-rate degrades when reused contexts evicted under load"), not a quantitative measure. Finding I's `magnitude="GPU idle 12% of decode time"` matches the spec's example pattern; Finding II's does not. Either tighten the Finding-II magnitude to a quantity (e.g., `"hit-rate drops from 0.74 → 0.41 above N concurrent agents"`) or move the prose part into `description`.

### N2 — Bundle C charter says `WorkloadProfile, TraceSummary, Anomaly` are its inputs; flow doc adds `code` (codebase) without flagging it as a contract gap on the Bundle C side

> `bundle_charters.md` Bundle C: "*In:* `WorkloadProfile`, `TraceSummary`, `Anomaly` from Bundle A. Query interface from Bundle B. Human hints in a format Bundle C defines."

> Flow doc: `code=codebase, # Bundle C reads the repo directly`

The flow doc's Open Q2 acknowledges code access has no contract surface in `repo_scaffolding_plan.md`, but does not note that this is also a charter gap — Bundle C's charter does not list "codebase" among its inputs. Worth either widening Open Q2 to "no contract in `bundle_charters.md` either" or asking for the charter's `In:` list to add codebase access.

### N3 — `repo_scaffolding_plan.md` puts Bundle C in `discovery-engine`'s `candidates/` subpackage; flow doc's pseudocode uses `bundle_a.process(...)` / `bundle_b.query` / `llm_prompt(...)` with no reference to the actual module layout

This is cosmetic — the doc says it is a "visual + sequential sketch only" — but a reader cross-checking against `repo_scaffolding_plan.md` will not find a `bundle_a` module. Consider either qualifying the pseudocode (e.g., `discovery_observability.process(...)`) or noting once that bundle names are stand-ins for the scaffolded subpackages (`discovery_observability`, `discovery_knowledge`, `discovery_engine.candidates`, `discovery_engine.changes`).

---

## Internal inconsistencies in the rewrite

### I1 — Doc claims to omit per-step ownership but pseudocode labels every step by Bundle

> Preamble: "Per-step ownership lives in those docs; this file is the visual + sequential sketch only — read `bundle_charters.md` for who owns what."

> Pseudocode comments: `# ── Bundle A: deterministic signal extraction (no LLM)`, `# ── Bundle C: LLM reasons over structured signals + code`, `# ── Bundle D: per-candidate Change spec + execution handoff`. Mermaid nodes are labeled `Bundle A`, `Bundle C`, `Bundle D`. Demo tables column-label by Bundle.

In practice the rewrite *does* label per-step ownership (and that's a good thing — it's what the first audit asked for). The preamble disclaimer reads as left-over from an earlier draft. Either drop the disclaimer, or reword to "*Detailed* ownership and contract surfaces live in those docs; this file labels each step at bundle granularity only."

### I2 — Pseudocode says "Two LLM calls per run" but the loop runs `llm_prompt("generate_change", ...)` per candidate

> "Two LLM calls per run (Bundle C, Bundle D), plus one execution-backend call per candidate."

> Pseudocode: `for c in candidates: change = llm_prompt("generate_change", candidate=c)`

There is one Bundle C call per run, but Bundle D's `generate_change` is called once per candidate inside the loop — so it is N+1 LLM calls, not 2. The trailing sentence either needs to be "One LLM call for Bundle C, one per candidate for Bundle D, plus one execution-backend call per candidate," or the Bundle D prompt needs to be reframed as a batched call over `candidates` (less natural — the per-candidate framing is right; fix the sentence).

### I3 — Mermaid edge `BC -.->|"get_raw_trace (on demand)"| BA` points back to Bundle A, but Bundle A in the diagram has already produced and emitted `WP / TS / AN`

This is a minor diagram-flow nit: the dashed back-edge implies a request channel that Bundle A still services after its `process()` returns. The contract (`signal_interface_spec.md`) does support this (raw access is exposed in addition to the three structured outputs), so the semantics are correct — but the diagram could read as a circular dependency. Worth a one-line note in *Visual vocabulary*: "dashed back-edge — on-demand call into a bundle's raw-access surface."

### I4 — Visual-vocabulary section lists "Dashed gray box — deferred for MVP" but the dashed Bundle B node uses light-purple (`fdf4ff` / `7e22ce`) styling? No — actually fine

Re-reading: `BB` is classed `deferred` (`fill:#f9fafb,stroke:#9ca3af,stroke-dasharray:4 2`), which is the "dashed gray" the legend describes. The `stub` class (purple) applies only to `EXEC`. Resolved on a closer reading; flagging only because the `stub` and `deferred` colors are close enough that a casual reader could conflate them. Consider listing `EXEC` as "purple subroutine — execution backend" (already done in vocab) and emphasizing the *dashed border* is the deferred marker, not the color alone.

---

## Catches the prior audit missed (re-read of the new doc end-to-end)

- *Provenance flow.* The contract requires `evidence_pointer` for every `Anomaly` and the rewrite uses it (`get_raw_trace(anomaly.evidence_pointer)`). Good.
- *Two-input framing.* The doc has Telemetry and Codebase as inputs (Literature/Bundle B is deferred). Proposal §3 also names "Repository activity" (issues, PRs, commits) and "Human hints" as primary inputs. Neither appears in the flow. This is reasonable for an MVP slice but the flow doc is silent on why — worth one line: "MVP slice: telemetry + codebase only; human hints, repo activity, and literature retrieval enter via Bundle B in later stages."
- *`software_version` and `workload_id` provenance.* `WorkloadProfile` carries `software_version` (commit SHA). The flow doc never uses it, which is fine, but if Bundle C is going to read the codebase, the *commit* it reads should be pinned to `WorkloadProfile.software_version` — otherwise you get drift between the code Bundle C reasons over and the code that produced the telemetry. This is a real issue for reproducibility and is not currently captured anywhere. Recommend adding it to Open Q2 (code-access contract).
- *Stage 0 vs Stage 1.* `signal_interface_spec.md` says the interface is "locked at the end of Stage 0 calibration." The flow doc never names a stage. Since the rewrite uses the locked schemas verbatim, this is implicit Stage-1 thinking — but a reader landing here cold might assume the flow is also Stage-0. One sentence near the top ("This is the Stage 1 MVP flow") would prevent that.
- *`Candidate.evidence` field type is unspecified in the proposal* (`evidence: traces, profiles, comparisons grounding the observation`). Finding I sets it to `evidence=traces` and Finding II to `evidence="hit-rate vs reuse-pattern correlation"` — different shapes. This isn't a flow-doc bug (the proposal is the source of looseness), but it is worth noting since the flow doc is the first place where the field is used in two different shapes side-by-side. Either the proposal pins the type or the flow doc cites the looseness explicitly.

---

## Questions to resolve

1. Tighten Finding II's `Anomaly.magnitude` to a quantitative measure (per `signal_interface_spec.md`), or move the prose into `description`. (N1)
2. Decide whether Bundle C's *codebase* input should be added to its charter `In:` list, or kept implicit pending a code-access contract. (N2 / Open Q2 in the doc)
3. Fix the trailing "Two LLM calls per run" sentence to reflect that Bundle D fires once per candidate. (I2)
4. Drop the preamble disclaimer about per-step ownership, or reword it to match what the doc actually does (label at bundle granularity). (I1)
5. Add one line near the top stating this is the Stage 1 MVP flow, and one line stating the deliberately-narrowed input set (telemetry + codebase only; human hints / repo activity / literature deferred via Bundle B).
6. Decide where commit-pinning lives: `WorkloadProfile.software_version` should pin the codebase Bundle C reasons over — folding this into Open Q2 captures the reproducibility concern.
7. Optional: clarify in *Visual vocabulary* that the dashed back-edge for `get_raw_trace` is a request channel, not a circular dependency. (I3)
