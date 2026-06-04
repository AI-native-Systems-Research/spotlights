# Bundle Charters

*One page per bundle. Generic — owners customize when assigned. Shared structure for comparability.*

---

## How to read these

Each charter has the same eight sections in the same order. This is deliberate: the charters should feel like instances of the same thing, not six different documents. When something changes in one charter's structure, change it in all six.

A charter is a **contract between the bundle owner and the rest of the project**. Once approved, scope decisions inside the bundle are the owner's call. Decisions that cross the contract surface to the weekly architecture review.

---

## Bundle A — Observability

**Repo:** `discovery-observability`

**Mission.** Make the subject system's runtime behavior legible to the rest of the Discovery Engine. Collect telemetry, summarize it into structured signals, surface anomalies. Bundle A is the eyes of the system.

**In scope.**
- Telemetry collection under representative workloads (PyTorch Profiler, NVIDIA Nsight, OpenTelemetry).
- Trace summarization, bottleneck heuristics, workload characterization, cross-run comparison.
- Anomaly detection: stall detection, utilization underuse, regression-vs-prior, and other detectors as needed.
- Producing `WorkloadProfile`, `TraceSummary`, and `Anomaly` objects per the Signal interface spec.

**Out of scope.**
- Deciding what a signal *means*. Interpretation is Bundle C's job.
- Ranking signals by importance. Bundle A reports what it saw, evenhandedly.
- Recommending changes. That is Bundle D.

**Contracts.**
- *Out:* `WorkloadProfile`, `TraceSummary`, `Anomaly` per `signal_interface_spec.md`. These are the public API.
- *In:* Access to the subject system (llm-d) and the compute to run it under load.

**Success criteria for Stage 1.**
- Can produce a `TraceSummary` and at least one `Anomaly` from a real run of the subject system under a defined workload.
- The output schema validates against the Signal interface spec.
- One end-to-end demo: workload runs, traces captured, summary and anomalies produced, consumed by Bundle C in the integration test.

**Dependencies on other bundles.**
- None for development. Bundle A can build to the spec without waiting for anyone.
- At integration time: Bundle F runs Bundle A's collectors as part of the loop.

**Open questions the owner resolves.**
- Push vs pull delivery to consumers.
- Persistence and freshness policy for raw traces.
- Which anomaly detectors ship in v1.
- Sampling overhead acceptable in production-shaped runs.

---

## Bundle B — Knowledge and Retrieval

**Repo:** `discovery-knowledge`

**Mission.** Make recent literature, informal sources, and the team's experimental history accessible to the Discovery Engine through a single retrieval interface. Bundle B is the memory of the system.

**In scope.**
- Ingestion of recent inference-serving papers, blog posts, newsletters, social-media threads, GitHub issue discussions.
- The experimental archive: past candidates, the changes tried for each, and outcomes.
- A retrieval interface supporting both signal-driven and technique-driven queries.
- The LLM Wiki experiment as the storage substrate.

**Out of scope.**
- Reasoning over retrieved content. Retrieval returns ranked passages; reasoning is Bundle C and D.
- Choosing which papers matter. The corpus is broad by default; relevance is the consumer's call.
- Permanent record-keeping for governance or compliance. The archive is for discovery, not audit.

**Contracts.**
- *Out:* `query(query: str, mode: Literal["signal_driven", "technique_driven"]) -> list[Result]`. `Result` includes the passage, the source, and provenance.
- *In:* Write access from Bundle F (or Bundle E) to record archive entries.

**Success criteria for Stage 1.**
- Can ingest at least 20 recent inference-serving papers and 10 informal sources.
- Both query modes return ranked results with provenance.
- The archive records at least one full candidate-change-outcome cycle from the Stage 1 demo.

**Dependencies on other bundles.**
- None for development. Bundle B can build to the interface without waiting.
- At integration time: Bundle C calls Bundle B's query; Bundle F writes archive entries.

**Open questions the owner resolves.**
- Specific embedding model and vector store (or whether to skip embeddings entirely in v1).
- Refresh cadence for the literature index.
- Curation policy for informal sources (signal-to-noise filtering).
- Whether archive entries are append-only or revisable.

---

## Bundle C — Candidate Generation

**Lives in:** `discovery-engine`, subpackage `candidates`

**Mission.** Turn signals and human hints into candidate objects — regions of suspicion with evidence. This is the project's defensible contribution. Bundle C is where discovery happens.

**In scope.**
- The candidate-generation prompt(s) and structured-output contracts.
- Both discovery modes: signal-driven (from observed signals to candidates) and technique-driven (from a paper or technique to places it might apply).
- Integration with Bundle A (consuming signals) and Bundle B (querying knowledge and archive).
- Ranking candidates by expected impact, when ranking is meaningful.

**Out of scope.**
- Proposing fixes. Candidates are findings, not solutions. Changes are Bundle D.
- Producing telemetry. Bundle A's job.
- Validating whether a candidate was right. Outcomes come from Bundle E.

**Contracts.**
- *Out:* `Candidate` objects per `discovery_engine_proposal.md` §4.1.
- *In:* `WorkloadProfile`, `TraceSummary`, `Anomaly` from Bundle A. Query interface from Bundle B. Human hints in a format Bundle C defines.

**Success criteria for Stage 1.**
- Can produce structured `Candidate` objects from real signals on the subject system.
- Both discovery modes demonstrably work (signal-driven and technique-driven).
- At least one Stage 1 candidate is non-obvious — i.e., not something the calibration baselines surfaced.

**Dependencies on other bundles.**
- Stubs from Bundle A and Bundle B for early development. Real outputs by mid-Stage 1.

**Open questions the owner resolves.**
- Which model (and at what cost) generates candidates.
- How human hints are ingested and combined with automated signals.
- How candidate ranking works, if at all in v1.
- How the two discovery modes are composed (sequentially, in parallel, on demand).

---

## Bundle D — Change Generation and Execution Handoff

**Lives in:** `discovery-engine`, subpackage `changes`

**Mission.** Turn candidates into concrete proposed changes, then hand them off to existing execution machinery (evolutionary search, coding agents). Bundle D is the bridge from "what" to "how."

**In scope.**
- Generating `Change` objects from candidates: change type, mechanism, expected effect, required code surface, evaluation metric.
- Integration with execution backends: OpenEvolve, ShinkaEvolve, Claude Code, possibly Nous if integration becomes appropriate.
- Returning execution results to Bundle E for validation and Bundle F for archiving.

**Out of scope.**
- Discovering candidates. That is Bundle C.
- Actually evolving or refining code. That is the execution backends' job; Bundle D is the handoff layer.
- Deciding whether a change worked. Bundle E.

**Contracts.**
- *Out:* `Change` objects per `discovery_engine_proposal.md` §4.1. Execution results in a format Bundle E and F can consume.
- *In:* `Candidate` objects from Bundle C.

**Success criteria for Stage 1.**
- Can produce structured `Change` objects from candidates.
- Can hand off at least one change to an execution backend and recover the result.
- The Stage 1 demo includes a candidate → change → execution → result trace.

**Dependencies on other bundles.**
- Bundle C's `Candidate` schema. Stubs early, real candidates by mid-Stage 1.
- Bundle E to consume results.

**Open questions the owner resolves.**
- Which execution backend(s) to support in v1.
- How tightly to couple to specific backends vs. abstracting over them.
- How to handle backends that take hours to complete (async, polling, callbacks).

---

## Bundle E — Validation

**Lives in:** `discovery-engine`, subpackage `validation`

**Mission.** Confirm that changes flowing through the loop preserve correctness, improve the targeted metric, and don't regress on others. Bundle E is the gate before the archive.

**In scope.**
- Functional correctness checks for proposed changes.
- Performance and cost regression measurement.
- Robustness checks under adversarial workloads, where applicable.
- Intent-alignment monitoring (did the change do what it was supposed to do).
- Coordination with the validation position paper's framework.

**Out of scope.**
- Proposing changes. Bundle D.
- Generating workloads. Bundle A provides those.
- Long-running governance, audit trails for compliance. Discovery-focused validation only.

**Contracts.**
- *Out:* Validation results: a structured pass/fail/conditional verdict plus measurements, consumed by Bundle F for archiving.
- *In:* Execution results from Bundle D. Workloads and benchmarks from Bundle A. The original `Change` and `Candidate` for context.

**Success criteria for Stage 1.**
- Can validate at least one change end-to-end: correctness check, performance comparison, archived result.
- Validation framework is consistent with the companion validation position paper.
- Stage 1 demo includes a validated change with measured improvement.

**Dependencies on other bundles.**
- Bundle D's execution results. Bundle A's workloads.
- The validation position paper as the methodological anchor.

**Open questions the owner resolves.**
- Co-leadership question (if any) with the validation position paper authors.
- Which correctness suites are required vs. nice-to-have in v1.
- How to handle changes that improve one metric and regress another.

---

## Bundle F — Integration, Evaluation, and Demo Path

**Repo:** `discovery-engine` (primary owner of the spine)

**Mission.** Make the loop close. Bundle F owns the orchestration that turns the other five bundles into a working system, runs the calibration baselines, tracks success metrics, and delivers the Stage 1 demo.

**In scope.**
- End-to-end orchestration: signals flow to candidates flow to changes flow to validation flow to archive.
- Stage 0 calibration baselines (long-context architectural review, autonomous coding agent with profiling). The oh-my-review tooling fits as the static-only baseline.
- The success-metrics dashboard.
- The Stage 1 demo and the supporting integration tests.
- Schema stewardship: when contracts need to change, Bundle F arbitrates.

**Out of scope.**
- Doing other bundles' work. Bundle F integrates; it does not implement signals or candidates or changes.
- Strategic decisions about scope. Those escalate to Udi.

**Contracts.**
- *Out:* A working closed loop. The Stage 1 demo. The success-metrics dashboard.
- *In:* Functioning bundles from A through E.

**Success criteria for Stage 1.**
- Stage 0 calibration baselines have run, with results recorded.
- End-to-end loop runs on at least one real candidate from Bundle C, producing a measured outcome.
- Demo is rehearsable and documented.
- Success metrics from §5.3 of the proposal are tracked, with values for at least the first metric.

**Dependencies on other bundles.**
- All of them. Bundle F is the most cross-cutting role.

**Open questions the owner resolves.**
- Demo audience and framing.
- How tightly to schedule integration milestones with the other bundles.
- How to handle contract changes after Stage 0 lock (the spec process).

**Special authority.** Bundle F can push back on other bundle owners when contracts slip or when integration friction becomes unsustainable. Disputes that can't be resolved between owners escalate to Udi.

---

## Charter conventions (for all owners)

- **One page per bundle.** If a charter grows past one page, it's doing the implementation's job.
- **Update the charter when scope changes.** A charter that hasn't been touched in three months and a bundle that's wandered are a bad combination.
- **Open questions are decisions deferred, not problems hidden.** They live in the charter visibly until resolved.
- **The owner is named in their charter** once assigned. Until assigned, the field reads `<unassigned>`.



## Bundles and ownership

| Bundle | Lead | Members | Notes |
|---|---|---|---|
| A — Observability | Evgeny.Y | — | Idan/Y.B will likley be involved in early stages |
| B — Knowledge & code structure | Foad | Michael |  |
| C/D — Signal-driven (telemetry → change) | Idan | Yevgeny.B | |
| C/D — Technique-driven (paper → change) | Ophir | Roi | |
| E — Validation | Paula | Orit, Michael | |
| F — Integration | Inbar | — | Owns the engine repo, integration tests, demo path. Arbitrates schema and contract changes. |
| H — Execution (Evolve) | Joint, ad-hoc from C/D leads | — | Each path owns running its own changes initially. First mover (Idan or Ophir) builds shared scaffolding inside the spine repo; second mover uses it. Inbar reviews placement. |

Michael is part-time (3 days/week). Split between Knowledge and Validation as needed (coordinate with Inbar)