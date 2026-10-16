# Discovery Engine — System Design

## Overview

The Discovery Engine is an automated system that observes a subject system's runtime behavior, identifies optimization opportunities, proposes concrete changes, executes them through evolution/coding backends, and validates the results. It operates as a closed loop: signals lead to candidates, candidates lead to changes, changes are executed and validated, and outcomes feed back into the knowledge base for future discovery.

The subject system for Stage 1 is **vLLM** (an LLM inference-serving system).

---

## Architecture

The system is composed of six core components, each with well-defined interfaces:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Orchestration                                │
│           (closes the loop, runs calibration baselines)             │
└────┬────────────┬────────────┬────────────┬────────────┬────────────┘
     │            │            │            │            │
     ▼            ▼            ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│Observa-  │ │Knowledge │ │Candidate │ │ Change   │ │Validation│
│bility    │→│& Retrieval│→│Generation│→│Generation│→│          │
└──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘
     │                                                    │
     └──────────────────── feedback loop ─────────────────┘
```

---

## Components

### Observability

**Purpose.** Make the subject system's runtime behavior legible to the rest of the engine. Collect telemetry, summarize it into structured signals, and surface anomalies.

**Responsibilities.**
- Telemetry collection under representative workloads (PyTorch Profiler, NVIDIA Nsight, OpenTelemetry).
- Trace summarization, bottleneck heuristics, workload characterization, cross-run comparison.
- Anomaly detection: stall detection, utilization underuse, regression-vs-prior.
- Producing `WorkloadProfile`, `TraceSummary`, and `Anomaly` objects per the Signal interface spec.

**Interfaces.**
- *Produces:* `WorkloadProfile`, `TraceSummary`, `Anomaly` (per `signal_interface_spec.md`).
- *Requires:* Access to the subject system and compute to run it under load.

**Boundary.** Reports observations only. Does not interpret signals, rank them by importance, or recommend changes.

---

### Knowledge and Retrieval

**Purpose.** Make recent literature, informal sources, and experimental history accessible through a single retrieval interface. This is the system's memory.

**Responsibilities.**
- Ingestion of inference-serving papers, blog posts, newsletters, social-media threads, GitHub issue discussions.
- The experimental archive: past candidates, changes tried, and outcomes.
- A retrieval interface supporting both signal-driven and technique-driven queries.
- LLM Wiki experiment as the storage substrate.

**Interfaces.**
- *Produces:* `query(query: str, mode: Literal["signal_driven", "technique_driven"]) -> list[Result]` where `Result` includes passage, source, and provenance.
- *Requires:* Write access from the orchestration layer to record archive entries.

**Boundary.** Returns ranked passages with provenance. Does not reason over content or judge relevance — consumers make that call.

---

### Candidate Generation

**Purpose.** Turn signals and human hints into candidate objects — regions of suspicion with evidence. This is where discovery happens.

**Responsibilities.**
- Candidate-generation prompts and structured-output contracts.
- Two discovery modes:
  - **Signal-driven:** from observed signals (Observability output) to candidates.
  - **Technique-driven:** from a paper or technique (Knowledge output) to places it might apply.
- Integration with Observability (consuming signals) and Knowledge (querying knowledge/archive).
- Ranking candidates by expected impact.

**Interfaces.**
- *Produces:* `Candidate` objects per `discovery_engine_proposal.md` §4.1.
- *Requires:* `WorkloadProfile`, `TraceSummary`, `Anomaly` from Observability. Query interface from Knowledge. Human hints in a format Candidate Generation defines.

**Boundary.** Candidates are findings, not solutions. Does not propose fixes or validate outcomes.

---

### Change Generation and Execution Handoff

**Purpose.** Turn candidates into concrete proposed changes, then hand them off to execution backends (evolutionary search, coding agents).

**Responsibilities.**
- Generating `Change` objects from candidates: change type, mechanism, expected effect, required code surface, evaluation metric.
- Integration with execution backends: OpenEvolve, ShinkaEvolve, Claude Code.
- Returning execution results for validation and archiving.

**Interfaces.**
- *Produces:* `Change` objects per `discovery_engine_proposal.md` §4.1. Execution results consumable by Validation and Orchestration.
- *Requires:* `Candidate` objects from Candidate Generation.

**Boundary.** Hands off to execution machinery — does not evolve or refine code itself. Does not assess whether a change worked.

---

### Validation

**Purpose.** Confirm that changes preserve correctness, improve the targeted metric, and don't regress on others. The gate before the archive.

**Responsibilities.**
- Functional correctness checks for proposed changes.
- Performance and cost regression measurement.
- Robustness checks under adversarial workloads.
- Intent-alignment monitoring (did the change do what it was supposed to do).

**Interfaces.**
- *Produces:* Structured pass/fail/conditional verdict plus measurements.
- *Requires:* Execution results from Change Generation. Workloads and benchmarks from Observability. The original `Change` and `Candidate` for context.

**Boundary.** Validates outcomes only. Does not propose changes or generate workloads.

---

### Orchestration, Evaluation, and Demo Path

**Purpose.** Close the loop. Orchestrate the flow from signals through candidates, changes, validation, and archiving. Run calibration baselines and track success metrics.

**Responsibilities.**
- End-to-end orchestration: Observability → Candidate Generation → Change Generation → Validation → archive.
- Stage 0 calibration baselines (long-context architectural review, autonomous coding agent with profiling).
- Success-metrics dashboard.
- Schema stewardship: arbitrates contract changes between components.

**Interfaces.**
- *Produces:* A working closed loop. Calibration baselines. Success-metrics dashboard.
- *Requires:* Functioning components Observability through Validation.

**Authority.** Can push back on component owners when contracts slip or integration friction becomes unsustainable.

---

## Data Flow

1. **Observe:** Observability instruments the subject system under load → produces `WorkloadProfile`, `TraceSummary`, `Anomaly`.
2. **Discover:** Candidate Generation consumes signals from A and queries B → produces ranked `Candidate` objects.
3. **Propose:** Change Generation turns candidates into `Change` objects → hands off to execution backends.
4. **Execute:** Backends (OpenEvolve, ShinkaEvolve, Claude Code) run the change → return execution results.
5. **Validate:** Validation checks correctness, measures performance delta → produces verdict.
6. **Archive:** Results flow back to Knowledge's experimental archive, informing future discovery.

Orchestration orchestrates this entire cycle.

---

## Design Decisions

- **Strict separation of observation and interpretation.** Observability reports; Candidate Generation interprets. This keeps observability reusable and prevents coupling between telemetry collection and discovery logic.
- **Two discovery modes.** Signal-driven (bottom-up from telemetry) and technique-driven (top-down from literature) are independent entry points into candidate generation, composable by the orchestrator.
- **Execution is external.** The engine does not implement code evolution. It hands off to existing tools and recovers results. This avoids rebuilding what OpenEvolve/ShinkaEvolve already do.
- **Validation as a gate, not a suggestion.** Changes must pass validation before entering the archive. No soft advisories.
- **Schema as contract.** Components communicate through versioned schemas (`signal_interface_spec.md`, `discovery_engine_proposal.md` §4.1). Schema changes are arbitrated by Orchestration.

---

## Open Design Questions

- Push vs pull delivery of signals from Observability to consumers.
- Persistence and freshness policy for raw traces.
- Embedding model and vector store for Knowledge (or whether to skip embeddings in v1).
- Which model generates candidates in Candidate Generation and at what cost.
- How tightly Change Generation couples to specific execution backends vs. abstracting over them.
- Async handling for backends that take hours to complete.
- How to handle changes that improve one metric and regress another (Validation's verdict policy).

