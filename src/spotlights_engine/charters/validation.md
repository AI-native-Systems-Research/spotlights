# Charter — Validation (Bundle E)

**Repo:** `spotlights-engine`, subpackage `validation`

**Mission.** Confirm that changes flowing through the discovery loop preserve correctness, improve the targeted metric, and don't regress on others. Bundle E is the gate before the archive — nothing enters the experimental record without a measured verdict.

**Control flow.**

```
F receives Change + ExecutionResult from D
  │
  ▼
F hands them to E: "validate this"
  │
  ▼
E runs (owns the entire validation process):
  1. Queries B for historical outcomes on similar candidates/components
  2. Selects relevant test scripts from TestHarnessMap (informed by 1)
  3. Selects workloads from module E's validation workload matrix (informed by 1)
  4. Executes selected test and benchmark scripts against the modified code under selected workloads
  5. Assembles ValidationResult from test/benchmark output
  6. Writes the complete record (Candidate + Change + ExecutionResult + ValidationResult) to archive (B)
  │
  ▼
E returns ValidationResult to F
  │
  ▼
F decides next steps (archive, revert, proceed)
```

Bundle E owns the full validation process end-to-end. It runs the target's own test and benchmark scripts directly — no dependency on Bundle A at runtime.

**In scope.**
- Automatic discovery of the target system's existing test and benchmark infrastructure (test harness discovery). Given the target's source tree, produce a structured map of available validation scripts — what they test, how to invoke them, what component they cover, and what kind of check they represent (unit, integration, benchmark, correctness).
- Selecting which discovered scripts are relevant to a given `Change`, based on the components affected (`Change.required_changes`) and the type of validation needed.
- Owning the **validation workload matrix**: the set of workloads a change must be tested against to confirm it doesn't regress any in-scope workload class. Seeded from workload/benchmark configurations discovered in the target repo and curated over time. This is a superset of what Bundle A uses for signal discovery — a change that improves one workload class must not silently regress others. Bundle E selects the relevant subset per change (based on affected components) and runs the target's benchmark scripts under workloads selected from module E's matrix.
- Functional correctness checks for proposed changes, primarily by invoking the target's own test suites rather than maintaining a separate correctness suite.
- Performance comparison: baseline (pre-change) vs. post-change measurements across workloads selected from module E's matrix, using the target's own benchmark scripts.
- Cost regression detection: memory, compute utilization, request cost.
- Robustness checks under adversarial or stress workloads where applicable.
- Intent-alignment verification: did the change produce the expected effect declared in `Change.expected_effect`, or did it improve something else by accident?
- Querying the archive for past outcomes on similar candidates or components to select relevant regression checks.
- Writing the full validation outcome (candidate → change → result) to the archive.

**Out of scope.**
- Proposing changes. That is Bundle D.
- Defining new workload *types* from scratch. Bundle E discovers and curates workload configurations from the target; it does not invent synthetic workload patterns.
- Deciding whether to *act* on a validation result (revert, merge, re-try with different parameters). That decision belongs to Bundle F.
- Long-term governance, audit trails for compliance. This is discovery-focused validation only.
- Implementing the change in code. Execution backends (evolutionary search, coding agents) do that; Bundle E receives the result.
- **Change-specific test generation (v1).** In v1, Bundle E relies on the target's existing test suites and performance measurements to validate changes. It does not generate new tests tailored to the specific code a change introduces. This is a deliberate scoping decision: existing scripts plus performance delta are sufficient for MVP. Post-v1, change-specific test generation (e.g., asking a coding agent to write tests exercising a newly added prefetch path) is a natural extension - which bundle owns this is TBD.

**Contracts.**

*In:*

| Source | Object | Description |
|---|---|---|
| Target repo | Source tree | The target system's source code, scanned to discover test and benchmark scripts. |
| Bundle D | `Change` | The proposed modification, including `candidate_ref`, `change_type`, `mechanism`, `expected_effect`, `evaluation_metric`. |
| Bundle D | `ExecutionResult` | Raw output from the execution backend: whether it compiled/ran, any errors, artifacts produced. |
| Bundle A (via Candidate) | `TraceSummary` / `Anomaly` IDs | References to the observability signals that originally motivated the candidate. Passed through for traceability in the archive record; not consumed by E's validation logic. |
| Bundle B | Past `ValidationResult` entries | Historical outcomes for the same components or similar change types, used to select regression checks. |

*Out:*

| Consumer | Object | Description |
|---|---|---|
| Internal | `TestHarnessMap` | Structured map of the target's available validation scripts (see shape below). Produced once per target version; reused across validation runs. |
| Internal | `ValidationWorkloadMatrix` | The set of workloads validation tests against, discovered from the target and curated. Bundle E selects from this per change. |
| Bundle F | `ValidationResult` | Structured verdict with measurements (see shape below). |
| Bundle B | Archive write | The complete record: `Candidate` + `Change` + `ExecutionResult` + `ValidationResult`, persisted for future reference. |

*`TestHarnessMap` shape (informative, not locked until end of Stage 0):*

```
TestHarnessMap {
  target_version:    commit SHA or tag of the target repo when scanned
  entries: [
    {
      id:            unique identifier for this entry
      name:          human-readable label (e.g., "unit tests — scheduler")
      kind:          unit | integration | benchmark | correctness | stress
      path:          repo-relative path to the script or test directory
      invoke:        command to run it (e.g., "pytest tests/unit/scheduler/")
      components:    list of target components this covers (same vocabulary as TraceSummary.top_components)
      output_format: how to parse results (pytest-json | custom | exit-code-only)
      estimated_duration: rough wall-clock time (seconds), if known
    },
    ...
  ]
}
```

*`ValidationWorkloadMatrix` shape (informative, not locked until end of Stage 0):*

```
ValidationWorkloadMatrix {
  target_version:    commit SHA or tag of the target repo when scanned
  workloads: [
    {
      workload_id:     unique identifier
      workload_class:  e.g., agentic, batch-inference, long-context, mixed
      source:          discovered (from target repo) | curated (manually added)
      config_path:     repo-relative path to the workload config, if discovered
      components_exercised: list of target components this workload stresses
    },
    ...
  ]
}
```

*`ValidationResult` shape (informative, not locked until end of Stage 0):*

```
ValidationResult {
  change_ref:           reference to the Change under test
  verdict:              pass | fail | conditional
  verdict_reasoning:    summary of why this verdict was reached (references key
                        metrics, test failures, or threshold breaches)
  conditions:           list[str] — if verdict is conditional, what conditions
                        must be met (e.g., "retest under long-context workload",
                        "acceptable only if memory regression is transient").
                        Empty when verdict is pass or fail.
  intent_aligned:       bool — did the change achieve its declared expected_effect?

  test_results: [
    {
      harness_id:       TestHarnessMap entry id
      script:           TestHarnessMap entry name (e.g., "unit tests — scheduler")
      kind:             unit | integration | correctness
      passed:           int
      failed:           int
      skipped:          int
      errors:           list[str]
      duration_seconds: wall-clock time
    },
    ...
  ]

  benchmark_results: [
    {
      workload_id:      which workload from module E's matrix
      workload_class:   e.g., agentic, batch-inference
      script:           TestHarnessMap entry name (e.g., "throughput bench — agentic")
      metrics: [
        {
          name:             e.g., "request_latency_p99", "throughput_rps", "gpu_memory_peak"
          baseline_value:   measured value before the change
          measured_value:   measured value after the change
          relative_change:  percentage
          regressed:        bool — worsened beyond threshold
        },
        ...
      ]
      optimization_target:  computed score representing the overall optimization
                            objective, derived from the individual metrics above
                            (e.g., weighted combination of latency and throughput)
    },
    ...
  ]

  observability_context: {
    signal_refs:        list of TraceSummary/Anomaly IDs from Bundle A that
                        originally motivated the candidate — carried through
                        for traceability, not produced by E
  }

  notes:                free-text observations or warnings
  timestamp:            when validation completed
}
```

**Success criteria for Stage 1.**
- Test harness discovery runs on the target repo and produces a `TestHarnessMap` with at least 3 entries covering distinct components and kinds. For the demo, the entries should include: (1) unit tests targeting the specific component a change touches, (2) integration/correctness tests verifying the component works within the broader system, and (3) a benchmark measuring performance delta (e.g., throughput/latency under a representative workload).
- Given a `Change`, Bundle E automatically selects the relevant subset of discovered scripts that is feasible to run (not the full suite — constrained by available resources, time budget, and configuration possibilities).
- Validation workload matrix contains at least 2 distinct workload classes discovered or curated from the target.
- Can validate at least one change end-to-end: receive `Change` + `ExecutionResult`, run selected target test and benchmark scripts across multiple workloads from module E's matrix, produce a `ValidationResult` with a measured performance delta.
- Correctness checks use the target's own test suites (not a Bundle E-maintained suite).
- At least one regression check is informed by a prior archive entry (demonstrating the read path from Bundle B).
- The validation outcome is successfully written to the archive (demonstrating the write path to Bundle B).
- Stage 1 demo includes a validated change with a clear pass or fail verdict and measured improvement or regression.

**Dependencies on other bundles.**
- **Bundle B** — archive read interface (`query` with mode to retrieve past validation outcomes by component or change type) and archive write interface (persist new `ValidationResult` entries). Bundle B's retrieval interface must support a validation-oriented query mode or accept structured filters.
- **Bundle D** — provides `Change` and `ExecutionResult` as the trigger for validation. Stubs are sufficient for early development; real execution results needed by mid-Stage 1.
- **Bundle F** — consumes `ValidationResult` and decides next steps. Bundle E does not depend on F for development, but integration testing requires F to close the loop.

**Open questions the owner resolves.**
- Whether validation runs synchronously (blocking the loop until complete) or asynchronously (producing results that F polls for).
- Threshold policy: who defines "beyond threshold" for regression detection — a per-metric config, a percentage heuristic, or a statistical test?
- How to handle flaky results (variance across runs). Re-run policy, confidence intervals, or minimum-N-runs requirement.
- Whether `ExecutionResult` is a new schema owned by Bundle D or a generic blob that Bundle E must parse.
- How adversarial/stress workloads are selected — curated list, or generated from the workload class.
- Test harness discovery: how much is static analysis (scan for `pytest.ini`, `Makefile` targets, CI configs) vs. LLM-assisted (read test files and infer which components they cover)? The `components` field in `TestHarnessMap` requires mapping test paths to system components — this may need the `modules_extractor`'s output or an LLM pass.
- How to handle target scripts with custom output formats that don't map cleanly to pass/fail/metrics. Adapter strategy vs. requiring a standard output contract.
- How often to re-run harness discovery and workload matrix refresh (on every target commit, on version bumps only, manually triggered).
- Workload matrix completeness: what's the minimum coverage bar? Must every known workload class be represented, or is "components affected by the change" a sufficient selection filter?
- Validation error handling: how to handle cases where a test or benchmark could not be launched due to environmental issues (e.g., missing dependencies, unavailable hardware, timeout before execution starts). Should these be reported as inconclusive, trigger a retry, or fail the validation run?
