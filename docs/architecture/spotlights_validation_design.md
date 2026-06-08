
# Charter — Validation (Bundle E)

This document describes the design for Bundle E, which is one of the bundles detailed in [spotlighs_design](spotlighs_design.md)


# Mission

Confirm that changes flowing through the discovery loop preserve correctness, improve the targeted metric, and don't regress on others. Bundle E is the gate before the archive — nothing enters the experimental record without a measured verdict.

# Control flow

```
Validation Discovery Phase — parallel to Bundle D (Change Generation and Execution):

  E starts as soon as the source tree and Candidate are available:
  1. Queries B for historical outcomes on this candidate's components
     (using Candidate.file, Candidate.symbol, and Candidate.anomaly_refs).
  2. Runs test harness discovery on the source tree (CI configs, test dirs, benchmark
     scripts, workload configs, issues, pull requests) to produce the TestHarnessMap and seed the workload matrix (skipped if cached from a prior run on the same target version).
  3. Produces a base validation plan — harness entries and workloads prioritized
     toward the candidate's component, ranked by historical failure signal and
     halt conditions. 
  └─→ Produces: ValidationPreparation

Change Validation Discovery (optional) — triggered when Change becomes available:

  May run before the Validation Execution, as soon as the Change object is delivered from Bundle D:
  4. Inspects the Change for embedded test cases, specific benchmarks, or other
     validation artifacts provided by the change author or generation mechanism.
  5. Scans the source tree for additional validation entries highly specific to the
     components and code paths affected by this Change (e.g., narrowly scoped tests
     or micro-benchmarks covering the exact function modified).
  6. Appends discovered change-specific entries to the base validation plan, producing
     an augmented ValidationPlan with updated priorities.
  └─→ Produces: augmented ValidationPlan (extends ValidationPreparation)

Validation Execution — triggered when Bundle D delivers Change + ExecutionResult:

  F hands Change + ExecutionResult + ValidationPreparation (possibly augmented) to E:
  "validate this"
  7. Executes the validation plan according to the rank (priorities) values.
  8. Assembles ValidationResult from test/benchmark output.
  9. Writes the complete record (Candidate + Change + ExecutionResult + ValidationResult)
     to archive (B).
  │
  ▼
E returns ValidationResult to F
  │
  ▼
F decides next steps (archive, revert, proceed)
```

Bundle E owns the full validation process end-to-end. It runs the target's own test and benchmark scripts directly — no dependency on Bundle A at runtime.

# In scope
- Automatic discovery of the target system's existing test and benchmark infrastructure (test harness discovery). Given the target's source tree, produce a structured map of available validation scripts — what they test, how to invoke them, what component they cover, and what kind of check they represent (unit, integration, benchmark, correctness).
- Prioritizing the discovered validation entities according to a given `Change`, based on the components affected (`Change.required_changes`) and the type of validation needed.
- Owning the **validation workload matrix**: the set of workloads a change must be tested against to confirm it doesn't regress any in-scope workload class. Seeded from workload/benchmark configurations discovered in the target repo and curated over time. This is a superset of what Bundle A uses for signal discovery — a change that improves one workload class must not silently regress others. Bundle E selects the relevant subset per change (based on affected components) and runs the target's benchmark scripts under workloads selected from module E's matrix.
- Functional correctness checks for proposed changes, primarily by invoking the target's own test suites rather than maintaining a separate correctness suite.
- Performance comparison: baseline (pre-change) vs. post-change measurements across workloads selected from module E's matrix, using the target's own benchmark scripts.
- Cost regression detection: memory, compute utilization, request cost.
- Robustness checks under adversarial or stress workloads where applicable.
- Intent-alignment verification: did the change produce the expected effect declared in `Change.expected_effect`, or did it improve something else by accident?
- Querying the archive for past outcomes on similar candidates or components to select relevant regression checks.
- Writing the full validation outcome (candidate → change → result) to the archive.

# Out of scope
- Proposing changes. That is Bundle D.
- Defining new workload *types* from scratch. Bundle E discovers and curates workload configurations from the target; it does not invent synthetic workload patterns.
- Deciding whether to *act* on a validation result (revert, merge, re-try with different parameters). That decision belongs to Bundle F.
- Long-term governance, audit trails for compliance. This is discovery-focused validation only.
- Implementing the change in code. Execution backends (evolutionary search, coding agents) do that; Bundle E receives the result.
- **Change-specific test generation (v1).** In v1, Bundle E relies on the target's existing test suites and performance measurements to validate changes. It does not generate new tests tailored to the specific code a change introduces. This is a deliberate scoping decision: existing scripts plus performance delta are sufficient for MVP. Post-v1, change-specific test generation (e.g., asking a coding agent to write tests exercising a newly added prefetch path) is a natural extension - which bundle owns this is TBD.

# Contracts

## Validation Discovery Phase (parallel to Bundle D)

*In:*

| Source | Object | Description |
|---|---|---|
| Bundle C | `Candidate` | The optimization candidate that triggered this validation cycle — provides the target component (`file`, `symbol`, `kind`), `anomaly_refs` linking to originating signals, and `evolve_rationale` explaining what improvement is expected. Used to focus archive queries and prioritize the base validation plan toward the candidate's component. |
| Target repo | Source tree | The target system's source code, scanned to discover test and benchmark scripts. |
| Bundle B | Past `ValidationResult` entries | Historical outcomes for the same components or similar change types (queried using the Candidate's component info), used to select regression checks. |

*Out:*

| Consumer | Object | Description |
|---|---|---|
| Internal | `TestHarnessMap` | Structured map of the target's available validation scripts (see shape below). Produced once per target version; reused across validation runs. |
| Internal | `ValidationWorkloadMatrix` | The set of workloads validation tests against, discovered from the target and curated. Bundle E selects from this per change. |
| Internal | `ValidationPlan` | Base validation plan covering all harness entries and workloads, prioritized toward the candidate's component and ranked by historical failure signal. Used as input to subsequent phases. |

## Change Validation Discovery (optional — triggered when Change becomes available)

*In:*

| Source | Object | Description |
|---|---|---|
| Bundle D | `Change` | The proposed modification — used to identify affected components, embedded test cases, and specific benchmarks bundled with the change. |
| Internal (Validation Discovery Phase) | `ValidationPreparation` | The base preparation including `TestHarnessMap`, `ValidationWorkloadMatrix`, and `ValidationPlan`. |
| Target repo | Source tree | Re-scanned for validation entries specific to the code paths affected by this Change. |

*Out:*

| Consumer | Object | Description |
|---|---|---|
| Internal | Augmented `ValidationPlan` | The base plan extended with change-specific entries (embedded tests, targeted micro-benchmarks, narrowly scoped correctness checks). New entries are ranked and merged into the existing priority order. |

## Validation Execution (triggered by Bundle D delivery)

*In:*

| Source | Object | Description |
|---|---|---|
| Bundle D | `Change` | The proposed modification, including `candidate_ref`, `change_type`, `mechanism`, `expected_effect`, `evaluation_metric`. |
| Bundle D | `ExecutionResult` | Raw output from the execution backend: whether it compiled/ran, any errors, artifacts produced. |
| Bundle A (via Candidate) | `TraceSummary` / `Anomaly` IDs | References to the observability signals that originally motivated the candidate. Passed through for traceability in the archive record; not consumed by E's validation logic. |
| Internal (Validation Discovery Phase + optional Change Validation Discovery) | `ValidationPreparation` | The completed preparation (possibly augmented with change-specific entries): `TestHarnessMap`, `ValidationWorkloadMatrix`, `ValidationPlan`, and archive context. |

*Out:*

| Consumer | Object | Description |
|---|---|---|
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

# APIs

## Public API — what Bundle F imports

```python
from spotlights_engine.validation import (
    prepare, start_validation, get_validation_status,
    ValidationPreparation, ValidationResult, ValidationStatus, ValidationRun, PreparationRun, ExecutionResult,
)
from spotlights_engine.schemas import Change, Candidate

# Validation Discovery Phase — fire and forget, runs in background parallel to Bundle D
prep: PreparationRun = await prepare(source_tree=target_repo_path, candidate=candidate)

# Validation Execution — triggered when Change + ExecutionResult arrive from Bundle D
# start_validation awaits the preparation internally if it is still running
# and runs Change Validation Discovery to augment the plan before executing
run: ValidationRun = await start_validation(
    change=change,
    execution_result=execution_result,
    preparation=prep,
)
status: ValidationStatus = await get_validation_status(run.run_id)
result: ValidationResult = await run.result()
```

```python
async def prepare(source_tree: Path, candidate: Candidate) -> PreparationRun: ...
# Returns immediately; Validation Discovery Phase work runs in the background.
# The Candidate provides the target component and anomaly context used to
# focus archive queries and prioritize the base validation plan.

async def start_validation(
    change: Change,
    execution_result: ExecutionResult,
    preparation: PreparationRun,
) -> ValidationRun: ...
# Awaits preparation internally, then runs Change Validation Discovery (optional)
# to augment the plan with change-specific entries before starting Validation Execution.

async def get_validation_status(run_id: str) -> ValidationStatus: ...
```

`PreparationRun` — the handle returned by `prepare`:

```python
class PreparationRun(BaseModel):
    prep_id: str

    async def result(self) -> ValidationPreparation: ...  # waits for Validation Discovery Phase to complete
```

`ValidationPreparation` — the completed Validation Discovery Phase output, held inside `PreparationRun`:

```python
class ValidationPreparation(BaseModel):
    source_tree:      Path
    candidate:        Candidate                # the optimization candidate driving this validation cycle
    harness_map:      TestHarnessMap
    workload_matrix:  ValidationWorkloadMatrix
    base_plan:        ValidationPlan           # prioritized toward candidate's component, ranked by historical signal
    archive_context:  list[str]                # ValidationResult IDs queried from B (filtered by candidate's component)
```

`ValidationRun` — the handle returned by `start_validation`:

```python
class ValidationRun(BaseModel):
    run_id: str

    async def status(self) -> ValidationStatus: ...   # shorthand for get_validation_status(self.run_id)
    async def result(self) -> ValidationResult: ...   # waits for completion, then returns the result
```

`ValidationStatus` — the progress snapshot returned by `get_validation_status`:

```python
class ValidationStatus(BaseModel):
    run_id:            str
    phase:             Literal["discovery", "planning", "change_discovery", "executing", "archiving", "complete", "failed"]
    completed_entries: int        # harness entries finished so far (Validation Execution only)
    total_entries:     int        # 0 until validation plan is ready
    current_entry:     str | None # name of the harness entry currently executing
    timestamp:         datetime
```

`phase` splits across the phases: `discovery → planning` run in the Validation Discovery Phase; `change_discovery` runs in the optional Change Validation Discovery phase (skipped if no change-specific entries are found); `executing → archiving → complete` run in the Validation Execution phase. It moves to `failed` from any phase if a halt condition is triggered.

`ExecutionResult` — Bundle E defines this interface until Bundle D firms it up:

```python
class ExecutionResult(BaseModel):
    change_ref: str
    compiled: bool
    ran: bool
    exit_code: int | None = None
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    raw_output: str = ""
```

## Internal APIs — exposed at module level for testing, not re-exported from `__init__`

```python
# discovery/harness_discovery.py
def discover_test_harness(source_tree: Path, target_version: str, candidate: Candidate) -> TestHarnessMap: ...

# discovery/workload_discovery.py
def seed_workload_matrix(source_tree: Path, target_version: str, candidate: Candidate) -> ValidationWorkloadMatrix: ...

# planning/planner.py  (Validation Discovery Phase)
def build_validation_plan(
    harness_map: TestHarnessMap,
    workload_matrix: ValidationWorkloadMatrix,
    archive_context: list[ValidationResult],
    candidate: Candidate,
) -> ValidationPlan: ...

# planning/change_discovery.py  (Change Validation Discovery — optional)
def discover_change_specific_entries(
    change: Change,
    base_plan: ValidationPlan,
    source_tree: Path,
    harness_map: TestHarnessMap,
) -> ValidationPlan: ...
# Inspects the Change for embedded tests/benchmarks and scans affected code paths
# in the source tree. Returns the augmented plan (base + change-specific entries).
# Returns the base plan unchanged if no change-specific entries are found.

# execution/runner.py
def run_validation_plan(plan: ValidationPlan, source_tree: Path) -> ValidationResult: ...
```

`ValidationPlan` is internal (not exported from `__init__`) — Bundle F has no need for it:

```python
class ValidationPlanEntry(BaseModel):
    harness_entry: TestHarnessEntry
    workloads: list[WorkloadEntry]
    priority: int
    halt_on_failure: bool
    source: Literal["base", "change_specific"] = "base"  # origin of this entry

class ValidationPlan(BaseModel):
    entries: list[ValidationPlanEntry]
    archive_context: list[str]  # ValidationResult IDs used to build this plan
    change_ref: str | None = None  # set when Change Validation Discovery augments the plan; None for the base plan
```

`TestHarnessMap` and `ValidationWorkloadMatrix` are Pydantic `BaseModel`s (matching the `ProjectTree` pattern in `spotlights-engine`) with `to_json`/`from_json` methods and filter helpers:

```python
class TestHarnessMap(BaseModel):
    target_version: str
    entries: list[TestHarnessEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "TestHarnessMap": ...
    def of_kind(self, kind: str) -> "TestHarnessMap": ...
    def to_json(self, path: Path) -> None: ...

    @classmethod
    def from_json(cls, path: Path) -> "TestHarnessMap": ...

class ValidationWorkloadMatrix(BaseModel):
    target_version: str
    workloads: list[WorkloadEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "ValidationWorkloadMatrix": ...
    def to_json(self, path: Path) -> None: ...

    @classmethod
    def from_json(cls, path: Path) -> "ValidationWorkloadMatrix": ...
```

The public API (`validate`) is asynchronous. Internal flows inside (discovery, planning, runner) might also be asynchronous. The `phase` states advance strictly in order — `discovery → planning → change_discovery → executing → archiving → complete` (or `failed` from any state) — each state must finish before the next begins. The `change_discovery` phase is optional and skipped when no change-specific entries are discovered. Within a single state, however, the work may itself be asynchronous: concurrent LLM calls during discovery, parallel harness executions during executing, etc. The blocking is at the state boundary, not inside it.

# Success criteria for Stage 1
- Test harness discovery runs on the target repo and produces a `TestHarnessMap` with at least 3 entries covering distinct components and kinds. For the demo, the entries should include: (1) unit tests targeting the specific component a change touches, (2) integration/correctness tests verifying the component works within the broader system, and (3) a benchmark measuring performance delta (e.g., throughput/latency under a representative workload).
- Given a `Change`, Bundle E automatically priorities the entries of TestHarnessMap. The tests will be executed according to these priorities. If a validation entry fails and is marked with `halt_on_failure`, the whole validation process will stop. 
- Validation workload matrix contains at least 2 distinct workload classes discovered or curated from the target.
- Can validate at least one change end-to-end: receive `Change` + `ExecutionResult`, run selected target test and benchmark scripts across multiple workloads from module E's matrix, produce a `ValidationResult` with a measured performance delta.
- Correctness checks use the target's own test suites (not a Bundle E-maintained suite).
- At least one regression check is informed by a prior archive entry (demonstrating the read path from Bundle B).
- The validation outcome is successfully written to the archive (demonstrating the write path to Bundle B).
- Stage 1 demo includes a validated change with a clear pass or fail verdict and measured improvement or regression.

# Dependencies on other bundles
- **Bundle B** — archive read interface (`query` with mode to retrieve past validation outcomes by component or change type) and archive write interface (persist new `ValidationResult` entries). Bundle B's retrieval interface must support a validation-oriented query mode or accept structured filters.
- **Bundle C** — provides `Candidate` as the trigger for the Validation Discovery Phase. The Candidate identifies the target component (`file`, `symbol`, `kind`), links to originating signals (`anomaly_refs`), and explains what improvement is expected (`evolve_rationale`). Bundle E uses this to focus archive queries and prioritize the base validation plan before any Change arrives.
- **Bundle D** — provides `Change` and `ExecutionResult` as the trigger for validation. Stubs are sufficient for early development; real execution results needed by mid-Stage 1.
- **Bundle F** — consumes `ValidationResult` and decides next steps. Bundle E does not depend on F for development, but integration testing requires F to close the loop.

# Open questions the owner resolves
- Threshold policy: who defines "beyond threshold" for regression detection — a per-metric config, a percentage heuristic, or a statistical test?
- How to handle flaky results (variance across runs). Re-run policy, confidence intervals, or minimum-N-runs requirement.
- Whether `ExecutionResult` is a new schema owned by Bundle D or a generic blob that Bundle E must parse.
- How adversarial/stress workloads are selected — curated list, or generated from the workload class.
- Test harness discovery: how much is static analysis (scan for `pytest.ini`, `Makefile` targets, CI configs) vs. LLM-assisted (read test files and infer which components they cover)? The `components` field in `TestHarnessMap` requires mapping test paths to system components — this may need the `modules_extractor`'s output or an LLM pass.
- How to handle target scripts with custom output formats that don't map cleanly to pass/fail/metrics. Adapter strategy vs. requiring a standard output contract.
- How often to re-run harness discovery and workload matrix refresh (on every target commit, on version bumps only, manually triggered).
- Workload matrix completeness: what's the minimum coverage bar? Must every known workload class be represented?
- Validation error handling: how to handle cases where a test or benchmark could not be launched due to environmental issues (e.g., missing dependencies, unavailable hardware, timeout before execution starts). Should these be reported as inconclusive, trigger a retry, or fail the validation run?
- Workload entities are specific to vLLM target system. Should that data be wrapped by an abstract data stracture ?
