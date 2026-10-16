# Validation Phase Flow Charts

## Overview: Full Validation Pipeline

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TB
    subgraph Trigger["Triggers"]
        C[Candidate available]
        CH[Change available]
        ER[ExecutionResult available<br/>— produced by signal_pipeline]
    end

    C --> P1
    CH --> P2
    CH --> P3
    ER --> P3

    subgraph P1["Phase 1: Validation Discovery — discover_validation_plan()"]
        direction TB
        P1A[Query Knowledge archive] --> P1D[Build base ValidationPlan]
        P1B[Run test harness discovery] --> P1D
        P1C[Seed workload matrix] --> P1D
    end

    subgraph P2["Phase 2: Change-Specific Discovery — specialize_validation_plan()"]
        direction TB
        P2A[Inspect Change for embedded tests] --> P2D[Augment & re-prioritize plan]
        P2B[Scan source tree for change-specific entries] --> P2D
    end

    subgraph P3["Phase 3: Validation Execution — run_validation_plan()"]
        direction TB
        P3A[Execute plan entries in priority order] --> P3B[Collect results]
        P3B --> P3C[Compute verdict]
        P3C --> P3D[Write to Knowledge archive]
    end

    P1 --> P2
    P2 --> P3
    P3 --> VR[ValidationResult]
```

---

## API: `discover_validation_plan(source_tree, artifacts_dir | knowledge db, candidate?, change?)`

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([discover_validation_plan called]) --> CheckTarget{candidate or<br/>change provided?}

    CheckTarget -->|No| Error([Raise ValueError:<br/>candidate or change required])
    CheckTarget -->|Yes| CheckArtifacts{artifacts_dir<br/>or knowledge db provided?}

    CheckArtifacts -->|Yes| LoadDisk[Load pre-built artifacts<br/>from disk]
    CheckArtifacts -->|No| LiveDiscovery[Run live discovery<br/>— not yet implemented]

    LoadDisk --> BuildHarness[Parse TestHarnessMap<br/>from harness_map.json]
    LoadDisk --> BuildWorkloads[Parse ValidationWorkloadMatrix<br/>from workload_matrix.json]
    LoadDisk --> BuildPlan[Parse ValidationPlan<br/>from validation_plan.json]

    LiveDiscovery --> QueryArchive[Query Knowledge archive<br/>for historical outcomes]
    LiveDiscovery --> ScanCI[Scan CI configs, test dirs,<br/>benchmarks in source_tree]
    LiveDiscovery --> DiscoverWorkloads[Discover workloads<br/>from target repo]

    QueryArchive --> AssemblePlan
    ScanCI --> AssemblePlan
    DiscoverWorkloads --> AssemblePlan

    AssemblePlan[Build prioritized ValidationPlan<br/>scoped to candidate/change components]

    BuildHarness --> PrepRun
    BuildWorkloads --> PrepRun
    BuildPlan --> PrepRun
    AssemblePlan --> PrepRun

    PrepRun[Create PreparationRun] --> Return([Return PreparationRun<br/>with prep_id])
```

---

## API: `specialize_validation_plan(change, execution_result, preparation)`

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([specialize_validation_plan called]) --> AwaitPrep{PreparationRun<br/>still running?}

    AwaitPrep -->|Yes| Wait[Await preparation.result]
    AwaitPrep -->|No| GetPrep[Get ValidationPreparation]

    Wait --> GetPrep

    GetPrep --> ChangeDiscovery[Change-Specific Discovery]

    subgraph ChangeDiscovery["Phase 2: Change Discovery"]
        direction TB
        CD1[Inspect Change for<br/>embedded test cases] --> CD3[Augment base plan<br/>with change-specific entries]
        CD2[Scan source tree for<br/>change-relevant tests] --> CD3
        CD3 --> CD4[Re-prioritize by<br/>affected code paths]
    end

    ChangeDiscovery --> Return([Return augmented<br/>ValidationPlan])
```

---

## API: `run_validation_plan(plan, source_tree, ...)`

This is the core execution engine (Phase 3 implementation).

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([run_validation_plan called]) --> Sort[Sort entries by priority<br/>lower = run first]
    Sort --> Loop{Next entry?}

    Loop -->|No more entries| Verdict

    Loop -->|Yes| CheckHalt{halt_triggered?}
    CheckHalt -->|Yes| SkipEntry[Record as skipped<br/>due to halt]
    SkipEntry --> Loop

    CheckHalt -->|No| CheckSkipped{entry.skipped?}
    CheckSkipped -->|Yes| LogSkip[Log skip reason]
    LogSkip --> Loop

    CheckSkipped -->|No| CheckDry{dry_run?}
    CheckDry -->|Yes| RecordDry[Record placeholder result]
    RecordDry --> Loop

    CheckDry -->|No| CheckKind{entry.kind?}

    CheckKind -->|benchmark| RunBench[_run_benchmark]
    CheckKind -->|test / unit /<br/>integration /<br/>correctness / stress| RunTest[_run_test]

    RunTest --> ParseOutput[Parse test output<br/>pytest-json / exit-code]
    ParseOutput --> BuildTestResult[Build TestResult<br/>passed/failed/skipped/errors]
    BuildTestResult --> CheckFail

    RunBench --> ExtractMetrics[Extract benchmark metrics<br/>json-file / json-stdout /<br/>regex / key-value]
    ExtractMetrics --> BuildBenchResult[Build BenchmarkResult<br/>with MetricResults]
    BuildBenchResult --> CheckFail

    CheckFail{Failed AND<br/>halt_on_failure?}
    CheckFail -->|Yes| SetHalt[halt_triggered = true]
    CheckFail -->|No| Loop
    SetHalt --> Loop

    subgraph Verdict["Compute Verdict"]
        direction TB
        V1{halt_triggered?}
        V1 -->|Yes| VFail1[verdict = FAIL<br/>halted after critical failure]
        V1 -->|No| V2{correctness<br/>failures?}
        V2 -->|Yes| VFail2[verdict = FAIL<br/>correctness checks failed]
        V2 -->|No| V3{any non-critical<br/>failures?}
        V3 -->|Yes| VCond[verdict = CONDITIONAL<br/>non-critical entries failed]
        V3 -->|No| VPass[verdict = PASS<br/>all checks passed]
    end

    Verdict --> Return([Return ValidationResult])
```

---

## API: `_run_test(entry, source_tree, timeout, logs_dir)`

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([_run_test]) --> BuildCmd[Build cmd from<br/>harness_entry.invoke]
    BuildCmd --> CheckJson{output_format ==<br/>pytest-json AND<br/>plugin available?}

    CheckJson -->|Yes| AddFlags[Append --json-report<br/>--json-report-file flags]
    CheckJson -->|No| SkipFlags[Use cmd as-is]

    AddFlags --> Stream
    SkipFlags --> Stream

    Stream[_stream_subprocess<br/>live output with prefix] --> CheckLog{logs_dir set?}

    CheckLog -->|Yes| WriteLog[Write captured output<br/>to logs_dir/entry.log]
    CheckLog -->|No| Parse

    WriteLog --> Parse

    Parse[_parse_test_output] --> CheckFormat{output_format?}

    CheckFormat -->|pytest-json<br/>+ report exists| ParseJSON[Parse JSON report<br/>extract summary counts<br/>+ per-test errors]
    CheckFormat -->|exit-code-only<br/>or fallback| ParseExit{returncode == 0?}

    ParseJSON --> CheckPytest{Framework error?<br/>No tests ran?}
    CheckPytest -->|Yes| SynthError[Synthesize error:<br/>pytest exit code N]
    CheckPytest -->|No| GotCounts[passed/failed/skipped<br/>+ TestError list]

    ParseExit -->|Yes| Pass1[passed=1, failed=0]
    ParseExit -->|No| ScrapeErrors[Scrape FAILED/ERROR lines<br/>from output]

    SynthError --> Result
    GotCounts --> Result
    Pass1 --> Result
    ScrapeErrors --> Result

    Result[Build TestResult] --> Return(["Return (TestResult, failed)"])
```

---

## API: `_run_benchmark(entry, source_tree, timeout, logs_dir)`

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([_run_benchmark]) --> BuildCmd[Build cmd from<br/>harness_entry.invoke]
    BuildCmd --> CheckWL{entry.workloads<br/>non-empty?}

    CheckWL -->|Yes| UseWorkloads[Iterate over workloads]
    CheckWL -->|No| UsePlaceholder[Use single 'default' workload]

    UseWorkloads --> WLLoop
    UsePlaceholder --> WLLoop

    WLLoop{Next workload?} -->|Yes| CheckConfig{workload has<br/>config_path?}
    WLLoop -->|No| Combine

    CheckConfig -->|Yes| AppendFlag[Append --workload=config_path<br/>if not already in invoke]
    CheckConfig -->|No| RunAsIs[Use base cmd]

    AppendFlag --> Stream
    RunAsIs --> Stream

    Stream[_stream_subprocess<br/>live output] --> SaveLog{logs_dir set?}

    SaveLog -->|Yes| WriteLog[Write to<br/>logs_dir/entry.workload_id.log]
    SaveLog -->|No| Extract

    WriteLog --> Extract

    Extract[_extract_benchmark_metrics] --> CheckTemplate{output_template.format?}

    CheckTemplate -->|json-file| JsonFile[Read JSON file from disk<br/>resolve glob, pick newest]
    CheckTemplate -->|json-stdout| JsonStdout[Find JSON blob in output<br/>extract by dot-paths]
    CheckTemplate -->|regex-text| Regex[Apply per-metric regex<br/>to output text]
    CheckTemplate -->|key-value-text| KV[Scrape key=value and<br/>Label: value lines]

    JsonFile --> BuildMetrics[Build MetricResult list]
    JsonStdout --> BuildMetrics
    Regex --> BuildMetrics
    KV --> BuildMetrics

    BuildMetrics --> AppendBR[Append BenchmarkResult<br/>for this workload]
    AppendBR --> WLLoop

    Combine[Combine all workload results<br/>into single BenchmarkResult] --> Return(["Return (BenchmarkResult, False)<br/>benchmarks never trigger halt"])
```

---

## API: `get_validation_status(run_id)`

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([get_validation_status called]) --> Lookup[Look up run by run_id]
    Lookup --> Found{Run exists?}

    Found -->|No| Error([Raise NotFoundError])
    Found -->|Yes| Build[Build ValidationStatus]

    Build --> Status["ValidationStatus {
        run_id,
        phase: discovery|planning|executing|archiving|complete|failed,
        completed_entries,
        total_entries,
        current_entry,
        timestamp
    }"]

    Status --> Return([Return ValidationStatus])
```

---

## CLI: `validation run` Command

```mermaid
%%{init: {'themeVariables': {'fontSize': '18px'}}}%%
flowchart TD
    Start([CLI: validation run]) --> LoadPlan[Load ValidationPlan<br/>from --plan path]
    LoadPlan --> CheckIndexes{--indexes<br/>provided?}

    CheckIndexes -->|Yes| Filter[Filter plan entries<br/>to requested indexes only]
    CheckIndexes -->|No| Full[Use full plan]

    Filter --> CheckRef
    Full --> CheckRef

    CheckRef{--change-ref?}
    CheckRef -->|Yes| SetRef[Set plan.change_ref]
    CheckRef -->|No| SetupLogs

    SetRef --> SetupLogs

    SetupLogs[Resolve logs_dir<br/>default: ./logs/timestamp] --> PrintSummary[Print plan summary<br/>total entries, halt count]

    PrintSummary --> Execute[run_validation_plan<br/>with dry_run, timeout_multiplier,<br/>logs_dir options]

    Execute --> PrintResult[Print formatted summary<br/>verdict, test table, failures,<br/>benchmark metrics]

    PrintResult --> CheckOut{--out<br/>provided?}
    CheckOut -->|Yes| WriteJSON[Write ValidationResult<br/>to --out path]
    CheckOut -->|No| Exit

    WriteJSON --> Exit

    Exit{verdict?}
    Exit -->|pass| Exit0([exit 0])
    Exit -->|fail or conditional| Exit1([exit 1])
```
