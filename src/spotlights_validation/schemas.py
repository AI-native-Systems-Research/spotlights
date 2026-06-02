from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Discovery outputs
# ---------------------------------------------------------------------------


class BenchmarkOutputTemplate(BaseModel):
    """Describes how to find and parse a specific benchmark's summary output.

    Each benchmark emits results in its own shape: some print key/value lines to
    stdout, some emit JSON on stdout, some write a JSON file to disk. The
    template tells the runner which case applies and which metric values to lift
    out of that output.
    """

    format: Literal[
        "json-file",       # benchmark writes a JSON file under source_tree
        "json-stdout",     # benchmark prints a JSON document on stdout
        "regex-text",      # extract values via per-metric regex
        "key-value-text",  # default: scrape `key = value` / `Label: value` lines
    ] = "key-value-text"
    # For json-file: path or glob (relative to source_tree, or absolute).
    # When the glob matches multiple files the most recently modified one wins.
    source: str | None = None
    # name -> dot-path (json-*) or regex with one capture group (regex-text).
    # For key-value-text, an optional allowlist of metric names to keep.
    metric_paths: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class TestHarnessEntry(BaseModel):
    id: str
    name: str
    kind: Literal["unit", "integration", "benchmark", "correctness", "stress"]
    path: str
    invoke: str
    components: list[str]
    output_format: Literal["pytest-json", "custom", "exit-code-only"]
    estimated_duration: int | None = None
    output_template: BenchmarkOutputTemplate | None = None
    comparison_guidance: str = ""


class TestHarnessMap(BaseModel):
    target_version: str
    entries: list[TestHarnessEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "TestHarnessMap":
        matched = [e for e in self.entries if any(c in e.components for c in components)]
        return TestHarnessMap(target_version=self.target_version, entries=matched)

    def of_kind(self, kind: str) -> "TestHarnessMap":
        matched = [e for e in self.entries if e.kind == kind]
        return TestHarnessMap(target_version=self.target_version, entries=matched)

    def to_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "TestHarnessMap":
        return cls.model_validate_json(path.read_text())


class WorkloadEntry(BaseModel):
    workload_id: str
    workload_class: Literal["agentic", "batch-inference", "long-context", "mixed", "stress"]
    source: Literal["discovered", "curated"]
    config_path: str | None = None
    components_exercised: list[str]
    description: str = ""


class ValidationWorkloadMatrix(BaseModel):
    target_version: str
    workloads: list[WorkloadEntry] = Field(default_factory=list)

    def for_components(self, components: list[str]) -> "ValidationWorkloadMatrix":
        matched = [w for w in self.workloads if any(c in w.components_exercised for c in components)]
        return ValidationWorkloadMatrix(target_version=self.target_version, workloads=matched)

    def to_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ValidationWorkloadMatrix":
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class ValidationPlanEntry(BaseModel):
    harness_entry: TestHarnessEntry
    workloads: list[WorkloadEntry]
    priority: int
    halt_on_failure: bool
    index: int | None = None
    skipped: bool = False
    skip_reason: str = ""


class ValidationPlan(BaseModel):
    entries: list[ValidationPlanEntry]
    archive_context: list[str] = Field(default_factory=list)
    change_ref: str | None = None

    def to_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "ValidationPlan":
        return cls.model_validate_json(path.read_text())


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class MetricResult(BaseModel):
    name: str
    baseline_value: float | None = Field(default=None, exclude=True)
    measured_value: float | None = None
    relative_change: float | None = Field(default=None, exclude=True)
    regressed: bool = Field(default=False, exclude=True)


class TestError(BaseModel):
    nodeid: str
    phase: str = "call"  # setup | call | teardown
    message: str = ""
    file: str | None = None
    lineno: int | None = None
    longrepr: str = ""  # truncated traceback / failure repr


class TestResult(BaseModel):
    harness_id: str
    script: str
    kind: str
    invoke: str = ""
    index: int | None = None
    priority: int | None = None
    halt_on_failure: bool = False
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: list[TestError] = Field(default_factory=list)
    duration_seconds: float = 0.0
    log_path: str | None = None  # path to captured combined stdout/stderr
    json_report_path: str | None = None  # path to pytest-json-report file


class BenchmarkResult(BaseModel):
    workload_id: str
    workload_class: str
    script: str
    invoke: str = ""
    index: int | None = None
    priority: int | None = None
    halt_on_failure: bool = False
    metrics: list[MetricResult] = Field(default_factory=list)
    optimization_target: float | None = None
    raw_output: str = ""


class ObservabilityContext(BaseModel):
    signal_refs: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    change_ref: str | None
    verdict: Literal["pass", "fail", "conditional"]
    verdict_reasoning: str
    conditions: list[str] = Field(default_factory=list)
    intent_aligned: bool | None = None
    test_results: list[TestResult] = Field(default_factory=list)
    benchmark_results: list[BenchmarkResult] = Field(default_factory=list)
    observability_context: ObservabilityContext = Field(default_factory=ObservabilityContext)
    notes: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_json(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))


# ---------------------------------------------------------------------------
# Public API types
# ---------------------------------------------------------------------------


class ExecutionResult(BaseModel):
    change_ref: str
    compiled: bool
    ran: bool
    exit_code: int | None = None
    errors: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    raw_output: str = ""


class PreparationRun(BaseModel):
    prep_id: str


class ValidationRun(BaseModel):
    run_id: str


class ValidationStatus(BaseModel):
    run_id: str
    phase: Literal["discovery", "planning", "executing", "archiving", "complete", "failed"]
    completed_entries: int = 0
    total_entries: int = 0
    current_entry: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationPreparation(BaseModel):
    source_tree: Path
    harness_map: TestHarnessMap
    workload_matrix: ValidationWorkloadMatrix
    full_plan: ValidationPlan
    archive_context: list[str] = Field(default_factory=list)
