"""Discovery-benchmark module.

Builds a reproducible answer key of merged perf PRs from a target repo,
emits a snapshot SHA + workload spec, and grades discovery findings
against the answer key by direct match against PR diffs.
"""

from spotlights_engine.repo_bench.schemas import (
    AggregationManifest,
    BenchSpec,
    BenchSpecConfig,
    ChangeType,
    FileChange,
    RawPR,
    RuleSpec,
    RunReport,
    SnapshotPin,
    StageReport,
)
from spotlights_engine.repo_bench.storage import window_id_for

__all__ = [
    "AggregationManifest",
    "BenchSpec",
    "BenchSpecConfig",
    "ChangeType",
    "FileChange",
    "RawPR",
    "RuleSpec",
    "RunReport",
    "SnapshotPin",
    "StageReport",
    "window_id_for",
]
