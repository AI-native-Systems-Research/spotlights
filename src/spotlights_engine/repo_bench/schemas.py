"""Schemas for the repo bench.

`RawPR` is the row written to `data/repo_bench/raw/<window_id>/prs.jsonl`.
`AggregationManifest` is the sibling file recording what was scraped.

Schemas are versioned via `schema_version` on the manifest. Bump it on
any field rename or removal; additions of optional fields are
backward-compatible.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.2.0"

# Closed set of change kinds. Mirrors `signal_pipeline.schemas.ChangeType`
# verbatim so any agent emitting findings writes in the same vocabulary.
ChangeType = Literal[
    "prefetch", "reorder", "replace", "tune", "add_cache", "fuse", "other"
]


class RuleSpec(BaseModel):
    """Serializable description of a filter rule.

    Used to compute `view_id` deterministically: same rule list → same
    canonical JSON → same `view_id`. Predicates are commutative (sorted
    by name+version before hashing); rankers are positional.

    `params` includes every parameter that affects rule behavior,
    including regex patterns and any internal thresholds — so a regex
    tweak forces a new `view_id` even with the same rule name.

    `version` is opaque (not semver) and bumped on any code change that
    affects rule behavior, including changes that aren't visible in
    `params` (e.g. a different cleanup pre-pass).
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    kind: Literal["predicate", "ranker"]
    params: dict[str, Any] = Field(default_factory=dict)


class FileChange(BaseModel):
    """One file touched by a PR."""

    model_config = ConfigDict(extra="forbid")

    path: str
    additions: int = 0
    deletions: int = 0
    status: str = ""  # "added" | "modified" | "removed" | "renamed" | ...


class RawPR(BaseModel):
    """One merged PR, mechanically scraped from GitHub."""

    model_config = ConfigDict(extra="forbid")

    pr_number: int
    title: str
    body: str = ""
    merged_at: datetime
    merge_sha: str
    parent_sha: str
    author: str
    labels: list[str] = Field(default_factory=list)
    files_changed: list[FileChange] = Field(default_factory=list)
    additions_total: int = 0
    deletions_total: int = 0
    commits_count: int = 0
    url: str


class AggregationManifest(BaseModel):
    """Sibling of `prs.jsonl`. Records exactly what was asked for."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    window_start: datetime
    window_end: datetime
    window_days: int
    github_query: str
    fetched_at: datetime
    total_prs_returned: int
    scraper_version: str = SCHEMA_VERSION
    repo: str = "vllm-project/vllm"


# ── Snapshot pin / bench spec / run report ────────────────────────────


# Default hours of buffer between the snapshot SHA and the earliest
# in-scope filtered PR's merge time. Overridable per-run.
SNAPSHOT_BUFFER_HOURS = 24


class SnapshotPin(BaseModel):
    """Which target-repo SHA the discovery pipeline must run against.

    The SHA is chosen so that every PR in the filtered set is "future"
    relative to it. We pick the latest PR merged at least
    `buffer_hours` before the earliest filtered PR's `merged_at`, and
    use its `merge_sha`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    window_id: str
    view_id: str
    snapshot_sha: str = Field(
        min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$"
    )
    snapshot_pr_number: int
    snapshot_merged_at: datetime
    earliest_in_view_pr_number: int
    earliest_in_view_merged_at: datetime
    buffer_hours: int = SNAPSHOT_BUFFER_HOURS
    n_view: int  # filtered PR count anchored against
    rationale: str = Field(min_length=1)
    pinned_at: datetime


class BenchSpecConfig(BaseModel):
    """Free-form notes rendered into the bench spec.

    Repo-bench does not own observability knobs — the bench module
    chooses workload, model, and config. Anything repo-bench wants to
    surface to that module goes in `notes`.
    """

    model_config = ConfigDict(extra="forbid")

    notes: str = ""


class BenchSpec(BaseModel):
    """Spec handed off to the observability bench module."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    window_id: str
    view_id: str
    snapshot: SnapshotPin
    config: BenchSpecConfig
    output_naming_template: str = "<timestamp>_<config>_repo-<short_sha>"
    failure_protocol: str = (
        "If the SHA won't build/run, surface the failure with evidence. "
        "Do NOT silently substitute a different SHA — substitution breaks "
        "the experimental contract."
    )
    sync_note: str = (
        "The repo bench also pins the same SHA on its own side; "
        "the SHA is the synchronization point between modules."
    )
    written_at: datetime


class StageReport(BaseModel):
    """One stage of the benchmark run, summarized for the report."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "aggregate", "filter", "fetch-diffs", "snapshot", "workloads",
        "bench-spec", "match",
    ]
    status: Literal["done", "skipped", "failed"]
    detail: str = ""
    counts: dict[str, int] = Field(default_factory=dict)
    duration_s: float | None = None


class RunReport(BaseModel):
    """End-to-end run summary.

    JSON canonical at `runs/repo_bench/<run_id>/report.json`,
    MD rendered alongside at `.../report.md`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    run_id: str
    window_id: str
    view_id: str
    started_at: datetime
    finished_at: datetime
    stages: list[StageReport]
    snapshot: SnapshotPin | None = None
    bench_spec_path: str | None = None
    next_steps: list[str] = Field(default_factory=list)
