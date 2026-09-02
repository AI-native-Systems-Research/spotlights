"""Public per-run manifest assembled from durable usage records."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.rates import (
    ByModelCost,
    CostCoverage,
    CostSummary,
    display_model_id,
)
from spotlights_engine.costing.records import UsageRecord


class UsageTotals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_create: int = 0


class ModelUsed(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    provider: str
    role: str
    usage: UsageTotals


class RunManifestTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repo_url: str = ""
    commit_sha: str = ""
    objective: str = ""


class RunManifestSpotlights(BaseModel):
    model_config = ConfigDict(extra="forbid")

    commit_sha: str = ""
    pipeline: str = "deep-research"
    config: dict[str, Any] = Field(default_factory=dict)


class RunManifestCost(BaseModel):
    """Cost block in the run manifest.

    `amount_usd` is a numeric sum of everything that got a rate row; when
    `priced_token_share < 1.0` it is a PARTIAL figure and readers must consult
    `coverage.unpriced_models` / `by_model` to see what was left out. The name
    is stable across full and partial runs to keep existing consumers working;
    partial-ness is signalled structurally via the sibling `priced_token_share`
    plus `coverage`, not by swapping the field.
    """

    model_config = ConfigDict(extra="forbid")

    amount_usd: float = 0.0
    priced_token_share: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = "contracted-rate-table"
    rate_note: str = ""
    coverage: CostCoverage = Field(default_factory=CostCoverage)
    by_model: list[ByModelCost] = Field(default_factory=list)


def cost_block(summary: CostSummary) -> RunManifestCost:
    """Project a `CostSummary` onto the manifest's cost block.

    Drops `CostSummary.unpriced_models`, which duplicates
    `coverage.unpriced_models`; the run manifest surfaces it in `notes`
    instead. Shared with `one_shot_apply`'s sibling manifest so both files'
    cost blocks stay one shape with one reader.
    """
    return RunManifestCost(
        amount_usd=summary.amount_usd,
        priced_token_share=summary.priced_token_share,
        source=summary.source,
        rate_note=summary.rate_note,
        coverage=summary.coverage,
        by_model=list(summary.by_model),
    )


class RunManifestTiming(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wall_clock_s: float = 0.0
    accumulated_duration_s: float = 0.0
    api_time_s: float = 0.0


class RunManifestOutputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidates_path: str = ""
    num_candidates: int = 0
    module_status: dict[str, int] = Field(default_factory=dict)


class RunManifestModelsRequested(BaseModel):
    """What the run *asked* each agent CLI for, per `models.yaml` / `--*-model`.

    Distinct from `models_used`, which is parsed back out of CLI output and says
    what actually ran. An empty string means the engine passed no `--model` and
    the CLI chose for itself, so the two can legitimately differ.
    """

    model_config = ConfigDict(extra="forbid")

    claude: str = ""
    codex: str = ""


class RunManifestRunConfig(BaseModel):
    """The knobs that shaped this run's cost and output volume.

    Purely descriptive, like `models_requested`: written for whoever reads the
    results later, never read back by the engine. Deliberately *not* part of
    `spotlights.config` -- that block is the `--resume` fingerprint, so adding
    keys to it would invalidate every run directory already on disk.

    Without this, results are individually correct but not comparable. Two runs
    on one target at the same module count can differ several-fold in spend
    purely because one had deep research off, and nothing else in the manifest
    distinguishes them.

    Optional on `RunManifest` rather than default-constructed: field defaults
    here are zero/False, not the engine's real defaults, so a block nobody
    populated would assert "this run used 0 review iterations". A caller that
    cannot supply the values leaves it `None`, which reaches the JSON as
    `"run_config": null` -- `write_run_manifest` dumps without `exclude_none`,
    so the key is present and empty rather than gone. `external_cost` reads as
    `null` rather than $0 for the same reason.
    """

    model_config = ConfigDict(extra="forbid")

    review_iterations: int = 0
    max_findings_per_module: int = 0
    include_candidate_hotspots: bool = False
    enable_claude_search: bool = False
    enable_deep_research: bool = False
    max_parallel_sessions: int = 0
    max_parallel_pairs: int = 0
    max_parallel_candidates: int = 0


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    date: str
    target: RunManifestTarget
    spotlights: RunManifestSpotlights
    models_requested: RunManifestModelsRequested = Field(
        default_factory=RunManifestModelsRequested
    )
    run_config: RunManifestRunConfig | None = None
    models_used: list[ModelUsed] = Field(default_factory=list)
    total_tokens: int = 0
    cost: RunManifestCost
    external_cost: RunManifestCost | None = None
    timing: RunManifestTiming
    outputs: RunManifestOutputs
    notes: str = ""


def aggregate_models_used(records: list[UsageRecord]) -> list[ModelUsed]:
    grouped: dict[tuple[str, str, str], UsageTotals] = defaultdict(UsageTotals)
    for record in records:
        model = display_model_id(record.model) if record.model else record.cli
        key = (record.provider, model, record.role)
        totals = grouped[key]
        totals.input += record.input
        totals.output += record.output
        totals.cache_read += record.cache_read
        totals.cache_create += record.cache_create

    return [
        ModelUsed(provider=provider, model=model, role=role, usage=totals)
        for (provider, model, role), totals in sorted(grouped.items())
    ]


def build_run_manifest(
    *,
    run_id: str,
    date: str,
    objective: str,
    provenance: dict[str, Any],
    config_fingerprint: dict[str, Any],
    records: list[UsageRecord],
    cost: CostSummary,
    external_cost: CostSummary | None = None,
    wall_clock_s: float,
    accumulated_duration_s: float = 0.0,
    candidates_path: str,
    num_candidates: int,
    module_status: dict[str, int],
    notes: list[str],
    models_requested: RunManifestModelsRequested | None = None,
    run_config: RunManifestRunConfig | None = None,
) -> RunManifest:
    note_parts = list(notes)
    if cost.unpriced_models:
        note_parts.append(
            "unpriced models excluded from cost: "
            + ", ".join(cost.unpriced_models)
        )
    if external_cost is not None and external_cost.unpriced_models:
        note_parts.append(
            "external: unpriced models excluded from cost: "
            + ", ".join(external_cost.unpriced_models)
        )
    if not provenance.get("target_commit_sha"):
        note_parts.append("target commit unavailable")
    if not provenance.get("spotlights_commit_sha"):
        note_parts.append("Spotlights commit unavailable")
    if not provenance.get("repo_url"):
        note_parts.append("target repo URL unavailable")

    return RunManifest(
        run_id=run_id,
        date=date,
        target=RunManifestTarget(
            repo_url=str(provenance.get("repo_url") or ""),
            commit_sha=str(provenance.get("target_commit_sha") or ""),
            objective=objective,
        ),
        spotlights=RunManifestSpotlights(
            commit_sha=str(provenance.get("spotlights_commit_sha") or ""),
            pipeline="deep-research",
            config=dict(config_fingerprint),
        ),
        models_requested=models_requested or RunManifestModelsRequested(),
        run_config=run_config,
        models_used=aggregate_models_used(records),
        total_tokens=sum(record.total_tokens for record in records),
        cost=cost_block(cost),
        external_cost=(cost_block(external_cost) if external_cost is not None else None),
        timing=RunManifestTiming(
            wall_clock_s=wall_clock_s,
            accumulated_duration_s=accumulated_duration_s,
            api_time_s=sum(record.api_time_s or 0.0 for record in records),
        ),
        outputs=RunManifestOutputs(
            candidates_path=candidates_path,
            num_candidates=num_candidates,
            module_status=dict(module_status),
        ),
        notes="; ".join(part for part in note_parts if part),
    )


__all__ = [
    "ModelUsed",
    "RunManifest",
    "UsageTotals",
    "RunManifestModelsRequested",
    "RunManifestRunConfig",
    "aggregate_models_used",
    "build_run_manifest",
    "cost_block",
]
