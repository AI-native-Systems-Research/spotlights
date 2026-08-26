"""The per-candidate usage manifest, written beside `apply.patch`.

`apply` spends real money — one `claude -p` session per candidate, up to 40
turns and 30 minutes — and reported none of it. This module turns that
session's `AgentUsage` into the sibling `manifest.json`.

Deliberately a *sibling* rather than a block in `run_manifest.json`: apply runs
after the run, against a repo state the run never analyzed, and can run many
times over one run's candidates. Folding it in would mutate a published
artifact and make a run's cost depend on how often someone applied its
candidates afterwards.

Field-for-field parity with `run_manifest.json` wherever the two describe the
same thing, so one reader parses both — `target`, `models_used`, `cost`, and
`external_cost` are the run manifest's own models. `spotlights` and `timing`
are local minimal models: `RunManifestSpotlights` would hard-code
`"pipeline": "deep-research"` into a file describing something that is not that
pipeline, next to a permanently empty `config`, and `RunManifestTiming`'s
`accumulated_duration_s` sums per-module durations across a parallel run, which
with one session is either zero or a duplicate of `wall_clock_s`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.manifest import (
    ModelUsed,
    RunManifestCost,
    RunManifestTarget,
    aggregate_models_used,
    cost_block,
)
from spotlights_engine.costing.rates import (
    ModelRate,
    compute_cost,
    load_external_rates,
    load_rates,
)
from spotlights_engine.costing.records import UsageRecord
from spotlights_engine.costing.usage import AgentUsage

MANIFEST_NAME = "manifest.json"


class ApplySpotlights(BaseModel):
    """Which engine produced this patch.

    Not `RunManifestSpotlights`: its `pipeline` and `config` describe a run's
    stage-prompt fingerprints and module filter, neither of which exists here.
    """

    model_config = ConfigDict(extra="forbid")

    commit_sha: str = ""


class ApplyTiming(BaseModel):
    """Not `RunManifestTiming`: `accumulated_duration_s` is a parallel-run sum.

    `api_time_s` can equal `wall_clock_s` exactly. That means the stream
    reported no API duration and the wall clock stood in — not that the session
    spent 100% of its time in the model.
    """

    model_config = ConfigDict(extra="forbid")

    wall_clock_s: float = 0.0
    api_time_s: float = 0.0


class ApplyUsageManifest(BaseModel):
    """What one candidate's apply cost.

    No `run_id`: `candidate_id` plus `module_qualified_name` identify this
    record, and borrowing the originating run's id would imply this file is
    part of that run's accounting. No `outputs`: what apply produced is the
    sibling files themselves.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    module_qualified_name: str
    date: str
    target: RunManifestTarget
    spotlights: ApplySpotlights
    models_used: list[ModelUsed] = Field(default_factory=list)
    total_tokens: int = 0
    cost: RunManifestCost
    external_cost: RunManifestCost | None = None
    timing: ApplyTiming
    notes: str = ""


def build_apply_usage_manifest(
    *,
    candidate_id: str,
    module_qualified_name: str,
    date: str,
    objective: str,
    target_commit_sha: str,
    repo_url: str,
    spotlights_commit_sha: str,
    usage: AgentUsage | None,
    duration_s: float,
    agent_error: str | None,
    rates: dict[str, ModelRate] | None = None,
    external_rates: dict[str, ModelRate] | None = None,
) -> ApplyUsageManifest:
    """Assemble the manifest. Touches neither the filesystem nor git.

    Every input is a plain value the caller already has — the git lookups stay
    in `api.py`, which resolves them once per invocation rather than once per
    candidate. `rates`/`external_rates` exist so tests can pin a table; in
    production both are None and the bundled tables load.

    Exactly one `UsageRecord`, at `role="one_shot_apply"`. There is one
    `claude -p` call per candidate and the agent writes both the edits and
    `CHANGE-SUMMARY.md` inside it, so one record accounts for 100% of the model
    spend behind this directory — it is complete, not a sample. Splitting it by
    phase is a non-goal: the totals come from the terminal `result` event, which
    carries no per-turn breakdown, and prompt-cache reads are billed against a
    context earlier turns built, so charging them to the reading turn
    misattributes shared cost. If it is ever revisited, no model here changes —
    `aggregate_models_used` already groups by `(provider, model, role)`.
    """
    notes: list[str] = []

    records: list[UsageRecord] = []
    if usage is None:
        # No terminal `result` event — which is exactly what a `--wallclock`
        # kill produces, since the timeout severs the CLI's exit handshake
        # after the work is already done. Mirrors `run_manifest.json`: no
        # record, stable field names, reason in `notes`.
        notes.append(
            f"no usage records found: {agent_error}"
            if agent_error
            else "no usage records found; the agent session reported no usage totals"
        )
    else:
        records.append(
            UsageRecord.from_usage(
                usage,
                step="one_shot_apply",
                module_qualified_name=module_qualified_name,
                session_index=1,
                invocation_index=0,
                # The run steps use a label like "iteration-3:claude"; the
                # candidate id is apply's equivalent.
                invocation_id=candidate_id,
                cli="claude",
                role="one_shot_apply",
                # No `fallback_model`: an invented name would silently price at
                # another model's rate. Unresolved falls back to the CLI family
                # and surfaces as an unpriced `anthropic:claude`.
                fallback_api_time_s=duration_s,
            )
        )

    # Accounting must never cost a patch. A missing table (`OSError`), malformed
    # JSON (`JSONDecodeError`), or a bad rate row (pydantic `ValidationError`)
    # — the latter two both `ValueError` subclasses — zero the cost blocks
    # instead of turning a successful 25-minute apply into a skip.
    try:
        contracted = compute_cost(records, load_rates() if rates is None else rates)
        external = compute_cost(
            records,
            load_external_rates() if external_rates is None else external_rates,
            source="public-api-rate-table",
        )
    except (OSError, ValueError) as exc:
        notes.append(f"cost unavailable: could not load the rate tables: {exc}")
        contracted = compute_cost([], {})
        external = compute_cost([], {}, source="public-api-rate-table")

    if contracted.unpriced_models:
        notes.append(
            "unpriced models excluded from cost: " + ", ".join(contracted.unpriced_models)
        )
    if external.unpriced_models:
        notes.append(
            "external: unpriced models excluded from cost: "
            + ", ".join(external.unpriced_models)
        )
    # No "target commit unavailable" counterpart: `require_git_repo` has already
    # failed the run if HEAD could not be resolved, so it is always populated.
    if not repo_url:
        notes.append("target repo URL unavailable")
    if not spotlights_commit_sha:
        notes.append("Spotlights commit unavailable")

    return ApplyUsageManifest(
        candidate_id=candidate_id,
        module_qualified_name=module_qualified_name,
        date=date,
        target=RunManifestTarget(
            repo_url=repo_url,
            commit_sha=target_commit_sha,
            # The bare goal, matching the run manifest. APPLY-NOTES.md already
            # renders "goal (direction: minimize)".
            objective=objective,
        ),
        spotlights=ApplySpotlights(commit_sha=spotlights_commit_sha),
        models_used=aggregate_models_used(records),
        total_tokens=sum(record.total_tokens for record in records),
        cost=cost_block(contracted),
        # Always populated — apply has no path that prices one table and not the
        # other. The type stays nullable for shape-parity with `RunManifest`, so
        # a consumer written against one file's cost blocks reads the other's.
        external_cost=cost_block(external),
        timing=ApplyTiming(
            wall_clock_s=duration_s,
            api_time_s=sum(record.api_time_s or 0.0 for record in records),
        ),
        notes="; ".join(part for part in notes if part),
    )


__all__ = [
    "MANIFEST_NAME",
    "ApplySpotlights",
    "ApplyTiming",
    "ApplyUsageManifest",
    "build_apply_usage_manifest",
]
