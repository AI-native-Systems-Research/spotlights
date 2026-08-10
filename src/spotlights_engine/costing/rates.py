"""Contracted rate table and the tokens-x-rate cost computation.

Cost for a run = Σ over usage records of
`input*r.input + output*r.output + cache_read*r.cache_read + cache_create*r.cache_create`.
This is only correct because the four `UsageRecord` buckets are disjoint by
construction (see `costing.usage`). Models without a contracted rate are
excluded from the total and surfaced on `CostSummary.unpriced_models` — we
never silently substitute list price.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.records import UsageRecord

RATES_ENV_VAR = "SPOTLIGHTS_RATES_FILE"
EXTERNAL_RATES_ENV_VAR = "SPOTLIGHTS_EXTERNAL_RATES_FILE"

_BUNDLED_RATES_PATH = Path(__file__).parent / "rates.json"
_BUNDLED_EXTERNAL_RATES_PATH = Path(__file__).parent / "external_rates.json"


class ModelRate(BaseModel):
    """Per-token contracted rates for one `(provider, model)`."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_create: float = Field(ge=0)
    unit: Literal["per_token"] = "per_token"
    note: str = ""


class CostCoverage(BaseModel):
    """Which rate keys got dollarized and what share of tokens they cover.

    `priced_token_share` is total-tokens of priced records over total-tokens of
    all records (0.0 when there were no records). 1.0 means every model this
    run used had a rate row and `amount_usd` is the full run cost; anything
    less means `amount_usd` is a partial figure and `unpriced_models` names the
    rate keys that were skipped.
    """

    model_config = ConfigDict(extra="forbid")

    priced_token_share: float = Field(default=0.0, ge=0.0, le=1.0)
    priced_models: list[str] = Field(default_factory=list)
    unpriced_models: list[str] = Field(default_factory=list)


class ByModelCost(BaseModel):
    """One row of the `by_model` breakdown: a `(provider, model, role)` group's
    token totals, whether it got a rate, and its dollar contribution.

    `amount_usd` is 0.0 when `priced` is false — an unpriced group contributes
    nothing to the run total. Rows are keyed as `provider:model` for the rate
    lookup; the `role` column is not part of the rate key but is preserved so
    the breakdown lines up 1:1 with `models_used`.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    role: str
    rate_key: str
    priced: bool
    amount_usd: float = Field(default=0.0, ge=0.0)
    input: int = Field(default=0, ge=0)
    output: int = Field(default=0, ge=0)
    cache_read: int = Field(default=0, ge=0)
    cache_create: int = Field(default=0, ge=0)


class CostSummary(BaseModel):
    """Result of `compute_cost` — feeds the run manifest's `cost` block."""

    model_config = ConfigDict(extra="forbid")

    amount_usd: float = 0.0
    source: Literal[
        "contracted-rate-table", "litellm-proxy-log", "public-api-rate-table"
    ] = "contracted-rate-table"
    rate_note: str = ""
    unpriced_models: list[str] = Field(default_factory=list)
    coverage: CostCoverage = Field(default_factory=CostCoverage)
    by_model: list[ByModelCost] = Field(default_factory=list)


def _load_rates_from(
    path: Path | None, env_var: str, bundled: Path
) -> dict[str, ModelRate]:
    """Load a rate table keyed by `"provider:model"`.

    Precedence: explicit `path` > `env_var` env var > the `bundled` default
    table. Keys starting with `_` are comments.
    """
    if path is None:
        env_path = os.environ.get(env_var)
        path = Path(env_path) if env_path else bundled
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: ModelRate.model_validate(value)
        for key, value in payload.items()
        if not key.startswith("_")
    }


def load_rates(path: Path | None = None) -> dict[str, ModelRate]:
    """Load the contracted (LiteLLM-proxy) rate table keyed by `"provider:model"`.

    Precedence: explicit `path` > `SPOTLIGHTS_RATES_FILE` env var > the
    checked-in contracted default table. Keys starting with `_` are comments.
    """
    return _load_rates_from(path, RATES_ENV_VAR, _BUNDLED_RATES_PATH)


def load_external_rates(path: Path | None = None) -> dict[str, ModelRate]:
    """Load the external (public list-price) rate table keyed by `"provider:model"`.

    Precedence: explicit `path` > `SPOTLIGHTS_EXTERNAL_RATES_FILE` env var > the
    checked-in external default table. Keys starting with `_` are comments.
    """
    return _load_rates_from(
        path, EXTERNAL_RATES_ENV_VAR, _BUNDLED_EXTERNAL_RATES_PATH
    )


# Context-window variant tag a CLI appends to the model id, e.g. the "[1m]" in
# "aws/claude-opus-4-8[1m]". Pricing is per model, not per context window, so we
# strip it before the rate lookup — all context variants share one rate row.
_CONTEXT_TAG_RE = re.compile(r"\[[^\]]*\]$")


def _rate_key(record: UsageRecord) -> tuple[str, bool]:
    """`(provider:model, resolved)`; unresolved models fall back to the CLI family."""
    if record.model:
        model = _CONTEXT_TAG_RE.sub("", record.model)
        return f"{record.provider}:{model}", True
    return f"{record.provider}:{record.cli}", False


def _group_key(record: UsageRecord) -> tuple[str, str, str]:
    """`(provider, model_or_cli, role)` — same grouping as `aggregate_models_used`.

    Kept in lockstep so `cost.by_model` lines up 1:1 with `models_used`; the
    context-tag stripping matches `_rate_key` so the group's `rate_key` is the
    exact string we look up in the rate table.
    """
    model = _CONTEXT_TAG_RE.sub("", record.model) if record.model else record.cli
    return (record.provider, model, record.role)


def compute_cost(
    records: Iterable[UsageRecord],
    rates: dict[str, ModelRate],
    source: Literal[
        "contracted-rate-table", "litellm-proxy-log", "public-api-rate-table"
    ] = "contracted-rate-table",
) -> CostSummary:
    # Materialize once so we can iterate twice (grouping + coverage) without
    # forcing the caller to hand us a list.
    record_list = list(records)

    total = 0.0
    applied: dict[str, ModelRate] = {}
    unpriced: set[str] = set()
    unresolved = 0

    # Group records the same way `aggregate_models_used` does so `by_model`
    # lines up 1:1 with `models_used`. Each group's rate_key comes from
    # `_rate_key` (its resolution flag drives the `unresolved` counter below).
    grouped: dict[
        tuple[str, str, str],
        dict[str, int | float | str | bool],
    ] = {}
    for record in record_list:
        key, resolved = _rate_key(record)
        if not resolved:
            unresolved += 1
        rate = rates.get(key)
        contribution = 0.0
        if rate is not None:
            contribution = (
                record.input * rate.input
                + record.output * rate.output
                + record.cache_read * rate.cache_read
                + record.cache_create * rate.cache_create
            )
            total += contribution
            applied[key] = rate
        else:
            unpriced.add(key)

        gk = _group_key(record)
        bucket = grouped.setdefault(
            gk,
            {
                "rate_key": key,
                "priced": rate is not None,
                "amount_usd": 0.0,
                "input": 0,
                "output": 0,
                "cache_read": 0,
                "cache_create": 0,
            },
        )
        bucket["amount_usd"] = float(bucket["amount_usd"]) + contribution
        bucket["input"] = int(bucket["input"]) + record.input
        bucket["output"] = int(bucket["output"]) + record.output
        bucket["cache_read"] = int(bucket["cache_read"]) + record.cache_read
        bucket["cache_create"] = int(bucket["cache_create"]) + record.cache_create

    by_model = [
        ByModelCost(
            provider=provider,
            model=model,
            role=role,
            rate_key=str(bucket["rate_key"]),
            priced=bool(bucket["priced"]),
            amount_usd=float(bucket["amount_usd"]),
            input=int(bucket["input"]),
            output=int(bucket["output"]),
            cache_read=int(bucket["cache_read"]),
            cache_create=int(bucket["cache_create"]),
        )
        for (provider, model, role), bucket in sorted(grouped.items())
    ]

    total_tokens = sum(
        row.input + row.output + row.cache_read + row.cache_create for row in by_model
    )
    priced_tokens = sum(
        row.input + row.output + row.cache_read + row.cache_create
        for row in by_model
        if row.priced
    )
    priced_token_share = (priced_tokens / total_tokens) if total_tokens else 0.0
    coverage = CostCoverage(
        priced_token_share=priced_token_share,
        priced_models=sorted({row.rate_key for row in by_model if row.priced}),
        unpriced_models=sorted(unpriced),
    )

    note_parts: list[str] = []
    if applied:
        note_parts.append(
            "rates applied: "
            + "; ".join(
                f"{key} ({rate.note})" if rate.note else key
                for key, rate in sorted(applied.items())
            )
        )
    if unpriced:
        note_parts.append(
            "PARTIAL cost — no contracted rate for: " + ", ".join(sorted(unpriced))
        )
    if unresolved:
        note_parts.append(
            f"{unresolved} invocation(s) had an unresolved model id "
            "(priced by CLI-family fallback key when available)"
        )

    return CostSummary(
        amount_usd=total,
        source=source,
        rate_note="; ".join(note_parts),
        unpriced_models=sorted(unpriced),
        coverage=coverage,
        by_model=by_model,
    )


__all__ = [
    "EXTERNAL_RATES_ENV_VAR",
    "RATES_ENV_VAR",
    "ByModelCost",
    "CostCoverage",
    "CostSummary",
    "ModelRate",
    "compute_cost",
    "load_external_rates",
    "load_rates",
]
