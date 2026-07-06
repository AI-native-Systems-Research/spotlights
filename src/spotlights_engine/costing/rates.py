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
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.costing.records import UsageRecord

RATES_ENV_VAR = "SPOTLIGHTS_RATES_FILE"

_BUNDLED_RATES_PATH = Path(__file__).parent / "rates.json"


class ModelRate(BaseModel):
    """Per-token contracted rates for one `(provider, model)`."""

    model_config = ConfigDict(extra="forbid")

    input: float = Field(ge=0)
    output: float = Field(ge=0)
    cache_read: float = Field(ge=0)
    cache_create: float = Field(ge=0)
    unit: Literal["per_token"] = "per_token"
    note: str = ""


class CostSummary(BaseModel):
    """Result of `compute_cost` — feeds the run manifest's `cost` block."""

    model_config = ConfigDict(extra="forbid")

    amount_usd: float = 0.0
    source: Literal["contracted-rate-table", "litellm-proxy-log"] = (
        "contracted-rate-table"
    )
    rate_note: str = ""
    unpriced_models: list[str] = Field(default_factory=list)


def load_rates(path: Path | None = None) -> dict[str, ModelRate]:
    """Load the rate table keyed by `"provider:model"`.

    Precedence: explicit `path` > `SPOTLIGHTS_RATES_FILE` env var > the
    checked-in placeholder table. Keys starting with `_` are comments.
    """
    if path is None:
        env_path = os.environ.get(RATES_ENV_VAR)
        path = Path(env_path) if env_path else _BUNDLED_RATES_PATH
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        key: ModelRate.model_validate(value)
        for key, value in payload.items()
        if not key.startswith("_")
    }


def _rate_key(record: UsageRecord) -> tuple[str, bool]:
    """`(provider:model, resolved)`; unresolved models fall back to the CLI family."""
    if record.model:
        return f"{record.provider}:{record.model}", True
    return f"{record.provider}:{record.cli}", False


def compute_cost(
    records: Iterable[UsageRecord], rates: dict[str, ModelRate]
) -> CostSummary:
    total = 0.0
    applied: dict[str, ModelRate] = {}
    unpriced: set[str] = set()
    unresolved = 0

    for record in records:
        key, resolved = _rate_key(record)
        if not resolved:
            unresolved += 1
        rate = rates.get(key)
        if rate is None:
            unpriced.add(key)
            continue
        total += (
            record.input * rate.input
            + record.output * rate.output
            + record.cache_read * rate.cache_read
            + record.cache_create * rate.cache_create
        )
        applied[key] = rate

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
        source="contracted-rate-table",
        rate_note="; ".join(note_parts),
        unpriced_models=sorted(unpriced),
    )


__all__ = [
    "RATES_ENV_VAR",
    "CostSummary",
    "ModelRate",
    "compute_cost",
    "load_rates",
]
