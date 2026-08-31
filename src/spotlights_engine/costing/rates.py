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
    """Which rate keys got dollarized.

    `amount_usd` at the parent level is a full-run figure only when
    `unpriced_models` is empty; otherwise it is partial and `unpriced_models`
    names the rate keys that were skipped. Token share sits on the parent as
    `priced_token_share` — it is a top-level property, not a coverage detail.
    """

    model_config = ConfigDict(extra="forbid")

    priced_models: list[str] = Field(default_factory=list)
    unpriced_models: list[str] = Field(default_factory=list)


class ByModelCost(BaseModel):
    """One row of the `by_model` breakdown: a `(provider, model, role)` group's
    rate-lookup result, its dollar contribution, and its share of the run's
    tokens.

    Token counts themselves live on `models_used` (the same
    `(provider, model, role)` grouping) — this row keeps only `priced_token_share`
    (this group's total tokens divided by the run's total tokens) so a reader
    can compare rows at a glance without cross-referencing `models_used`.
    `amount_usd` is 0.0 when `priced` is false.
    """

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    role: str
    rate_key: str
    priced: bool
    amount_usd: float = Field(default=0.0, ge=0.0)
    priced_token_share: float = Field(default=0.0, ge=0.0, le=1.0)


class CostSummary(BaseModel):
    """Result of `compute_cost` — feeds the run manifest's `cost` block.

    `priced_token_share` (0.0–1.0) is the fraction of this run's tokens that
    landed on a priced group. 1.0 means every model got a rate row and
    `amount_usd` is the full run cost; anything less means `amount_usd` is a
    partial figure and `coverage.unpriced_models` names what was left out.
    """

    model_config = ConfigDict(extra="forbid")

    amount_usd: float = 0.0
    priced_token_share: float = Field(default=0.0, ge=0.0, le=1.0)
    source: Literal["contracted-rate-table", "litellm-proxy-log", "public-api-rate-table"] = (
        "contracted-rate-table"
    )
    rate_note: str = ""
    unpriced_models: list[str] = Field(default_factory=list)
    coverage: CostCoverage = Field(default_factory=CostCoverage)
    by_model: list[ByModelCost] = Field(default_factory=list)


def _build_canonical_rate_map(
    items: Iterable[tuple[str, ModelRate]],
) -> dict[str, ModelRate]:
    """Build a canonical-key rate map, raising on canonical-key collision.

    Two raw keys that canonicalize to the same string (e.g.
    `anthropic:aws/claude-opus-5` and `anthropic:claude-opus-5` — both legal,
    both naming the same rate row) would otherwise silently overwrite in a
    plain dict comprehension, and pricing would land on whichever entry JSON
    iteration happened to visit last. Raise instead: pricing is per canonical
    model, so a duplicate is unambiguously an author error.
    """
    result: dict[str, ModelRate] = {}
    raw_by_canonical: dict[str, str] = {}
    for raw_key, rate in items:
        canonical = _canonical_rate_key(raw_key)
        prior = raw_by_canonical.get(canonical)
        if prior is not None:
            raise ValueError(
                f"duplicate rate-table entry: {prior!r} and {raw_key!r} both "
                f"canonicalize to {canonical!r}"
            )
        result[canonical] = rate
        raw_by_canonical[canonical] = raw_key
    return result


def _load_rates_from(path: Path | None, env_var: str, bundled: Path) -> dict[str, ModelRate]:
    """Load a rate table keyed by `"provider:model"`.

    Precedence: explicit `path` > `env_var` env var > the `bundled` default
    table. Keys starting with `_` are comments. Model portions are
    canonicalized (context tag + LiteLLM route prefix stripped) so a table
    keyed `anthropic:aws/claude-opus-5` and one keyed `anthropic:claude-opus-5`
    are equivalent — pricing is per model, not per route. A table containing
    two entries that canonicalize to the same key is rejected rather than
    silently applying only one of them.
    """
    if path is None:
        env_path = os.environ.get(env_var)
        path = Path(env_path) if env_path else bundled
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return _build_canonical_rate_map(
            (key, ModelRate.model_validate(value))
            for key, value in payload.items()
            if not key.startswith("_")
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


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
    return _load_rates_from(path, EXTERNAL_RATES_ENV_VAR, _BUNDLED_EXTERNAL_RATES_PATH)


# Context-window variant tag a CLI appends to the model id, e.g. the "[1m]" in
# "aws/claude-opus-4-8[1m]". Pricing is per model, not per context window, so we
# strip it before the rate lookup — all context variants share one rate row.
_CONTEXT_TAG_RE = re.compile(r"\[[^\]]*\]$")

# LiteLLM route prefix at the start of a model id, e.g. the "aws/" in
# "aws/claude-opus-5" or the "azure/" in "azure/gpt-5.5". Pricing is per model,
# not per route (the same Opus row prices both a bedrock and a direct-API call),
# so we strip it before the rate lookup — a CLI reporting plain "claude-opus-5"
# and one reporting "aws/claude-opus-5" hit the same row. The set is a small
# allowlist of LiteLLM's own custom_llm_provider labels rather than a generic
# "strip anything before /" — some legitimate model ids contain a slash
# (e.g. HuggingFace org/model paths) and we don't want to eat those. The `+`
# collapses stacked LiteLLM routes: LiteLLM's OpenRouter provider emits
# "openrouter/anthropic/claude-opus-5", which must strip both segments.
_ROUTE_PREFIX_RE = re.compile(
    r"^(?:(?:aws|bedrock|azure|azure_ai|vertex|vertex_ai|"
    r"openrouter|anthropic|openai|gemini)/)+"
)


def canonical_model_id(model: str) -> str:
    """Strip the context-window tag and LiteLLM route prefix from a model id.

    Both variants map to the same rate row: pricing is per model, not per
    context window or route. Called on both sides of the lookup (record model
    id + rate-table key) so `aws/claude-opus-5`, `claude-opus-5`, and
    `aws/claude-opus-5[1m]` all resolve to the same key.

    A degenerate id that strips to empty (e.g. `"[1m]"` on its own, or
    `"aws/"` — malformed, not emitted by any real CLI) is returned unchanged
    so the bad id surfaces in `unpriced_models` under its own visibly-broken
    key rather than colliding silently on `"provider:"`.
    """
    stripped = _ROUTE_PREFIX_RE.sub("", _CONTEXT_TAG_RE.sub("", model))
    return stripped or model


def display_model_id(model: str) -> str:
    """Strip only the LiteLLM route prefix from a model id, keeping the
    context-window tag.

    Used for `models_used[*].model` and `cost.by_model[*].model` — the
    manifest's display side. Same rate row prices bedrock and direct-API
    variants (`aws/claude-opus-5` vs `claude-opus-5`), so the route prefix
    is noise for a reader scanning what actually ran; but the context tag
    (`[1m]`) distinguishes real product variants a reader wants to see. Rate
    lookup still uses `canonical_model_id` (strips both) — this pair keeps
    the display readable while the lookup stays route-and-context-agnostic.
    """
    stripped = _ROUTE_PREFIX_RE.sub("", model)
    return stripped or model


def _canonical_rate_key(key: str) -> str:
    """Canonicalize a rate-table key `"provider:model"` — provider left alone,
    model canonicalized. A key with no colon (malformed) is returned as-is so
    Pydantic validation surfaces the error at the value level.
    """
    provider, sep, model = key.partition(":")
    if not sep:
        return key
    return f"{provider}:{canonical_model_id(model)}"


def _rate_key(record: UsageRecord) -> tuple[str, bool]:
    """`(provider:model, resolved)`; unresolved models fall back to the CLI family."""
    if record.model:
        return f"{record.provider}:{canonical_model_id(record.model)}", True
    return f"{record.provider}:{record.cli}", False


def _group_key(record: UsageRecord) -> tuple[str, str, str]:
    """`(provider, model_or_cli, role)` — same grouping as `aggregate_models_used`.

    Kept in lockstep so `cost.by_model` lines up 1:1 with `models_used`. Uses
    `display_model_id` (strips route prefix, keeps context tag) so the two
    sections of the manifest carry the same reader-facing model string; the
    `rate_key` on the same row uses `canonical_model_id` (strips both) for the
    rate-table lookup.
    """
    model = display_model_id(record.model) if record.model else record.cli
    return (record.provider, model, record.role)


def compute_cost(
    records: Iterable[UsageRecord],
    rates: dict[str, ModelRate],
    source: Literal[
        "contracted-rate-table", "litellm-proxy-log", "public-api-rate-table"
    ] = "contracted-rate-table",
) -> CostSummary:
    # Canonicalize table keys on entry so a caller-supplied dict keyed with a
    # LiteLLM route prefix (e.g. `anthropic:aws/claude-opus-5`) still matches
    # records whose model id the CLI reported without the prefix. Dicts coming
    # from `load_rates`/`load_external_rates` are already canonical, so this
    # is a no-op there; the rebuild is cheap and shields internal API callers.
    # A collision (two entries canonicalizing to the same key) is rejected so
    # a misconfigured caller doesn't silently price against the last-visited row.
    rates = _build_canonical_rate_map(rates.items())

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

    # Compute the run's token total once so each by_model row can carry its
    # share directly (a reader gets the split at a glance without dividing
    # against a separate models_used table).
    def _bucket_tokens(bucket: dict[str, int | float | str | bool]) -> int:
        return (
            int(bucket["input"])
            + int(bucket["output"])
            + int(bucket["cache_read"])
            + int(bucket["cache_create"])
        )

    total_tokens = sum(_bucket_tokens(bucket) for bucket in grouped.values())
    priced_tokens = sum(_bucket_tokens(bucket) for bucket in grouped.values() if bucket["priced"])

    by_model = [
        ByModelCost(
            provider=provider,
            model=model,
            role=role,
            rate_key=str(bucket["rate_key"]),
            priced=bool(bucket["priced"]),
            amount_usd=float(bucket["amount_usd"]),
            priced_token_share=(_bucket_tokens(bucket) / total_tokens if total_tokens else 0.0),
        )
        for (provider, model, role), bucket in sorted(grouped.items())
    ]

    priced_token_share = (priced_tokens / total_tokens) if total_tokens else 0.0
    coverage = CostCoverage(
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
        note_parts.append("PARTIAL cost — no contracted rate for: " + ", ".join(sorted(unpriced)))
    if unresolved:
        note_parts.append(
            f"{unresolved} invocation(s) had an unresolved model id "
            "(priced by CLI-family fallback key when available)"
        )

    return CostSummary(
        amount_usd=total,
        priced_token_share=priced_token_share,
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
    "canonical_model_id",
    "compute_cost",
    "display_model_id",
    "load_external_rates",
    "load_rates",
]
