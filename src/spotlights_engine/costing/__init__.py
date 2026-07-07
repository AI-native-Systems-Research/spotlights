"""Per-invocation usage capture and contracted-rate costing.

This package is dependency-free within the engine (only pydantic) so every
step package and the manager can import it without cycles.

- `usage`: the `AgentUsage` capture shape plus tolerant parsers for the
  Claude (`stream-json` / `--output-format json`) and Codex (`--json`) CLI
  output formats.
- `records`: the durable per-invocation `UsageRecord` written next to each
  step's output; the run manifest is a pure aggregation over these.
- `rates`: the `(provider, model)` -> per-token contracted rate table and
  `compute_cost`. Billing never uses the CLI's `total_cost_usd`.
"""

from spotlights_engine.costing.manifest import (
    ModelUsed,
    RunManifest,
    UsageTotals,
    aggregate_models_used,
    build_run_manifest,
)
from spotlights_engine.costing.rates import (
    RATES_ENV_VAR,
    CostSummary,
    ModelRate,
    compute_cost,
    load_rates,
)
from spotlights_engine.costing.records import (
    PROVIDER_FOR_CLI,
    UsageRecord,
    UsageStep,
)
from spotlights_engine.costing.usage import (
    AgentUsage,
    CliUsage,
    claude_usage_from_payload,
    claude_usage_from_stream,
    codex_usage_from_stream,
)

__all__ = [
    "PROVIDER_FOR_CLI",
    "RATES_ENV_VAR",
    "AgentUsage",
    "CliUsage",
    "CostSummary",
    "ModelRate",
    "ModelUsed",
    "RunManifest",
    "UsageRecord",
    "UsageStep",
    "UsageTotals",
    "aggregate_models_used",
    "build_run_manifest",
    "claude_usage_from_payload",
    "claude_usage_from_stream",
    "codex_usage_from_stream",
    "compute_cost",
    "load_rates",
]
