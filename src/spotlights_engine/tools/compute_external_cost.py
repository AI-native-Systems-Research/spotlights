"""Back-fill the external (public list-price) cost for an existing run manifest.

The engine writes ``external_cost`` into ``run_manifest.json`` for runs made
after that feature landed. Older manifests only carry the contracted
``cost`` block. This tool reconstructs the per-model usage from the manifest's
``models_used`` aggregate and reprices it through the same
``compute_cost`` + ``load_external_rates`` path the orchestrator uses, then
writes ``external_cost.json`` next to the manifest.

Repricing the aggregate is exact: cost is linear in token counts and the four
usage buckets are disjoint, so Σ over aggregated rows equals Σ over the
original per-invocation records.

Examples:
    uv run --no-sync python -m spotlights_engine.tools.compute_external_cost \\
        output/llmd_router_lsf/run_manifest.json

    # custom output path and an override external-rate table
    uv run --no-sync python -m spotlights_engine.tools.compute_external_cost \\
        run_manifest.json --out external_cost.json \\
        --external-rates my_rates.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from spotlights_engine.costing import (
    PROVIDER_FOR_CLI,
    CostSummary,
    UsageRecord,
    UsageStep,
    compute_cost,
    load_external_rates,
)

# Reverse of PROVIDER_FOR_CLI: which CLI family a provider was invoked through.
# Needed to rebuild the CLI-family fallback rate key for unresolved model ids.
_CLI_FOR_PROVIDER: dict[str, str] = {
    provider: cli for cli, provider in PROVIDER_FOR_CLI.items()
}

# Manifest `role` -> a valid `UsageStep` literal. The step never affects cost
# (compute_cost keys only on provider/model/cli + token buckets); it only keeps
# the reconstructed UsageRecord schema-valid.
_STEP_FOR_ROLE: dict[str, UsageStep] = {
    "candidate_discovery": "candidate_discovery",
    "deep_research": "module_deep_research",
    "proposal_from_finding_creator": "proposal_from_finding_creator",
    "agent_proposals": "agent_proposals",
}


def _records_from_models_used(models_used: list[dict[str, Any]]) -> list[UsageRecord]:
    """Rebuild costable usage records from the manifest's aggregated rows.

    Each `models_used` row already stores `model` as `record.model or
    record.cli`, so for CLI families that report no resolvable model id (e.g.
    Codex) the stored model equals the CLI name and the reconstructed rate key
    (`provider:model`) coincides with the original CLI-family fallback key.
    """
    records: list[UsageRecord] = []
    for i, row in enumerate(models_used):
        provider = str(row["provider"])
        cli = _CLI_FOR_PROVIDER.get(provider)
        if cli is None:
            raise ValueError(
                f"models_used[{i}]: unknown provider {provider!r} "
                f"(expected one of {sorted(_CLI_FOR_PROVIDER)})"
            )
        role = str(row["role"])
        usage = row.get("usage") or {}
        records.append(
            UsageRecord(
                step=_STEP_FOR_ROLE.get(role, "candidate_discovery"),
                module_qualified_name=f"reconstructed:{role}",
                invocation_index=i,
                provider=provider,  # type: ignore[arg-type]
                cli=cli,  # type: ignore[arg-type]
                model=str(row["model"]),
                role=role,
                input=int(usage.get("input", 0)),
                output=int(usage.get("output", 0)),
                cache_read=int(usage.get("cache_read", 0)),
                cache_create=int(usage.get("cache_create", 0)),
            )
        )
    return records


def compute_external_cost(
    manifest: dict[str, Any], external_rates_path: Path | None = None
) -> CostSummary:
    """Reprice a run manifest through the external (public list-price) table."""
    models_used = manifest.get("models_used") or []
    if not models_used:
        raise ValueError(
            "manifest has no `models_used`; cannot reconstruct usage to reprice"
        )
    records = _records_from_models_used(models_used)
    return compute_cost(
        records,
        load_external_rates(external_rates_path),
        source="public-api-rate-table",
    )


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "manifest",
        type=Path,
        help="Path to a run_manifest.json to reprice.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Where to write the external cost JSON "
            "(default: external_cost.json next to the manifest)."
        ),
    )
    p.add_argument(
        "--external-rates",
        type=Path,
        default=None,
        help=(
            "Override external rate table (JSON). Defaults to the bundled table "
            "or SPOTLIGHTS_EXTERNAL_RATES_FILE."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    manifest_path: Path = args.manifest
    if not manifest_path.is_file():
        print(f"error: manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    external = compute_external_cost(manifest, args.external_rates)

    contracted = manifest.get("cost") or {}
    contracted_amount = float(contracted.get("amount_usd", 0.0))
    payload: dict[str, Any] = {
        "run_id": manifest.get("run_id", ""),
        "external_cost": external.model_dump(),
        "contracted_cost": {
            "amount_usd": contracted_amount,
            "source": contracted.get("source", ""),
        },
        "delta_vs_contracted_usd": external.amount_usd - contracted_amount,
        "manifest_path": str(manifest_path),
    }

    out_path: Path = args.out or (manifest_path.parent / "external_cost.json")
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(
        f"external cost ${external.amount_usd:.2f} "
        f"(contracted ${contracted_amount:.2f}, "
        f"delta ${payload['delta_vs_contracted_usd']:+.2f}) -> {out_path}"
    )
    if external.unpriced_models:
        print(f"unpriced (excluded): {', '.join(external.unpriced_models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
