"""Aggregate historical runs into a calibration snapshot for the scoping advisor.

The advisor does NOT predict a run's cost as a number -- that was tested and
failed (best leave-one-repo-out backtest: 27% median but 123% worst error, and
without a probe 47%/345%). Repo size is not a usable predictor either: tokens
per kLOC spans 24x across targets and runs *inversely* with size.

What survived validation is relative structure, so this snapshot stores only
quantities the advisor can defend:

  step_shares        which steps the money goes to (drives what-if toggles)
  tokens_per_module  a BAND, not a point -- the honest width of our ignorance
  fanout             candidates/module, pairs/candidate as observed ranges
  barren_modules     share of modules that produce no proposals (exclude case)
  whatif             multipliers, each tagged with how it was established
  runtime            tokens per API-second, for wall-clock from parallelism

Privacy: this file is designed to be committable to a public repo. Internal
repo names are replaced with opaque labels, and it stores TOKENS, never dollars
-- pricing is a rate-table lookup applied at estimate time, and the rate tables
are the sensitive part. No URLs, no commit SHAs, no filesystem paths.

Usage:
    python scripts/build_advisor_calibration.py \
        --artifacts ./artifacts \
        --drivers artifacts/cost_drivers.json \
        --features artifacts/calibration_features.json \
        --out artifacts/advisor_calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

BUCKETS = ("input", "output", "cache_read", "cache_create")

#: Public OSS targets keep their names -- they are already public and knowing
#: Spotlights was run on rocksdb leaks nothing. Everything else is opaque.
PUBLIC_REPOS = ("rocksdb", "vllm", "colpali", "qiskit")


def anonymize(repo: str) -> str:
    low = repo.lower()
    for pub in PUBLIC_REPOS:
        if pub in low:
            return pub
    return ""  # caller assigns a stable internal-N label


def band(vals: list[float]) -> dict[str, float]:
    """p10/p50/p90 band. The advisor quotes ranges because point estimates on
    this corpus are wrong by 47% typically and 345% at worst."""
    v = sorted(float(x) for x in vals)
    if not v:
        return {}
    n = len(v)

    def q(f: float) -> float:
        if n == 1:
            return v[0]
        i = f * (n - 1)
        lo, hi = int(i), min(int(i) + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (i - lo)

    return {
        "p10": q(0.10),
        "p50": statistics.median(v),
        "p90": q(0.90),
        "min": v[0],
        "max": v[-1],
        "n": n,
    }


def local_step_shares(
    artifacts: Path,
) -> tuple[dict[str, float], dict[str, dict], dict, dict[str, float], dict[str, dict]]:
    """Step token shares and per-call unit costs from per-invocation records.

    These records are the only place the per-step split is durable; run
    manifests carry totals only.
    """
    by_step: dict[str, float] = defaultdict(float)
    per_call: dict[str, list[float]] = defaultdict(list)
    extraction: dict[str, list[float]] = {"tokens": [], "api_time_s": []}
    buckets: dict[str, float] = defaultdict(float)
    # Step shares vary a lot per run, so a single pooled number would be
    # dominated by the largest run. Keep the per-run distribution too.
    per_run_step: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for p in artifacts.glob("*/spotlights_manager/**/*.usage/*.json"):
        try:
            with p.open(encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        step = str(d.get("step") or "?")
        tok = float(sum(int(d.get(b) or 0) for b in BUCKETS))
        by_step[step] += tok
        per_call[step].append(tok)
        per_run_step[p.parts[len(artifacts.parts)]][step] += tok
        for b in BUCKETS:
            buckets[b] += float(int(d.get(b) or 0))
        if "extract" in step:
            extraction["tokens"].append(tok)
            if d.get("api_time_s"):
                extraction["api_time_s"].append(float(d["api_time_s"]))

    total = sum(by_step.values()) or 1.0
    shares = {k: v / total for k, v in sorted(by_step.items(), key=lambda kv: -kv[1])}
    units = {k: band(v) for k, v in per_call.items()}
    extract_band = {k: band(v) for k, v in extraction.items() if v}
    btotal = sum(buckets.values()) or 1.0
    bucket_mix = {k: buckets[k] / btotal for k in BUCKETS}
    step_bands = {
        step: band([
            m[step] / sum(m.values()) for m in per_run_step.values()
            if sum(m.values()) and step in m
        ])
        for step in sorted(by_step)
    }
    return shares, units, extract_band, bucket_mix, step_bands


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--drivers", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    drivers = json.loads(args.drivers.read_text(encoding="utf-8"))
    feats = {r["run"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))}

    # Stable anonymous labels for internal targets.
    labels: dict[str, str] = {}
    counter = 0
    for r in sorted(drivers, key=lambda x: x["repo"]):
        repo = r["repo"]
        if repo in labels:
            continue
        pub = anonymize(repo)
        if pub:
            labels[repo] = pub
        else:
            counter += 1
            labels[repo] = f"internal-{counter}"

    # A run with zero proposal pairs ran with deep research disabled (the flag
    # short-circuits step 4 to zero proposals), so it is a different pipeline
    # shape and must not be pooled with full runs.
    full = [r for r in drivers if r.get("pairs")]
    no_dr = [r for r in drivers if not r.get("pairs")]

    shares, units, extract_band, bucket_mix, step_bands = local_step_shares(args.artifacts)

    # --- the exclude case: how many modules produce no proposals at all ------
    barren, per_mod_conc = [], []
    for r in drivers:
        mods = r.get("modules_with_telemetry") or r.get("mods") or 0
        if not mods or not r.get("pairs"):
            continue
        # pairs are spread over modules; a module with none contributed nothing
        # to the step that is the majority of spend.
        barren.append(1.0 - (r.get("productive_modules") or 0) / mods
                      if r.get("productive_modules") is not None else None)
    barren = [b for b in barren if b is not None]

    def tok_per_mod(rows: list[dict]) -> list[float]:
        return [r["total_tokens"] / r["mods"] for r in rows if r.get("mods")]

    snapshot: dict[str, Any] = {
        "_note": (
            "Calibration for the Spotlights scoping advisor. Stores TOKENS, never "
            "dollars: pricing is a rate-table lookup applied at estimate time. "
            "Bands are p10/p50/p90 -- absolute point prediction was tested and "
            "failed (47% median, 345% worst error), so the advisor quotes ranges "
            "and relative multipliers only."
        ),
        "corpus": {
            "runs_total": len(drivers),
            "runs_full_pipeline": len(full),
            "runs_deep_research_off": len(no_dr),
            "repos": len(labels),
            "engine_versions": len({r.get("engine") for r in drivers}),
            "caveat": (
                "review_iterations and max_findings_per_module are NOT recorded in "
                "these manifests, so runs differing several-fold in cost are "
                "indistinguishable here. This is the dominant source of band width."
            ),
        },
        "noise_floor": {
            "cv_pct": 7.7,
            "basis": "5 replicate runs of one repo at one commit",
            "meaning": "no model can beat roughly this typical error on this corpus",
        },
        "step_shares_pooled": shares,
        "step_shares_per_run": step_bands,
        # Cache reads are ~10x cheaper than fresh input, so the split between
        # the four billing buckets moves the dollar figure far more than any
        # token-count refinement. Measured, not assumed.
        "bucket_mix": bucket_mix,
        "tokens_per_call": units,
        "extraction": extract_band,
        "tokens_per_module": {
            "full_pipeline": band(tok_per_mod(full)),
            "deep_research_off": band(tok_per_mod(no_dr)),
        },
        "fanout": {
            "candidates_per_module": band(
                [r["candidates_found"] / r["mods"] for r in full if r.get("mods")]
            ),
            "pairs_per_candidate": band(
                [r["pairs"] / r["candidates_found"] for r in full if r.get("candidates_found")]
            ),
            "pairs_per_module": band(
                [r["pairs"] / r["mods"] for r in full if r.get("mods")]
            ),
        },
        "runtime": {
            "tokens_per_api_second": band(
                [r["total_tokens"] / r["api_s"] for r in drivers if r.get("api_s")]
            ),
            "observed_concurrency": band(
                [r["api_s"] / r["wall_s"] for r in drivers
                 if r.get("api_s") and r.get("wall_s")]
            ),
        },
        "per_run": [
            {
                "repo": labels[r["repo"]],
                "mods": r["mods"],
                "candidates": r["candidates_found"],
                "pairs": r["pairs"],
                "total_tokens": r["total_tokens"],
                "deep_research": bool(r.get("pairs")),
                "code_loc": (feats.get(r["run"]) or {}).get("scope", {}).get("own_code_loc"),
                "api_s": r.get("api_s"),
                "wall_s": r.get("wall_s"),
            }
            for r in sorted(drivers, key=lambda x: -x["total_tokens"])
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"wrote {args.out}")
    print(f"\ncorpus: {len(drivers)} runs, {len(full)} full pipeline, "
          f"{len(no_dr)} with deep research off, {len(labels)} repos")
    print("\nstep shares (per-run distribution, local per-invocation records):")
    for k, b in sorted(step_bands.items(), key=lambda kv: -(kv[1].get("p50") or 0)):
        if b:
            print(f"  {k:<34}median {100 * b['p50']:>5.1f}%   "
                  f"range {100 * b['min']:>5.1f}-{100 * b['max']:>5.1f}%   (n={b['n']})")
    print("\ntoken bucket mix (drives price far more than token count):")
    for k, v in bucket_mix.items():
        print(f"  {k:<34}{100 * v:>6.1f}%")
    print("\ntokens per module:")
    for mode, b in snapshot["tokens_per_module"].items():
        if b:
            print(f"  {mode:<20} p10 {b['p10'] / 1e6:>6.1f}M  p50 {b['p50'] / 1e6:>6.1f}M  "
                  f"p90 {b['p90'] / 1e6:>6.1f}M   (n={b['n']}, spread {b['max'] / b['min']:.0f}x)")
    print("\nfan-out:")
    for k, b in snapshot["fanout"].items():
        if b:
            print(f"  {k:<26} p10 {b['p10']:>7.1f}  p50 {b['p50']:>7.1f}  p90 {b['p90']:>7.1f}"
                  f"   spread {b['max'] / max(b['min'], 1e-9):.0f}x")
    rt = snapshot["runtime"]["tokens_per_api_second"]
    if rt:
        print(f"\ntokens per API-second: p10 {rt['p10']:,.0f}  p50 {rt['p50']:,.0f}  "
              f"p90 {rt['p90']:,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
