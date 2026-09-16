"""Test whether per-step tokens-per-call is stable enough to build a calculator.

Run totals are unpredictable (best backtest: 42% median error). But a run's
tokens are just (number of model calls) x (tokens per call), summed over steps.
If tokens-per-call is tight *within a step*, the hard part is only counting
calls -- and call counts are structural: they follow from module count,
candidate count, findings count, and the user's own flags
(`--review-iterations`, `--max-findings-per-module`).

This measures the per-call distribution per step from durable per-invocation
usage records, which carry the four disjoint token buckets and `api_time_s`.

A low coefficient of variation here means a structural calculator is viable
even though a regression on repo features is not.

Usage:
    python scripts/analyze_per_call_stability.py --artifacts ./artifacts
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

BUCKETS = ("input", "output", "cache_read", "cache_create")


def load_records(artifacts: Path) -> list[dict]:
    out = []
    for p in artifacts.glob("*/spotlights_manager/modules/*/*.usage/*.json"):
        try:
            with p.open(encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        d["_run"] = p.parts[len(artifacts.parts)]
        d["_tokens"] = sum(int(d.get(b) or 0) for b in BUCKETS)
        out.append(d)
    return out


def describe(vals: list[float]) -> dict:
    vals = sorted(vals)
    n = len(vals)

    def q(f: float) -> float:
        if n == 1:
            return vals[0]
        i = f * (n - 1)
        lo, hi = int(i), min(int(i) + 1, n - 1)
        return vals[lo] + (vals[hi] - vals[lo]) * (i - lo)

    mean = statistics.fmean(vals)
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(vals),
        "p10": q(0.10),
        "p90": q(0.90),
        "cv": 100 * statistics.stdev(vals) / mean if n > 1 and mean else 0.0,
        "p90_over_p10": q(0.90) / q(0.10) if q(0.10) else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    recs = load_records(args.artifacts)
    if not recs:
        print(f"no usage records under {args.artifacts}")
        return 1
    runs = sorted({r["_run"] for r in recs})
    print(f"{len(recs)} invocations across {len(runs)} local runs")

    # ---------------------------------------------------------------- per call
    print("\n" + "=" * 122)
    print("TOKENS PER CALL, BY STEP AND CLI   (the calculator's unit cost)")
    print("=" * 122)
    print(f"{'step':<32}{'cli':<8}{'calls':>7}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}{'CV':>7}{'p90/p10':>9}")
    print("-" * 122)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in recs:
        groups[(r.get("step") or "?", r.get("cli") or "?")].append(float(r["_tokens"]))
    unit_costs = {}
    for k in sorted(groups):
        d = describe(groups[k])
        unit_costs["|".join(k)] = d
        print(
            f"{k[0]:<32}{k[1]:<8}{d['n']:>7}{d['median']:>10,.0f}{d['mean']:>10,.0f}"
            f"{d['p10']:>10,.0f}{d['p90']:>10,.0f}{d['cv']:>6.0f}%{d['p90_over_p10']:>9.1f}x"
        )

    # Is the unit cost stable across *runs*? That is what a shipped snapshot
    # depends on -- a tight within-run spread is worthless if the mean moves.
    print("\n" + "=" * 122)
    print("IS THE UNIT COST STABLE ACROSS RUNS?   (per-run median tokens/call)")
    print("=" * 122)
    per_run: dict[tuple, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in recs:
        per_run[(r.get("step") or "?", r.get("cli") or "?")][r["_run"]].append(float(r["_tokens"]))
    print(f"{'step':<32}{'cli':<8}{'runs':>6}{'lo median':>12}{'hi median':>12}{'spread':>9}")
    print("-" * 122)
    for k in sorted(per_run):
        meds = [statistics.median(v) for v in per_run[k].values() if v]
        if len(meds) < 2:
            continue
        print(
            f"{k[0]:<32}{k[1]:<8}{len(meds):>6}{min(meds):>12,.0f}{max(meds):>12,.0f}"
            f"{max(meds) / min(meds):>8.1f}x"
        )

    # -------------------------------------------------------------- call counts
    print("\n" + "=" * 122)
    print("CALLS PER MODULE, BY STEP   (the part a structural model must predict)")
    print("=" * 122)
    counts: dict[tuple, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))
    for r in recs:
        counts[(r.get("step") or "?", r.get("cli") or "?")][
            (r["_run"], r.get("module_qualified_name") or "?")
        ] += 1
    print(f"{'step':<32}{'cli':<8}{'modules':>9}{'median':>9}{'p10':>8}{'p90':>8}{'CV':>7}")
    print("-" * 122)
    for k in sorted(counts):
        vals = [float(v) for v in counts[k].values()]
        d = describe(vals)
        print(
            f"{k[0]:<32}{k[1]:<8}{d['n']:>9}{d['median']:>9.1f}{d['p10']:>8.1f}"
            f"{d['p90']:>8.1f}{d['cv']:>6.0f}%"
        )

    # ---------------------------------------------------------- seconds per call
    print("\n" + "=" * 122)
    print("API SECONDS PER CALL   (runtime side of the estimate)")
    print("=" * 122)
    print(f"{'step':<32}{'cli':<8}{'calls':>7}{'median':>9}{'p10':>8}{'p90':>8}{'CV':>7}")
    print("-" * 122)
    for k in sorted(groups):
        vals = [
            float(r.get("api_time_s") or 0)
            for r in recs
            if (r.get("step"), r.get("cli")) == k and r.get("api_time_s")
        ]
        if not vals:
            continue
        d = describe(vals)
        print(
            f"{k[0]:<32}{k[1]:<8}{d['n']:>7}{d['median']:>9.0f}{d['p10']:>8.0f}"
            f"{d['p90']:>8.0f}{d['cv']:>6.0f}%"
        )

    # ------------------------------------------------- reconstruct run totals
    print("\n" + "=" * 122)
    print("SANITY CHECK: rebuild each local run from median unit costs x its own call counts")
    print("=" * 122)
    print(f"{'run':<44}{'actual':>12}{'rebuilt':>12}{'error':>8}")
    print("-" * 122)
    apes = []
    for run in runs:
        actual = sum(r["_tokens"] for r in recs if r["_run"] == run)
        rebuilt = 0.0
        for k in groups:
            n = sum(
                1 for r in recs
                if r["_run"] == run and (r.get("step"), r.get("cli")) == k
            )
            rebuilt += n * unit_costs["|".join(k)]["median"]
        if not actual:
            continue
        ape = 100 * abs(rebuilt - actual) / actual
        apes.append(ape)
        print(f"{run[:43]:<44}{actual / 1e6:>11.1f}M{rebuilt / 1e6:>11.1f}M{ape:>7.0f}%")
    if apes:
        print(
            f"\n  median error from call counts alone: {statistics.median(apes):.0f}%"
            f"  (worst {max(apes):.0f}%)"
        )
        print("  This is the calculator's floor GIVEN exact call counts -- it isolates")
        print("  unit-cost error from count-prediction error.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(unit_costs, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
