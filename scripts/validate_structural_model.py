"""Validate the structural token model out-of-sample.

Regressing run totals on repo features failed (42% median error). But a run is
a sum of model calls, and the counts are structural rather than statistical:

    candidate_discovery            2 calls per module, per CLI   (measured CV 0%)
    module_deep_research           1 call  per module, per CLI   (measured CV 0%)
    agent_proposals                1 call  per candidate, per CLI
    proposal_from_finding_creator  1 call  per (candidate, finding) pair

So  tokens = SUM over steps of (calls x tokens-per-call), and the only unknowns
are how many candidates discovery finds and how many findings pair with them.

This script is the honest test of that claim: unit costs are taken from the
**local** artifacts corpus and applied to the **paper** corpus, which shares no
runs with it. Nothing is fitted on the data being scored.

Two scenarios are reported:

  exact-counts  -- feed the run's true candidate and pair counts. Isolates
                   unit-cost error: the accuracy ceiling of a probe-based skill.
  a-priori      -- feed only module count, with candidates and pairs from
                   corpus-median rates. What a user gets before running anything.

Usage:
    python scripts/validate_structural_model.py \
        --unit-costs artifacts/unit_costs.json \
        --drivers artifacts/cost_drivers.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BAR_MEDIAN_SHIP, BAR_WORST_SHIP, BAR_MEDIAN_ORDER = 30.0, 100.0, 60.0


def unit(costs: dict, step: str, cli: str) -> float:
    d = costs.get(f"{step}|{cli}")
    return float(d["median"]) if d else 0.0


def estimate(u: dict, mods: float, candidates: float, pairs: float) -> dict:
    """Token estimate broken out by step, from structural call counts."""
    disc = mods * 2 * (unit(u, "candidate_discovery", "claude")
                       + unit(u, "candidate_discovery", "codex"))
    dr = mods * 1 * (unit(u, "module_deep_research", "claude")
                     + unit(u, "module_deep_research", "codex"))
    ap = candidates * (unit(u, "agent_proposals", "claude")
                       + unit(u, "agent_proposals", "codex"))
    pf = pairs * unit(u, "proposal_from_finding_creator", "claude")
    return {
        "candidate_discovery": disc,
        "module_deep_research": dr,
        "agent_proposals": ap,
        "proposal_from_finding_creator": pf,
        "total": disc + dr + ap + pf,
    }


def verdict(median: float, worst: float) -> str:
    if median <= BAR_MEDIAN_SHIP and worst <= BAR_WORST_SHIP:
        return "PASS -> ship with bands"
    if median <= BAR_MEDIAN_ORDER:
        return "PARTIAL -> order-of-magnitude only"
    return "FAIL -> do not ship"


def score(rows: list[tuple[str, float, float]], label: str) -> tuple[float, float]:
    print("\n" + "=" * 108)
    print(label)
    print("=" * 108)
    print(f"{'run':<46}{'actual':>11}{'estimated':>12}{'error':>9}")
    print("-" * 108)
    apes = []
    for name, actual, est in sorted(rows, key=lambda t: -abs(t[2] - t[1]) / t[1]):
        ape = 100 * abs(est - actual) / actual
        apes.append(ape)
        print(f"{name[:45]:<46}{actual / 1e6:>10.1f}M{est / 1e6:>11.1f}M{ape:>8.0f}%")
    med, worst = statistics.median(apes), max(apes)
    within30 = sum(1 for a in apes if a <= 30)
    print(
        f"\n  n={len(apes)}  median {med:.0f}%  mean {statistics.fmean(apes):.0f}%  "
        f"worst {worst:.0f}%  within-30%: {within30}/{len(apes)}"
    )
    print(f"  VERDICT: {verdict(med, worst)}")
    return med, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unit-costs", type=Path, required=True)
    ap.add_argument("--drivers", type=Path, required=True)
    args = ap.parse_args()

    u = json.loads(args.unit_costs.read_text(encoding="utf-8"))
    runs = json.loads(args.drivers.read_text(encoding="utf-8"))

    print("UNIT COSTS (median tokens/call, measured on the LOCAL corpus only)")
    print("-" * 108)
    for k in sorted(u):
        print(f"  {k:<44}{u[k]['median']:>12,.0f}   (n={u[k]['n']}, CV {u[k]['cv']:.0f}%)")

    # Full-pipeline runs only: the two runs whose telemetry records no proposal
    # pairs never ran the step that is 52% of tokens, so no model of the
    # current pipeline can or should reproduce them.
    full = [r for r in runs if r.get("pairs")]
    skipped = [r["run"] for r in runs if not r.get("pairs")]
    print(f"\n{len(full)} full-pipeline runs; excluded {len(skipped)} without a proposal step:")
    for s in skipped:
        print(f"  - {s}")

    # ------------------------------------------------- scenario 1: exact counts
    rows = [
        (r["run"], float(r["total_tokens"]),
         estimate(u, r["mods"], r["candidates_found"], r["pairs"])["total"])
        for r in full
    ]
    med_exact, worst_exact = score(
        rows, "SCENARIO A -- exact candidate and pair counts (probe-based skill ceiling)"
    )

    # ------------------------------------------------- scenario 2: a-priori
    # Rates a user has before running: corpus medians per module / per candidate.
    cands_per_mod = statistics.median([r["candidates_found"] / r["mods"] for r in full if r["mods"]])
    pairs_per_cand = statistics.median(
        [r["pairs"] / r["candidates_found"] for r in full if r["candidates_found"]]
    )
    print(f"\n  a-priori rates: {cands_per_mod:.1f} candidates/module, "
          f"{pairs_per_cand:.1f} pairs/candidate")
    rows2 = []
    for r in full:
        c = r["mods"] * cands_per_mod
        p = c * pairs_per_cand
        rows2.append((r["run"], float(r["total_tokens"]), estimate(u, r["mods"], c, p)["total"]))
    med_ap, worst_ap = score(
        rows2, "SCENARIO B -- module count only, corpus-median fan-out (no probe)"
    )

    # ------------------------------------------------- where the money goes
    print("\n" + "=" * 108)
    print("STEP MIX OF THE ESTIMATE  (exact counts, summed over runs)")
    print("=" * 108)
    agg: dict[str, float] = {}
    for r in full:
        e = estimate(u, r["mods"], r["candidates_found"], r["pairs"])
        for k, v in e.items():
            if k != "total":
                agg[k] = agg.get(k, 0.0) + v
    tot = sum(agg.values()) or 1
    for k, v in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<34}{v / 1e6:>9.0f}M{100 * v / tot:>7.1f}%")

    print("\n" + "=" * 108)
    print("CONCLUSION")
    print("=" * 108)
    print(f"  with a discovery probe : median {med_exact:.0f}%, worst {worst_exact:.0f}%"
          f"  -> {verdict(med_exact, worst_exact)}")
    print(f"  without a probe        : median {med_ap:.0f}%, worst {worst_ap:.0f}%"
          f"  -> {verdict(med_ap, worst_ap)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
