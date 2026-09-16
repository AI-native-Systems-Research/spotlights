"""Build and validate a structural token estimator for Spotlights.

Approach: a run's tokens are spent per *structural unit* of pipeline work --

    candidate_discovery            per in-scope module
    module_deep_research           per in-scope module
    agent_proposals                per candidate
    proposal_from_finding_creator  per (candidate, finding) pair

Rather than assuming how many calls each unit makes -- an earlier version
assumed both CLIs ran deep research on every module, which is false and
inflated estimates 3x -- the per-unit token rates are measured directly from
durable per-invocation usage records.

Validation is leave-one-repo-out over the paper corpus, so the rates scoring a
target are never fitted on that target. The local artifacts corpus can be added
as extra training data, but `rocksdb-unhinted` appears in *both* corpora, so it
is matched by repo and held out with its paper twin.

Usage:
    python scripts/structural_estimator.py \
        --paper-runs <dir> --drivers artifacts/cost_drivers.json \
        --artifacts ./artifacts --out artifacts/calibration.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

BUCKETS = ("input", "output", "cache_read", "cache_create")
BAR_MEDIAN_SHIP, BAR_WORST_SHIP, BAR_MEDIAN_ORDER = 30.0, 100.0, 60.0

# Which structural unit each step's spend scales with.
STEP_UNIT = {
    "candidate_discovery": "mods",
    "module_deep_research": "mods",
    "deep_research": "mods",
    "agent_proposals": "candidates",
    "proposal_from_finding_creator": "pairs",
}


@dataclass
class Obs:
    """One run, reduced to structural counts and per-step token totals."""

    run: str
    repo: str
    mods: float
    candidates: float
    pairs: float
    total_tokens: float
    by_step: dict[str, float]

    def unit_count(self, unit: str) -> float:
        return {"mods": self.mods, "candidates": self.candidates, "pairs": self.pairs}[unit]


def local_observations(artifacts: Path) -> list[Obs]:
    """Reduce the fine-grained local corpus to the same shape as a paper run."""
    per_run: dict[str, dict] = defaultdict(
        lambda: {"steps": defaultdict(float), "modules": set(), "ap_calls": 0, "pf_calls": 0}
    )
    for p in artifacts.glob("*/spotlights_manager/modules/*/*.usage/*.json"):
        try:
            with p.open(encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        run = p.parts[len(artifacts.parts)]
        step = d.get("step") or "?"
        tok = sum(int(d.get(b) or 0) for b in BUCKETS)
        r = per_run[run]
        r["steps"][step] += tok
        r["modules"].add(d.get("module_qualified_name") or "?")
        if step == "agent_proposals" and (d.get("cli") == "claude"):
            r["ap_calls"] += 1
        if step == "proposal_from_finding_creator":
            r["pf_calls"] += 1

    out = []
    for run, r in per_run.items():
        by_step = dict(r["steps"])
        out.append(
            Obs(
                run=f"local:{run}",
                repo=_repo_of_local(run),
                mods=float(len(r["modules"])),
                candidates=float(r["ap_calls"]),
                pairs=float(r["pf_calls"]),
                total_tokens=float(sum(by_step.values())),
                by_step=by_step,
            )
        )
    return out


def _repo_of_local(run: str) -> str:
    """Map a local artifacts dir to a repo key so it shares held-out folds with
    its paper twin -- `rocksdb-unhinted` exists in both corpora."""
    r = run.lower()
    if "rocksdb" in r:
        return "rocksdb"
    if "kv" in r or "sql-opt" in r or "verify" in r:
        return "vllm"
    return f"local-{r}"


def paper_observations(drivers: Path) -> list[Obs]:
    rows = json.loads(drivers.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        if not r.get("pairs"):
            continue  # never ran the proposal step; a different pipeline shape
        out.append(
            Obs(
                run=r["run"],
                repo=r["repo"],
                mods=float(r["mods"]),
                candidates=float(r["candidates_found"]),
                pairs=float(r["pairs"]),
                total_tokens=float(r["total_tokens"]),
                by_step={},
            )
        )
    return out


def fit_rates(obs: list[Obs]) -> dict[str, float]:
    """Median tokens per structural unit, per step, over the training runs.

    Only runs carrying a per-step breakdown contribute; a run with totals only
    contributes to the whole-run scale factor below.
    """
    per_step: dict[str, list[float]] = defaultdict(list)
    for o in obs:
        for step, tok in o.by_step.items():
            unit = STEP_UNIT.get(step)
            if not unit:
                continue
            n = o.unit_count(unit)
            if n > 0 and tok > 0:
                per_step[step].append(tok / n)
    return {s: statistics.median(v) for s, v in per_step.items()}


def rates_from_totals(obs: list[Obs], rates: dict[str, float]) -> float:
    """One scale factor correcting per-unit rates to observed run totals.

    Per-step rates come from the few runs with a breakdown; this rescales them
    against every training run's total, which is the quantity being predicted.
    """
    ratios = []
    for o in obs:
        raw = raw_estimate(o, rates)
        if raw > 0 and o.total_tokens > 0:
            ratios.append(o.total_tokens / raw)
    return statistics.median(ratios) if ratios else 1.0


def raw_estimate(o: Obs, rates: dict[str, float]) -> float:
    total = 0.0
    for step, rate in rates.items():
        unit = STEP_UNIT.get(step)
        if unit:
            total += rate * o.unit_count(unit)
    return total


def estimate(o: Obs, rates: dict[str, float], scale: float) -> float:
    return raw_estimate(o, rates) * scale


def nnls(X: list[list[float]], y: list[float]) -> list[float] | None:
    """Least squares with coefficients clamped to >= 0.

    Tokens are additive over pipeline units, so a negative rate is physically
    meaningless: drop any variable that fits negative and refit on the rest.
    """
    active = list(range(len(X[0])))
    while active:
        Xa = [[row[i] for i in active] for row in X]
        p = len(active)
        a = [[sum(Xa[k][i] * Xa[k][j] for k in range(len(Xa))) for j in range(p)]
             + [sum(Xa[k][i] * y[k] for k in range(len(Xa)))] for i in range(p)]
        for col in range(p):
            piv = max(range(col, p), key=lambda r: abs(a[r][col]))
            if abs(a[piv][col]) < 1e-9:
                return None
            a[col], a[piv] = a[piv], a[col]
            d = a[col][col]
            a[col] = [v / d for v in a[col]]
            for r in range(p):
                if r != col and a[r][col]:
                    f = a[r][col]
                    a[r] = [v - f * w for v, w in zip(a[r], a[col])]
        beta = [a[i][p] for i in range(p)]
        worst = min(range(p), key=lambda i: beta[i])
        if beta[worst] >= 0:
            full = [0.0] * len(X[0])
            for i, idx in enumerate(active):
                full[idx] = beta[i]
            return full
        active.pop(worst)
    return None


ADDITIVE_UNITS = ("mods", "candidates", "pairs")


def loocv_additive(paper: list[Obs], units: tuple[str, ...], apriori: bool) -> list:
    """Fit tokens = sum(rate_u * count_u) on the paper corpus, held out by repo."""
    repos = sorted({o.repo for o in paper})
    results = []
    for held in repos:
        train = [o for o in paper if o.repo != held]
        if len(train) < len(units):
            continue
        X = [[o.unit_count(u) for u in units] for o in train]
        y = [o.total_tokens for o in train]
        beta = nnls(X, y)
        if beta is None:
            continue
        cpm = statistics.median([o.candidates / o.mods for o in train if o.mods])
        ppc = statistics.median([o.pairs / o.candidates for o in train if o.candidates])
        for o in [x for x in paper if x.repo == held]:
            t = o
            if apriori:
                c = o.mods * cpm
                t = Obs(o.run, o.repo, o.mods, c, c * ppc, o.total_tokens, o.by_step)
            est = sum(b * t.unit_count(u) for b, u in zip(beta, units))
            if est <= 0:
                continue
            results.append((o, est, 100 * abs(est - o.total_tokens) / o.total_tokens))
    return results


def verdict(med: float, worst: float) -> str:
    if med <= BAR_MEDIAN_SHIP and worst <= BAR_WORST_SHIP:
        return "PASS -> ship with bands"
    if med <= BAR_MEDIAN_ORDER:
        return "PARTIAL -> order-of-magnitude only"
    return "FAIL -> do not ship"


def loocv(paper: list[Obs], local: list[Obs], apriori: bool) -> list[tuple[Obs, float, float]]:
    """Leave-one-repo-out. `apriori=True` replaces the run's true candidate and
    pair counts with corpus-median fan-out rates, i.e. what a user knows before
    running anything."""
    repos = sorted({o.repo for o in paper})
    results = []
    for held in repos:
        tr_paper = [o for o in paper if o.repo != held]
        tr_local = [o for o in local if o.repo != held]
        rates = fit_rates(tr_local + tr_paper)
        if not rates:
            continue
        scale = rates_from_totals(tr_paper or tr_local, rates)
        cpm = statistics.median([o.candidates / o.mods for o in tr_paper if o.mods])
        ppc = statistics.median([o.pairs / o.candidates for o in tr_paper if o.candidates])
        for o in [x for x in paper if x.repo == held]:
            t = o
            if apriori:
                c = o.mods * cpm
                t = Obs(o.run, o.repo, o.mods, c, c * ppc, o.total_tokens, o.by_step)
            est = estimate(t, rates, scale)
            results.append((o, est, 100 * abs(est - o.total_tokens) / o.total_tokens))
    return results


def report(results: list[tuple[Obs, float, float]], label: str) -> tuple[float, float]:
    print("\n" + "=" * 104)
    print(label)
    print("=" * 104)
    print(f"{'run':<46}{'mods':>5}{'cands':>7}{'pairs':>7}{'actual':>10}{'est':>10}{'error':>8}")
    print("-" * 104)
    for o, est, ape in sorted(results, key=lambda t: -t[2]):
        print(
            f"{o.run[:45]:<46}{int(o.mods):>5}{int(o.candidates):>7}{int(o.pairs):>7}"
            f"{o.total_tokens / 1e6:>9.1f}M{est / 1e6:>9.1f}M{ape:>7.0f}%"
        )
    apes = [a for _, _, a in results]
    med, worst = statistics.median(apes), max(apes)
    print(
        f"\n  n={len(apes)}  median {med:.0f}%  mean {statistics.fmean(apes):.0f}%  "
        f"worst {worst:.0f}%  within-30%: {sum(1 for a in apes if a <= 30)}/{len(apes)}"
        f"  within-50%: {sum(1 for a in apes if a <= 50)}/{len(apes)}"
    )
    print(f"  VERDICT: {verdict(med, worst)}")
    return med, worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drivers", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    local = local_observations(args.artifacts)
    paper = paper_observations(args.drivers)
    print(f"local corpus: {len(local)} runs (per-step breakdown available)")
    for o in sorted(local, key=lambda x: x.run):
        print(f"  {o.run:<40} repo={o.repo:<10} mods={int(o.mods):>3} "
              f"cands={int(o.candidates):>4} pairs={int(o.pairs):>5} tok={o.total_tokens / 1e6:>6.1f}M")
    print(f"\npaper corpus: {len(paper)} full-pipeline runs")

    rates = fit_rates(local)
    print("\nper-unit token rates measured on the local corpus:")
    for s, v in sorted(rates.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<34}{v:>14,.0f} tokens per {STEP_UNIT[s]}")
    print(f"\nglobal scale factor vs paper totals: {rates_from_totals(paper, rates):.2f}x")

    med_a, worst_a = report(
        loocv(paper, local, apriori=False),
        "SCENARIO A -- discovery probe supplies true candidate and pair counts",
    )
    med_b, worst_b = report(
        loocv(paper, local, apriori=True),
        "SCENARIO B -- module count only, corpus-median fan-out (no probe)",
    )

    # Additive fit on the paper corpus itself, held out by repo. Per-unit rates
    # transferred from the local corpus are biased because 4 of its 5 runs cover
    # exactly one module.
    combos = {
        "mods + candidates + pairs": ("mods", "candidates", "pairs"),
        "mods + pairs": ("mods", "pairs"),
        "candidates + pairs": ("candidates", "pairs"),
        "pairs only": ("pairs",),
        "mods + candidates": ("mods", "candidates"),
        "candidates only": ("candidates",),
        "mods only": ("mods",),
    }
    print("\n" + "=" * 104)
    print("SCENARIO C -- additive per-unit fit on the paper corpus (leave-one-repo-out)")
    print("=" * 104)
    print(f"{'units':<30}{'probe?':<9}{'n':>4}{'median':>9}{'mean':>8}{'worst':>8}{'<=30%':>7}   verdict")
    print("-" * 104)
    best = None
    for label, units in combos.items():
        for apriori in (False, True):
            res = loocv_additive(paper, units, apriori)
            if not res:
                continue
            apes = [a for _, _, a in res]
            med, worst = statistics.median(apes), max(apes)
            tag = "no" if apriori else "yes"
            print(
                f"{label:<30}{tag:<9}{len(apes):>4}{med:>8.0f}%{statistics.fmean(apes):>7.0f}%"
                f"{worst:>7.0f}%{sum(1 for a in apes if a <= 30):>7}   {verdict(med, worst)}"
            )
            # Rank by the tail, not the median: a calculator whose median is
            # good but whose worst case is 5x wrong is not shippable.
            if best is None or (worst, med) < (best[2], best[1]):
                best = ((label, units, apriori), med, worst, res)

    if best:
        (label, units, apriori), med_c, worst_c, res = best
        report(res, f"BEST ADDITIVE MODEL: {label}  (probe={'no' if apriori else 'yes'})")
        X = [[o.unit_count(u) for u in units] for o in paper]
        beta = nnls(X, [o.total_tokens for o in paper])
        if beta:
            print("  full-corpus rates: " + ",  ".join(
                f"{b / 1e3:,.0f}k tokens per {u}" for b, u in zip(beta, units) if b
            ))
    else:
        med_c = worst_c = float("inf")

    print("\n" + "=" * 104)
    print("CONCLUSION")
    print("=" * 104)
    print(f"  transferred rates, with probe : median {med_a:>3.0f}%  worst {worst_a:>4.0f}%  -> {verdict(med_a, worst_a)}")
    print(f"  transferred rates, no probe   : median {med_b:>3.0f}%  worst {worst_b:>4.0f}%  -> {verdict(med_b, worst_b)}")
    print(f"  additive fit (best)           : median {med_c:>3.0f}%  worst {worst_c:>4.0f}%  -> {verdict(med_c, worst_c)}")

    if args.out:
        snap = {
            "_note": "Aggregate calibration measured from historical runs. Tokens, "
                     "not dollars: pricing is a rate-table lookup applied at estimate time.",
            "per_unit_token_rates": rates,
            "step_units": STEP_UNIT,
            "scale_factor": rates_from_totals(paper, rates),
            "fanout": {
                "candidates_per_module_median": statistics.median(
                    [o.candidates / o.mods for o in paper if o.mods]
                ),
                "pairs_per_candidate_median": statistics.median(
                    [o.pairs / o.candidates for o in paper if o.candidates]
                ),
            },
            "backtest": {
                "with_probe": {"median_ape": med_a, "worst_ape": worst_a},
                "without_probe": {"median_ape": med_b, "worst_ape": worst_b},
                "n_runs": len(paper),
                "method": "leave-one-repo-out over the paper corpus",
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snap, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
