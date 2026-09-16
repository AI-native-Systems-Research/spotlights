"""Fit and backtest a token-cost model for Spotlights runs.

Consumes `artifacts/calibration_features.json` from
`scripts/measure_calibration_targets.py` and answers the ship/no-ship question:
can total tokens for a repo we have never run be predicted from quantities a
user can measure *before* running (module count, files, lines)?

The honest test is leave-one-REPO-out cross-validation. Leaving out single runs
would leak: colpali contributes five near-identical replicates, so a model that
memorized colpali would score well while being useless on a new target.

Model forms are power laws fitted by ordinary least squares in log space:

    tokens = 10^c * x1^a1 * x2^a2 * ...

Judged against a pre-registered bar, fixed before any result was seen:
    median APE <= 30% and worst <= 100%  -> ship with bands
    median 30-60%                        -> order-of-magnitude guidance only
    median > 60%                          -> do not ship a calculator

Usage:
    python scripts/fit_cost_model.py --features artifacts/calibration_features.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BAR_MEDIAN_SHIP = 30.0
BAR_WORST_SHIP = 100.0
BAR_MEDIAN_ORDER = 60.0


# --------------------------------------------------------------------------- #
# Tiny OLS (pure Python -- matrices are at most 4x4 here)
# --------------------------------------------------------------------------- #
def ols(X: list[list[float]], y: list[float]) -> list[float] | None:
    """Solve min ||Xb - y|| via normal equations with Gaussian elimination."""
    n, p = len(X), len(X[0])
    if n < p:
        return None
    a = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(p)]
         + [sum(X[k][i] * y[k] for k in range(n))] for i in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        for r in range(p):
            if r != col and a[r][col]:
                f = a[r][col]
                a[r] = [v - f * w for v, w in zip(a[r], a[col])]
    return [a[i][p] for i in range(p)]


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def features(run: dict) -> dict[str, float]:
    s = run.get("scope") or {}
    mods = float(run.get("n_modules_billed") or 0)
    f = {
        "mods": mods,
        "loc": float(s.get("own_code_loc") or 0),
        "all_loc": float(s.get("own_loc") or 0),
        "files": float(s.get("own_code_files") or 0),
        "bytes": float(s.get("own_bytes") or 0),
        "rec_loc": float(s.get("rec_code_loc") or 0),
        # Probe-measurable: known only after the cheap front of the pipeline
        # (extraction + discovery) has actually run on the target.
        "candidates": float(run.get("candidates") or 0),
        "disc_tokens": float(run.get("disc_tokens") or 0),
        "disc_calls": float(run.get("disc_calls") or 0),
        "pairs": float(run.get("pairs") or 0),
    }
    f["loc_per_mod"] = f["loc"] / mods if mods else 0.0
    f["files_per_mod"] = f["files"] / mods if mods else 0.0
    f["cands_per_mod"] = f["candidates"] / mods if mods else 0.0
    return f


# Split by what the user can know *when they ask*. A-priori models need only a
# checkout; probe models need the cheap front of the pipeline to have run.
MODELS: dict[str, list[str]] = {
    "[a-priori] constant": [],
    "[a-priori] mods": ["mods"],
    "[a-priori] code_loc": ["loc"],
    "[a-priori] code_files": ["files"],
    "[a-priori] mods + code_loc": ["mods", "loc"],
    "[a-priori] mods + code_files": ["mods", "files"],
    "[a-priori] mods + loc_per_mod": ["mods", "loc_per_mod"],
    "[probe] candidates": ["candidates"],
    "[probe] candidates + mods": ["candidates", "mods"],
    "[probe] candidates + code_loc": ["candidates", "loc"],
    "[probe] discovery_tokens": ["disc_tokens"],
    "[probe] discovery_tokens + candidates": ["disc_tokens", "candidates"],
    "[probe] pairs": ["pairs"],
}


def design(rows: list[dict], names: list[str]) -> tuple[list[list[float]], list[float]]:
    X, y = [], []
    for r in rows:
        f = features(r)
        vals = [f[n] for n in names]
        if any(v <= 0 for v in vals) or r["total_tokens"] <= 0:
            continue
        X.append([1.0] + [math.log10(v) for v in vals])
        y.append(math.log10(float(r["total_tokens"])))
    return X, y


def predict(beta: list[float], row: dict, names: list[str]) -> float | None:
    f = features(row)
    vals = [f[n] for n in names]
    if any(v <= 0 for v in vals):
        return None
    lg = beta[0] + sum(b * math.log10(v) for b, v in zip(beta[1:], vals))
    return 10.0**lg


# --------------------------------------------------------------------------- #
def loocv(rows: list[dict], names: list[str]) -> list[tuple[dict, float]]:
    """Leave-one-repo-out. Returns (row, absolute percentage error) pairs."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["repo"]].append(r)

    out = []
    for held, test_rows in groups.items():
        train = [r for r in rows if r["repo"] != held]
        X, y = design(train, names)
        if not X:
            continue
        beta = ols(X, y)
        if beta is None:
            continue
        for r in test_rows:
            p = predict(beta, r, names)
            if p is None:
                continue
            actual = float(r["total_tokens"])
            out.append((r, 100.0 * abs(p - actual) / actual))
    return out


def verdict(median: float, worst: float) -> str:
    if median <= BAR_MEDIAN_SHIP and worst <= BAR_WORST_SHIP:
        return "PASS -> ship with bands"
    if median <= BAR_MEDIAN_ORDER:
        return "PARTIAL -> order-of-magnitude guidance only"
    return "FAIL -> do not ship a calculator"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--drivers", type=Path, default=None,
                    help="cost_drivers.json, to enable the [probe] models")
    ap.add_argument("--full-pipeline-only", action="store_true",
                    help="keep only runs whose telemetry shows the proposal step ran")
    args = ap.parse_args()

    data = json.loads(args.features.read_text(encoding="utf-8"))
    if args.drivers:
        drv = {d["run"]: d for d in json.loads(args.drivers.read_text(encoding="utf-8"))}
        for r in data:
            d = drv.get(r["run"]) or {}
            r["candidates"] = d.get("candidates_found") or 0
            r["disc_tokens"] = d.get("disc_tokens") or 0
            r["disc_calls"] = d.get("disc_calls") or 0
            r["pairs"] = d.get("pairs") or 0

    rows = [
        r for r in data
        if r.get("total_tokens") and r.get("clone_ok") and (r.get("scope") or {}).get("own_code_loc")
    ]
    if args.full_pipeline_only:
        rows = [r for r in rows if r.get("pairs")]
    excluded = [r["run"] for r in data if r not in rows]

    print(f"usable runs: {len(rows)}   (excluded {len(excluded)})")
    for e in excluded:
        print(f"  excluded: {e}")

    print("\n" + "=" * 104)
    print("MEASURED SCOPE PER RUN")
    print("=" * 104)
    print(f"{'run':<46}{'repo':<20}{'mods':>5}{'code_loc':>10}{'loc/mod':>9}{'files':>7}{'tokens':>10}{'tok/kLOC':>10}")
    print("-" * 104)
    for r in sorted(rows, key=lambda x: -x["total_tokens"]):
        f = features(r)
        print(
            f"{r['run'][:45]:<46}{r['repo'][:19]:<20}{int(f['mods']):>5}"
            f"{int(f['loc']):>10,}{int(f['loc_per_mod']):>9,}{int(f['files']):>7,}"
            f"{r['total_tokens'] / 1e6:>9.1f}M{r['total_tokens'] / max(f['loc'], 1) * 1000 / 1e6:>10.2f}M"
        )

    # Noise floor: true replicates only (same repo, same commit, same scope).
    print("\n" + "=" * 104)
    print("NOISE FLOOR  (identical target+scope, repeated)")
    print("=" * 104)
    reps: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        reps[(r["repo"], r["commit"], r["n_modules_billed"])].append(float(r["total_tokens"]))
    floor = None
    for k, v in sorted(reps.items()):
        if len(v) < 2:
            continue
        mean = statistics.fmean(v)
        cv = 100 * statistics.stdev(v) / mean
        spread = 100 * (max(v) - min(v)) / statistics.median(v)
        print(f"  {k[0]:<22} n={len(v)}  mean={mean / 1e6:.1f}M  CV={cv:.1f}%  peak-to-peak={spread:.0f}%")
        floor = cv if floor is None else max(floor, cv)
    if floor is not None:
        print(f"\n  -> no model can beat roughly {floor:.0f}% typical error on this corpus")

    print("\n" + "=" * 104)
    print("LEAVE-ONE-REPO-OUT BACKTEST  (target: run total_tokens)")
    print("=" * 104)
    print(f"{'model':<32}{'n':>4}{'median APE':>12}{'mean':>8}{'worst':>8}{'<=30%':>7}{'<=50%':>7}   verdict")
    print("-" * 104)
    results = {}
    for label, names in MODELS.items():
        errs = loocv(rows, names)
        if not errs:
            print(f"{label:<32}  (not fittable)")
            continue
        apes = [e for _, e in errs]
        med, worst = statistics.median(apes), max(apes)
        results[label] = (errs, med, worst)
        w30 = sum(1 for a in apes if a <= 30)
        w50 = sum(1 for a in apes if a <= 50)
        print(
            f"{label:<32}{len(apes):>4}{med:>11.0f}%{statistics.fmean(apes):>7.0f}%"
            f"{worst:>7.0f}%{w30:>7}{w50:>7}   {verdict(med, worst)}"
        )

    if results:
        best = min(results.items(), key=lambda kv: kv[1][1])
        label, (errs, med, worst) = best
        print("\n" + "=" * 104)
        print(f"PER-RUN ERRORS FOR BEST MODEL: {label}")
        print("=" * 104)
        print(f"{'run':<46}{'repo':<20}{'actual':>10}{'predicted':>11}{'error':>8}")
        print("-" * 104)
        for r, ape in sorted(errs, key=lambda t: -t[1]):
            groups = [x for x in rows if x["repo"] != r["repo"]]
            X, y = design(groups, MODELS[label])
            beta = ols(X, y) if X else None
            p = predict(beta, r, MODELS[label]) if beta else None
            print(
                f"{r['run'][:45]:<46}{r['repo'][:19]:<20}"
                f"{r['total_tokens'] / 1e6:>9.1f}M{(p or 0) / 1e6:>10.1f}M{ape:>7.0f}%"
            )
        print(f"\nPRE-REGISTERED VERDICT: median {med:.0f}%, worst {worst:.0f}% -> {verdict(med, worst)}")

        # Fitted exponents on the full corpus, for interpretation only.
        X, y = design(rows, MODELS[label])
        beta = ols(X, y)
        if beta:
            terms = "  ".join(
                f"{n}^{b:.2f}" for n, b in zip(MODELS[label], beta[1:])
            )
            print(f"full-corpus fit:  tokens = 10^{beta[0]:.2f} * {terms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
