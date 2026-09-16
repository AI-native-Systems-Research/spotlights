"""Find what actually drives Spotlights token spend.

Repo size failed: tokens/kLOC varies 24x across targets and runs *inversely*
with size, because an agent reads a bounded slice of a module however large the
module is. So the driver must be pipeline work, not code volume.

The pipeline is a fan-out: discovery proposes candidates per module, then
`proposal_from_finding_creator` -- a measured median 63% of a run's tokens --
runs once per (candidate, finding) pair. This script measures the fan-out counts recorded in
`per_module_telemetry` and tests them as predictors of run total tokens.

If a structural count predicts tokens tightly, the estimation problem becomes
"predict that count", and the cheap front of the pipeline can measure it
directly instead of guessing.

Usage:
    python scripts/analyze_cost_drivers.py \
        --paper-runs <dir> --features artifacts/calibration_features.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

BILLED = ("SUCCEEDED", "DEGRADED", "FAILED")
BUCKETS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_create_tokens")


def _load(p: Path):
    with p.open(encoding="utf-8") as fh:
        return json.load(fh)


def _durations(v) -> list[float]:
    """Per-item durations are a list in some manifest generations and a
    {item_id: seconds} mapping in others. Return just the seconds."""
    if isinstance(v, dict):
        v = list(v.values())
    out = []
    for x in v or []:
        try:
            out.append(float(x or 0))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def run_structure(result: dict, manifest: dict) -> dict:
    """Fan-out counts and step-1 tokens for one run, from per_module_telemetry."""
    pmt = result.get("per_module_telemetry") or {}
    module_runs = result.get("module_runs") or {}
    billed = {
        k for k, v in module_runs.items()
        if not isinstance(v, dict) or str(v.get("status", "")).upper() in BILLED
    }

    out = {
        "disc_calls": 0,
        "disc_tokens": 0,
        "disc_api_s": 0.0,
        "candidates_found": 0,
        # Discovery split into the first pass and the review passes that follow
        # it. Review is an alternating claude/codex debate, so the two halves are
        # billed at different rates and are worth keeping apart.
        "disc_tokens_first": 0,
        "disc_tokens_review": 0,
        "candidates_first": 0,
        "pairs": 0,
        "proposal_api_s": 0.0,
        "agent_prop_calls": 0,
        "agent_prop_api_s": 0.0,
        "deep_research_s": 0.0,
        "modules_with_telemetry": 0,
        "long_context_calls": 0,
        # Modules that yielded at least one (candidate, finding) pair. The rest
        # were analyzed by steps 1-3 but contributed nothing to step 4, which is
        # the majority of spend -- so they are the exclude-recommendation case.
        "productive_modules": 0,
    }
    iter_counts: list[int] = []
    for mod, t in pmt.items():
        if mod not in billed:
            continue
        out["modules_with_telemetry"] += 1
        iters = t.get("discovery_iterations") or []
        for it in iters:
            out["disc_calls"] += 1
            tok = sum(int(it.get(b) or 0) for b in BUCKETS)
            out["disc_tokens"] += tok
            out["disc_api_s"] += float(it.get("api_time_s") or 0)
            if int(it.get("n") or 0) == 0:
                out["disc_tokens_first"] += tok
                out["candidates_first"] += int(it.get("candidate_count") or 0)
            else:
                out["disc_tokens_review"] += tok
            if "[1m]" in (it.get("model") or ""):
                out["long_context_calls"] += 1
        # `candidate_count` is the RUNNING TOTAL after an iteration, not that
        # iteration's own yield -- a module reads 9, 13, 13, 16 with `added`
        # explaining each step. Summing it across iterations multiplied every
        # review-enabled run's candidate count by ~4x. Take the last iteration
        # instead; `candidates_in_report` below cross-checks the result.
        if iters:
            out["candidates_found"] += int(iters[-1].get("candidate_count") or 0)
            iter_counts.append(len(iters))

        pairs = _durations(t.get("proposal_from_finding_per_pair_durations_s"))
        out["pairs"] += len(pairs)
        out["proposal_api_s"] += sum(pairs)
        if pairs:
            out["productive_modules"] += 1
        cands = _durations(t.get("agent_proposals_per_candidate_durations_s"))
        out["agent_prop_calls"] += len(cands)
        out["agent_prop_api_s"] += sum(cands)
        out["deep_research_s"] += float(t.get("deep_research_duration_s") or 0)

    # The engine never recorded --review-iterations in these runs, but it is
    # exactly recoverable: one discovery iteration per module means --no-review,
    # four means the default 3 review passes. Uniform within a run.
    out["review_iterations"] = (max(iter_counts) - 1) if iter_counts else None
    rep_c = (result.get("report") or {}).get("candidates")
    out["candidates_in_report"] = len(rep_c) if isinstance(rep_c, list) else None

    outs = manifest.get("outputs") or {}
    out["num_candidates"] = outs.get("num_candidates") or 0
    return out


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else 0.0


def cv_of_ratio(rows: list[dict], num: str, den: str) -> tuple[float, float, float, float]:
    """How stable is num/den across runs? (median, min, max, CV%)"""
    vals = [r[num] / r[den] for r in rows if r.get(den)]
    if len(vals) < 2:
        return (0, 0, 0, 0)
    return (
        statistics.median(vals),
        min(vals),
        max(vals),
        100 * statistics.stdev(vals) / statistics.fmean(vals),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-runs", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    feats = {r["run"]: r for r in json.loads(args.features.read_text(encoding="utf-8"))}

    rows = []
    for res in sorted(args.paper_runs.glob("**/result.json")):
        name = str(res.parent.relative_to(args.paper_runs)).replace("\\", "/")
        man_p = res.parent / "run_manifest.json"
        # A run whose manifest is missing (or reports zero tokens) still carries
        # full per-module telemetry, so it can inform the fan-out counts that do
        # not need a token total. Keep it with total_tokens=None and let each
        # token-dependent analysis below filter it out, rather than discarding
        # the run's structure entirely.
        man = _load(man_p) if man_p.exists() else {}
        fe = feats.get(name) or {}
        s = run_structure(_load(res), man)
        s.update(
            run=name,
            repo=fe.get("repo", name.split("/")[0]),
            engine=fe.get("engine_sha") or "?",
            total_tokens=man.get("total_tokens") or None,
            mods=fe.get("n_modules_billed") or s["modules_with_telemetry"] or 0,
            loc=(fe.get("scope") or {}).get("own_code_loc") or 0,
            wall_s=(man.get("timing") or {}).get("wall_clock_s"),
            api_s=(man.get("timing") or {}).get("api_time_s"),
        )
        # Probe runs abort before any module is analyzed, so they carry no
        # telemetry and no structure to learn from.
        if s["modules_with_telemetry"]:
            rows.append(s)

    # Runs without a token total inform the fan-out counts but cannot appear in
    # any tokens-per-X statistic, so every such analysis below uses `tokrows`.
    tokrows = [r for r in rows if r.get("total_tokens")]
    print(f"{len(rows)} runs with telemetry, {len(tokrows)} of them with a token total\n")
    print("=" * 128)
    print("PIPELINE FAN-OUT PER RUN")
    print("=" * 128)
    hdr = (
        f"{'run':<40}{'eng':<9}{'rev':>4}{'mods':>5}{'disc':>6}{'cands':>7}{'rep':>6}"
        f"{'pairs':>7}{'aprop':>7}{'tokens':>9}{'tok/mod':>9}{'tok/pair':>10}{'[1m]':>6}"
    )
    print(hdr)
    print("-" * 128)
    for r in sorted(rows, key=lambda x: -(x["total_tokens"] or 0)):
        tok = r["total_tokens"] or 0
        tpp = tok / r["pairs"] if r["pairs"] else 0
        rep = r["candidates_in_report"]
        # `rep` is the final report's candidate count. It must equal `cands`; a
        # mismatch means the running-total semantics were mishandled again.
        flag = "" if rep in (None, r["candidates_found"]) else "!"
        print(
            f"{r['run'][:39]:<40}{r['engine']:<9}"
            f"{('?' if r['review_iterations'] is None else r['review_iterations']):>4}"
            f"{r['mods']:>5}{r['disc_calls']:>6}"
            f"{r['candidates_found']:>7}{(str(rep) + flag if rep is not None else '-'):>6}"
            f"{r['pairs']:>7}{r['agent_prop_calls']:>7}"
            f"{tok / 1e6:>8.1f}M{tok / max(r['mods'], 1) / 1e6:>8.1f}M"
            f"{tpp / 1e6:>9.2f}M{r['long_context_calls']:>6}"
        )

    # ---------------------------------------------------- the review multiplier
    # This is the knob the advisor previously had to flag as unvalidated. It is
    # measurable within a single run: the first discovery pass is the no-review
    # cost, everything after it is what review added.
    print("\n" + "=" * 128)
    print("WHAT DOES --review-iterations COST?   (within-run, so no cross-repo noise)")
    print("=" * 128)
    print(f"{'run':<40}{'rev':>4}{'first Mtok':>12}{'review Mtok':>13}"
          f"{'disc x':>8}{'cands first':>12}{'cands final':>12}{'cand x':>8}"
          f"{'disc share':>12}")
    print("-" * 128)
    rev_rows = [r for r in tokrows if r.get("review_iterations")]
    for r in sorted(rev_rows, key=lambda x: -(x["total_tokens"] or 0)):
        first, extra = r["disc_tokens_first"], r["disc_tokens_review"]
        mult = (first + extra) / first if first else 0
        cmult = r["candidates_found"] / r["candidates_first"] if r["candidates_first"] else 0
        share = r["disc_tokens"] / r["total_tokens"]
        print(f"{r['run'][:39]:<40}{r['review_iterations']:>4}{first / 1e6:>12.1f}"
              f"{extra / 1e6:>13.1f}{mult:>7.2f}x{r['candidates_first']:>12}"
              f"{r['candidates_found']:>12}{cmult:>7.2f}x{100 * share:>11.1f}%")
    if rev_rows:
        mults = [(r["disc_tokens_first"] + r["disc_tokens_review"]) / r["disc_tokens_first"]
                 for r in rev_rows if r["disc_tokens_first"]]
        cmults = [r["candidates_found"] / r["candidates_first"]
                  for r in rev_rows if r["candidates_first"]]
        print(f"\n  review multiplies the discovery step by "
              f"{statistics.median(mults):.2f}x  (range {min(mults):.2f}-{max(mults):.2f}x, "
              f"n={len(mults)})")
        print(f"  and grows the candidate count by {statistics.median(cmults):.2f}x  "
              f"(range {min(cmults):.2f}-{max(cmults):.2f}x)")
        print("  -> review mostly REWRITES candidates rather than adding them: the")
        print("     `modified` lists cover nearly every existing candidate each pass.")

    print("\n" + "=" * 128)
    print("HOW WELL DOES EACH COUNT TRACK TOTAL TOKENS?")
    print("=" * 128)
    print(f"{'predictor':<26}{'spearman':>10}{'tokens/unit median':>21}{'min':>11}{'max':>11}{'CV':>8}{'max/min':>9}")
    print("-" * 128)
    for pred in ("mods", "loc", "disc_calls", "candidates_found", "pairs",
                 "agent_prop_calls", "disc_tokens", "num_candidates"):
        xs = [float(r[pred]) for r in tokrows]
        ys = [float(r["total_tokens"]) for r in tokrows]
        rho = spearman(xs, ys)
        med, lo, hi, cv = cv_of_ratio(tokrows, "total_tokens", pred)
        ratio = hi / lo if lo else 0
        unit = 1e6
        print(
            f"{pred:<26}{rho:>10.2f}{med / unit:>20.2f}M{lo / unit:>10.2f}M"
            f"{hi / unit:>10.2f}M{cv:>7.0f}%{ratio:>9.1f}x"
        )

    print("\n  (a low CV and a max/min near 1 means that count alone predicts tokens)")

    print("\n" + "=" * 128)
    print("WHERE DO THE PAIRS COME FROM?")
    print("=" * 128)
    print(f"{'run':<40}{'rev':>4}{'cands':>7}{'pairs':>7}{'pairs/cand':>12}{'pairs/mod':>11}{'cands/mod':>11}")
    print("-" * 128)
    for r in sorted(rows, key=lambda x: -x["pairs"]):
        pc = r["pairs"] / r["candidates_found"] if r["candidates_found"] else 0
        print(
            f"{r['run'][:39]:<40}"
            f"{('?' if r['review_iterations'] is None else r['review_iterations']):>4}"
            f"{r['candidates_found']:>7}{r['pairs']:>7}{pc:>12.1f}"
            f"{r['pairs'] / max(r['mods'], 1):>11.1f}{r['candidates_found'] / max(r['mods'], 1):>11.1f}"
        )

    print("\n" + "=" * 128)
    print("IS THE CHEAP FRONT A PROXY FOR THE WHOLE RUN?")
    print("=" * 128)
    med, lo, hi, cv = cv_of_ratio(tokrows, "total_tokens", "disc_tokens")
    print(f"  total_tokens / discovery_tokens:  median {med:.1f}x   range {lo:.1f}-{hi:.1f}x   CV {cv:.0f}%")
    print("  -> discovery is step 2 of 5; if this multiplier is stable, measuring it")
    print("     on a real repo predicts the rest without running the expensive steps.")

    print("\n" + "=" * 128)
    print("RUNTIME")
    print("=" * 128)
    print(f"{'run':<44}{'wall_h':>8}{'api_h':>8}{'conc':>7}{'tok/api_s':>11}")
    print("-" * 128)
    for r in sorted(tokrows, key=lambda x: -(x["api_s"] or 0)):
        w, a = r["wall_s"], r["api_s"]
        print(
            f"{r['run'][:43]:<44}{(w or 0) / 3600:>8.1f}{(a or 0) / 3600:>8.1f}"
            f"{(a / w if w and a else 0):>7.1f}{(r['total_tokens'] / a if a else 0):>11,.0f}"
        )
    med, lo, hi, cv = cv_of_ratio(
        [r for r in tokrows if r.get("api_s")], "total_tokens", "api_s"
    )
    print(f"\n  tokens per API-second: median {med:,.0f}  range {lo:,.0f}-{hi:,.0f}  CV {cv:.0f}%")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
