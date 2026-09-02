"""Scoping advisor: what a Spotlights run on your repo will cost, and which
knobs change that.

This is deliberately NOT a calculator. Predicting a run's absolute cost from
repo features was built and backtested, and it failed: leave-one-repo-out
median error 47% and worst 345%, against a measured noise floor of ~8%. Repo
size is not a predictor -- tokens per kLOC spans 24x across targets and runs
*inversely* with size, because an agent reads a bounded slice of a module
however big the module is.

The band width here is itself backtested (`scripts/advisor_backtest.py`,
leave-one-repo-out over 12 runs / 7 repos). A p10-p90 band covered only 58% of
held-out runs -- it understated its own uncertainty -- so this quotes the
observed min-max instead, which covers 83% but is ~23x wide. That width is the
honest answer: use it as a ceiling, not a budget.

What survived validation is relative structure, so that is all this reports:

  * a token RANGE (observed min-max), never a point estimate
  * relative multipliers between configs, each labelled with how it was
    established -- exact / measured / derived / unvalidated
  * exact model pricing, which is a rate-table lookup and carries no
    estimation error at all

The advisor's job is to tell a user who knows their repo but not Spotlights
which levers exist and what each one is worth. The number it should act on is
`--max-cost`, a hard ceiling, not any figure printed here.

Usage:
    python scripts/advisor.py --modules 12
    python scripts/advisor.py --modules 34 --exclude 8 --claude-model aws/claude-opus-5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
RATES = REPO_ROOT / "src/spotlights_engine/costing/rates.json"
EXTERNAL_RATES = REPO_ROOT / "src/spotlights_engine/costing/external_rates.json"
BUCKETS = ("input", "output", "cache_read", "cache_create")


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def load_rate_table(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def blended_rate(rate: dict, mix: dict[str, float]) -> float:
    """Dollars per token for a given bucket mix.

    Weighting by the measured mix matters more than any refinement to the token
    count: 87% of a run's tokens are cache reads at ~0.1x the input rate, so
    pricing a run at the headline input rate overstates it several-fold.
    """
    return sum(float(rate.get(b) or 0.0) * mix.get(b, 0.0) for b in BUCKETS)


def pick_rate(table: dict[str, dict], wanted: str) -> tuple[str, dict] | None:
    """Find a rate row by loose model-name match (keys are `provider:model`)."""
    if not wanted:
        return None
    w = wanted.lower()
    for key, val in table.items():
        if w in key.lower():
            return key, val
    return None


# --------------------------------------------------------------------------- #
# Config ladder
# --------------------------------------------------------------------------- #
class Row:
    """One config on the ladder."""

    def __init__(
        self,
        label: str,
        flags: str,
        factor: float,
        basis: str,
        confidence: str,
        anchor: str = "full_pipeline",
    ) -> None:
        self.label = label
        self.flags = flags
        self.factor = factor
        self.basis = basis
        self.confidence = confidence
        self.anchor = anchor


def build_ladder(cal: dict[str, Any]) -> list[Row]:
    """The what-if ladder, with every multiplier derived from the snapshot.

    Nothing here is a guess pulled from the air: each factor is either measured
    on real runs, arithmetic on measured step shares, or explicitly flagged as
    direction-only where the corpus has no variation to measure.
    """
    steps = cal["step_shares_per_run"]
    prop = steps["proposal_from_finding_creator"]["p50"]
    disc = steps["candidate_discovery"]["p50"]
    dr = steps["module_deep_research"]["p50"]

    fan = cal["fanout"]
    cands_per_mod = fan["candidates_per_module"]["p50"]
    findings_per_cand = fan["pairs_per_candidate"]["p50"]
    pairs_per_mod = fan["pairs_per_module"]["p50"]

    # Measured, same repo and module count, deep research on vs off.
    per_run = cal["per_run"]
    on = [r for r in per_run if r["deep_research"]]
    off = [r for r in per_run if not r["deep_research"]]
    measured_no_dr = None
    for a in off:
        for b in on:
            if a["repo"] == b["repo"] and a["mods"] == b["mods"]:
                measured_no_dr = a["total_tokens"] / b["total_tokens"]
                break
    predicted_no_dr = 1.0 - prop - dr

    rows = [
        Row(
            "discovery only (no deep research)",
            "--no-deep-research",
            measured_no_dr if measured_no_dr else predicted_no_dr,
            (
                f"measured on one repo at equal module count "
                f"(predicted {predicted_no_dr:.2f}x from step shares)"
                if measured_no_dr
                else f"drops the {100 * (prop + dr):.0f}% of spend in steps 3+4"
            ),
            "measured" if measured_no_dr else "derived",
            anchor="deep_research_off",
        ),
    ]

    # Findings cap. It only bites when it is below the natural yield, and the
    # default (30) is ABOVE the observed median findings per candidate, so
    # nudging it down from 30 changes nothing for a typical repo.
    for cap in (5, 10, 20):
        if cap >= findings_per_cand:
            note = (
                f"no effect: observed yield is {findings_per_cand:.0f} findings/"
                f"candidate, already under this cap"
            )
            factor = 1.0
            conf = "derived"
        else:
            scaled = cands_per_mod * cap
            factor = 1.0 - prop + prop * (scaled / pairs_per_mod)
            note = (
                f"caps step 4 at ~{scaled:.0f} pairs/module vs {pairs_per_mod:.0f} "
                f"observed; scales the {100 * prop:.0f}% step"
            )
            conf = "derived"
        rows.append(
            Row(f"findings capped at {cap}", f"--max-findings-per-module {cap}",
                factor, note, conf)
        )

    rows.append(Row("default", "(no flags)", 1.0, "the anchor band itself", "measured"))
    rows.append(
        Row(
            "no discovery review",
            "--no-review",
            1.0 - disc / 2,
            f"halves the {100 * disc:.0f}% discovery step; knock-on to steps 4/5 "
            f"unknown (fewer candidates found)",
            "unvalidated",
        )
    )
    rows.append(
        Row(
            "deep research on both CLIs",
            "--enable-claude-search",
            1.0 + dr,
            f"roughly doubles the {100 * dr:.0f}% step 3 (Codex-only by default)",
            "unvalidated",
        )
    )
    return rows


# --------------------------------------------------------------------------- #
def fmt_money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", type=Path,
                    default=REPO_ROOT / "artifacts/advisor_calibration.json")
    ap.add_argument("--modules", type=int, required=True,
                    help="In-scope module count, from `spotlights modules extract`.")
    ap.add_argument("--exclude", type=int, default=0,
                    help="Modules you plan to drop from scope.")
    ap.add_argument("--claude-model", default="claude-opus-5")
    ap.add_argument("--codex-model", default="codex")
    ap.add_argument("--parallel", type=float, default=None,
                    help="Expected effective concurrency (default: historical median).")
    ap.add_argument("--band", choices=("observed", "p10p90"), default="observed",
                    help="`observed` = min-max, backtested at 83%% coverage (default). "
                         "`p10p90` is narrower but only covered 58%% of held-out runs.")
    args = ap.parse_args()

    cal = json.loads(args.calibration.read_text(encoding="utf-8"))
    mix = cal["bucket_mix"]
    # Which quantiles bound the quoted range. See --band: p10-p90 was backtested
    # and covered only 58% of held-out runs, so min-max is the default.
    LO, HI = ("min", "max") if args.band == "observed" else ("p10", "p90")
    COVERAGE = "83%" if args.band == "observed" else "58%"
    in_scope = max(args.modules - args.exclude, 0)
    if not in_scope:
        print("nothing in scope after exclusions")
        return 1

    external = load_rate_table(EXTERNAL_RATES)
    contracted = load_rate_table(RATES)

    # Both CLIs run, so a run's blended rate is a mix of the two models'. Split
    # evenly: the observed per-CLI token split is close to even and the
    # difference is well inside the band width.
    def blended_for(table: dict[str, dict]) -> tuple[float, list[str]]:
        used, total, n = [], 0.0, 0
        for wanted in (args.claude_model, args.codex_model):
            hit = pick_rate(table, wanted)
            if hit:
                used.append(hit[0])
                total += blended_rate(hit[1], mix)
                n += 1
        return (total / n if n else 0.0), used

    ext_rate, ext_keys = blended_for(external)
    con_rate, con_keys = blended_for(contracted)

    print("=" * 96)
    print("SPOTLIGHTS SCOPING ADVISOR")
    print("=" * 96)
    print(f"  in-scope modules      {in_scope}"
          + (f"   ({args.modules} found, {args.exclude} excluded)" if args.exclude else ""))
    print(f"  models                claude={args.claude_model}  codex={args.codex_model}")
    print(f"  token mix             {100 * mix['cache_read']:.0f}% cache read, "
          f"{100 * mix['cache_create']:.0f}% cache write, "
          f"{100 * mix['input']:.0f}% input, {100 * mix['output']:.0f}% output")
    if ext_rate:
        print(f"  blended list price    ${ext_rate * 1e6:.2f} / MTok   "
              f"(matched {', '.join(ext_keys)})")
    if con_rate:
        print(f"  blended contracted    ${con_rate * 1e6:.2f} / MTok")
    print(f"\n  calibration: {cal['corpus']['runs_total']} runs / "
          f"{cal['corpus']['repos']} repos / "
          f"{cal['corpus']['engine_versions']} engine versions")

    anchors = cal["tokens_per_module"]

    print("\n" + "=" * 96)
    print(f"CONFIG LADDER   (ranges are the observed spread of history, backtested at")
    print(f"                 {COVERAGE} coverage -- NOT confidence intervals)")
    print("=" * 96)
    print(f"{'config':<34}{'rel':>6}{'tokens range':>22}{'list price range':>26}")
    print("-" * 96)
    rows = build_ladder(cal)
    for row in sorted(rows, key=lambda r: r.factor):
        anchor = anchors.get(row.anchor) or anchors["full_pipeline"]
        # A row with its own measured anchor already includes the effect, so its
        # factor must not be applied twice.
        f = 1.0 if row.anchor != "full_pipeline" else row.factor
        lo = anchor[LO] * in_scope * f
        hi = anchor[HI] * in_scope * f
        price = (
            f"{fmt_money(lo * ext_rate)} - {fmt_money(hi * ext_rate)}"
            if ext_rate else "n/a"
        )
        print(f"{row.label:<34}{row.factor:>5.2f}x"
              f"{lo / 1e6:>9.0f}M -{hi / 1e6:>8.0f}M{price:>26}")

    print("\n" + "=" * 96)
    print("HOW EACH NUMBER WAS ESTABLISHED")
    print("=" * 96)
    order = {"exact": 0, "measured": 1, "derived": 2, "unvalidated": 3}
    for row in sorted(rows, key=lambda r: (order.get(r.confidence, 9), r.factor)):
        print(f"  [{row.confidence:<11}] {row.flags:<32} {row.basis}")
    print(f"  [{'exact':<11}] {'--claude-model / --codex-model':<32} "
          f"rate-table lookup; the price ratio between models carries no error")

    # --------------------------------------------------------------- exclude
    print("\n" + "=" * 96)
    print("SCOPE: WHICH MODULES TO EXCLUDE")
    print("=" * 96)
    per_run = [r for r in cal["per_run"] if r["deep_research"]]
    print(f"  Per-module cost varies only ~2x within a repo, so dropping k of n")
    print(f"  modules saves roughly k/n of the per-module steps "
          f"({100 * (cal['step_shares_per_run']['candidate_discovery']['p50'] + cal['step_shares_per_run']['module_deep_research']['p50']):.0f}% of spend).")
    print(f"\n  But step 4 -- "
          f"{100 * cal['step_shares_per_run']['proposal_from_finding_creator']['p50']:.0f}% "
          f"of spend -- already skips modules that yield nothing, so excluding")
    print("  barren modules saves less than you would expect. Exclude for signal")
    print("  quality first, cost second.")
    if args.exclude:
        anchor = anchors["full_pipeline"]
        saved_lo = anchor[LO] * args.exclude
        saved_hi = anchor[HI] * args.exclude
        print(f"\n  your {args.exclude} exclusions: saves roughly "
              f"{saved_lo / 1e6:.0f}M-{saved_hi / 1e6:.0f}M tokens "
              f"({fmt_money(saved_lo * ext_rate)}-{fmt_money(saved_hi * ext_rate)})")

    # --------------------------------------------------------------- runtime
    rt = cal["runtime"]["tokens_per_api_second"]
    conc = args.parallel or cal["runtime"]["observed_concurrency"]["p50"]
    print("\n" + "=" * 96)
    print("RUNTIME")
    print("=" * 96)
    anchor = anchors["full_pipeline"]
    for name, tok in (("low ", anchor[LO] * in_scope), ("high", anchor[HI] * in_scope)):
        api_h = tok / rt["p50"] / 3600
        print(f"  {name} tokens -> {api_h:>6.1f} API-hours -> "
              f"{api_h / conc:>6.1f} wall-hours at {conc:.1f}x concurrency")
    print(f"\n  concurrency is yours to choose: --max-parallel (default 1), "
          f"--max-parallel-pairs (5),")
    print(f"  --max-parallel-candidates (5). Observed range "
          f"{cal['runtime']['observed_concurrency']['min']:.1f}x-"
          f"{cal['runtime']['observed_concurrency']['max']:.1f}x.")

    # --------------------------------------------------------------- guardrail
    anchor = anchors["full_pipeline"]
    ceiling = anchor[HI] * in_scope * ext_rate
    print("\n" + "=" * 96)
    print("WHAT TO ACTUALLY DO")
    print("=" * 96)
    print(f"  1. Run with a hard ceiling:  --max-cost {ceiling:.0f}")
    print("     That is a guarantee. Every number above is not.")
    print("  2. Start with --no-deep-research to see candidate quality cheaply,")
    print("     then re-run the full pipeline on the modules that looked useful.")
    print("     This is the best-evidenced lever: predicted 0.28x, measured 0.32x.")
    print("  3. The range above is wide -- backtested honestly, it has to be. Two")
    print("     reasons: this corpus does not record --review-iterations or")
    print("     --max-findings-per-module (runs differing 3x in cost look identical),")
    print("     and tokens per module spans 23x across repos, so your module count")
    print("     barely narrows it. Use the ratios to choose a config, --max-cost to")
    print("     bound the bill, and do not treat the absolute range as a forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
