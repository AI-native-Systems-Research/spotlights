"""Aggregate historical Spotlights runs into cost/runtime calibration statistics.

Reads two corpora:

  1. `spotlights-paper/experiments/full-runs/` -- one `run_manifest.json` per
     completed run (coarse, but broad: many target repos).
  2. A local `artifacts/` tree -- durable per-invocation usage records
     (fine-grained: per step, per module, per CLI, with api_time_s).

Everything is normalized **per in-scope module**, because module count is the
one target-side quantity that is measurable on a repo we have never run.

Deliberately reports *tokens*, not dollars. Pricing is a rate-table lookup
applied at estimate time, so a contract change must not invalidate the
calibration. Dollars appear only in the audit columns that compare what a run
recorded against what its token buckets reprice to -- which is how the
long-context `[1m]` understatement surfaces.

Usage:
    python scripts/calibrate_cost_model.py --paper-runs <dir> [--artifacts <dir>]
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Module statuses that consumed real model calls. SKIPPED modules were never
# processed, so counting them would deflate every per-module figure.
BILLED_STATUSES = ("SUCCEEDED", "DEGRADED", "FAILED")
STEPS = (
    "candidate_discovery",
    "deep_research",
    "module_deep_research",
    "proposal_from_finding_creator",
    "agent_proposals",
)
BUCKETS = ("input", "output", "cache_read", "cache_create")


def _load(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Run:
    """One completed run, normalized across manifest schema generations."""

    path: str
    run_id: str = ""
    repo_url: str = ""
    engine_sha: str = ""
    date: str = ""
    total_tokens: int = 0
    # Recomputed from models_used buckets; may differ from `total_tokens`.
    summed_tokens: int = 0
    stored_contracted: float | None = None
    stored_public: float | None = None
    priced_share: float | None = None
    unpriced_models: list[str] = field(default_factory=list)
    long_context_tokens: int = 0
    models: Counter = field(default_factory=Counter)
    module_status: dict[str, int] = field(default_factory=dict)
    tokens_by_role: Counter = field(default_factory=Counter)
    num_candidates: int | None = None
    wall_s: float | None = None
    api_s: float | None = None
    accum_s: float | None = None

    @property
    def billed_modules(self) -> int:
        return sum(self.module_status.get(k, 0) for k in BILLED_STATUSES)

    @property
    def all_modules(self) -> int:
        return sum(self.module_status.values())

    @property
    def tokens_per_module(self) -> float | None:
        n = self.billed_modules
        return self.total_tokens / n if n else None

    @property
    def concurrency(self) -> float | None:
        if self.api_s and self.wall_s:
            return self.api_s / self.wall_s
        return None

    @property
    def repo_key(self) -> str:
        """Group key for backtesting: replicates of one target must not be
        split across train/test."""
        url = self.repo_url or self.path
        return url.rstrip("/").removesuffix(".git").split("/")[-1].lower() or self.path


def read_manifest(path: Path, root: Path) -> Run:
    d = _load(path)
    cost = d.get("cost") or {}
    ext = d.get("external_cost") or {}
    run = Run(
        path=str(path.relative_to(root)).replace("\\", "/"),
        run_id=d.get("run_id") or "",
        repo_url=(d.get("target") or {}).get("repo_url") or "",
        engine_sha=((d.get("spotlights") or {}).get("commit_sha") or "")[:8],
        date=(d.get("date") or "")[:10],
        total_tokens=d.get("total_tokens") or 0,
        stored_contracted=cost.get("amount_usd"),
        stored_public=ext.get("amount_usd"),
        priced_share=cost.get("priced_token_share"),
        unpriced_models=list((cost.get("coverage") or {}).get("unpriced_models") or []),
        module_status=dict((d.get("outputs") or {}).get("module_status") or {}),
        num_candidates=(d.get("outputs") or {}).get("num_candidates"),
    )
    timing = d.get("timing") or {}
    run.wall_s = timing.get("wall_clock_s")
    run.api_s = timing.get("api_time_s")
    run.accum_s = timing.get("accumulated_duration_s")

    for entry in d.get("models_used") or []:
        usage = entry.get("usage") or {}
        tok = sum(int(usage.get(b) or 0) for b in BUCKETS)
        model = entry.get("model") or "?"
        run.models[f"{entry.get('provider')}:{model}"] += tok
        run.tokens_by_role[entry.get("role") or "?"] += tok
        run.summed_tokens += tok
        if "[1m]" in model:
            run.long_context_tokens += tok
    return run


def dedupe(runs: list[Run]) -> tuple[list[Run], list[str]]:
    """Drop zero-token probes and byte-identical copies of the same run.

    A shared `run_id` alone does NOT mean duplicate -- the colpali fleet
    deliberately reuses one id across five distinct runs -- so identity is
    (run_id, total_tokens, wall_clock).
    """
    kept: list[Run] = []
    dropped: list[str] = []
    seen: dict[tuple, str] = {}
    for r in runs:
        if r.total_tokens == 0:
            dropped.append(f"{r.path}  (zero-token probe)")
            continue
        key = (r.run_id, r.total_tokens, round(r.wall_s or 0, 3))
        if key in seen:
            dropped.append(f"{r.path}  (duplicate of {seen[key]})")
            continue
        seen[key] = r.path
        kept.append(r)
    return kept, dropped


def read_usage_records(artifacts: Path) -> list[dict]:
    """Per-invocation usage records from a local artifacts tree."""
    pattern = "*/spotlights_manager/modules/*/*.usage/*.json"
    out = []
    for p in artifacts.glob(pattern):
        try:
            d = _load(p)
        except (OSError, json.JSONDecodeError):
            continue
        d["_run"] = p.parts[len(artifacts.parts)]
        out.append(d)
    return out


def fmt(x: float | None, spec: str = ".1f") -> str:
    return "-" if x is None else format(x, spec)


def pct_spread(values: list[float]) -> str:
    if len(values) < 2:
        return "-"
    lo, hi = min(values), max(values)
    mid = statistics.median(values)
    return f"{lo:.1f}-{hi:.1f} (median {mid:.1f}, ±{100 * (hi - lo) / 2 / mid:.0f}%)"


def report_runs(runs: list[Run]) -> None:
    print("\n" + "=" * 118)
    print("RUN INVENTORY  (tokens recomputed from models_used buckets)")
    print("=" * 118)
    hdr = (
        f"{'run':<46}{'repo':<14}{'eng':<9}{'tokens':>9}{'mods':>6}"
        f"{'tok/mod':>9}{'share':>7}{'[1m]%':>7}{'conc':>6}"
    )
    print(hdr)
    print("-" * 118)
    for r in sorted(runs, key=lambda x: -x.total_tokens):
        lc = 100 * r.long_context_tokens / r.summed_tokens if r.summed_tokens else 0
        print(
            f"{r.path[:45]:<46}{r.repo_key[:13]:<14}{r.engine_sha:<9}"
            f"{r.total_tokens / 1e6:>8.1f}M{r.billed_modules:>6}"
            f"{fmt((r.tokens_per_module or 0) / 1e6, '.2f') + 'M':>9}"
            f"{fmt(r.priced_share, '.2f'):>7}{lc:>6.0f}%{fmt(r.concurrency, '.1f') + 'x':>6}"
        )


def report_consistency(runs: list[Run]) -> None:
    print("\n" + "=" * 118)
    print("DATA INTEGRITY")
    print("=" * 118)
    for r in runs:
        issues = []
        if r.summed_tokens and abs(r.summed_tokens - r.total_tokens) > 1:
            d = r.summed_tokens - r.total_tokens
            issues.append(f"total_tokens off by {d:+,} vs models_used sum")
        if r.priced_share is not None and r.priced_share < 1.0:
            issues.append(f"priced_token_share={r.priced_share:.3f} -> cost is an UNDERCOUNT")
        if r.unpriced_models:
            issues.append(f"unpriced: {', '.join(r.unpriced_models)}")
        if r.long_context_tokens:
            issues.append(
                f"{100 * r.long_context_tokens / r.summed_tokens:.0f}% of tokens on [1m] ids"
            )
        if not r.module_status:
            issues.append("NO module_status -> per-module normalization impossible")
        if r.stored_public is None:
            issues.append("no external_cost (public $ needs recompute)")
        if issues:
            print(f"\n  {r.path}")
            for i in issues:
                print(f"      - {i}")


def report_per_module(runs: list[Run]) -> None:
    usable = [r for r in runs if r.tokens_per_module]
    print("\n" + "=" * 118)
    print(f"PER-MODULE TOKEN MODEL   (n={len(usable)} runs with module_status)")
    print("=" * 118)
    if not usable:
        print("  none -- no run carried outputs.module_status")
        return
    per = [(r.tokens_per_module or 0) / 1e6 for r in usable]
    print(f"  tokens/module (M): {pct_spread(per)}")
    print(f"  ratio max/min:     {max(per) / min(per):.1f}x")

    by_repo: dict[str, list[float]] = defaultdict(list)
    for r in usable:
        by_repo[r.repo_key].append((r.tokens_per_module or 0) / 1e6)
    print("\n  by target repo (replicates reveal the noise floor):")
    for repo, vals in sorted(by_repo.items(), key=lambda kv: -statistics.median(kv[1])):
        tag = f"  <-- {len(vals)} replicates: NOISE FLOOR" if len(vals) > 1 else ""
        print(f"    {repo:<20} n={len(vals)}  {pct_spread(vals)}{tag}")


def report_roles(runs: list[Run]) -> None:
    print("\n" + "=" * 118)
    print("STEP SHARE OF TOKENS  (which step to squeeze)")
    print("=" * 118)
    agg: Counter = Counter()
    for r in runs:
        agg.update(r.tokens_by_role)
    tot = sum(agg.values()) or 1
    for role, tok in agg.most_common():
        print(f"  {role:<34}{tok / 1e6:>9.1f}M{100 * tok / tot:>7.1f}%")


def report_timing(runs: list[Run]) -> None:
    print("\n" + "=" * 118)
    print("RUNTIME MODEL")
    print("=" * 118)
    conc = [r.concurrency for r in runs if r.concurrency]
    if conc:
        print(f"  effective concurrency (api/wall): {pct_spread(conc)}x")
    per_mod_wall = [
        r.wall_s / 3600 / r.billed_modules for r in runs if r.wall_s and r.billed_modules
    ]
    if per_mod_wall:
        print(f"  wall-clock HOURS per module:      {pct_spread(per_mod_wall)}")


def report_usage_records(records: list[dict]) -> None:
    if not records:
        return
    print("\n" + "=" * 118)
    print(f"FINE-GRAINED FAN-OUT  (local artifacts, {len(records)} invocations)")
    print("=" * 118)
    per_mod: dict[tuple, Counter] = defaultdict(Counter)
    tok: dict[tuple, int] = defaultdict(int)
    secs: dict[tuple, float] = defaultdict(float)
    for d in records:
        key = (d["step"], d["cli"])
        mod = (d["_run"], d["module_qualified_name"])
        per_mod[key][mod] += 1
        tok[key] += sum(int(d.get(b) or 0) for b in BUCKETS)
        secs[key] += float(d.get("api_time_s") or 0)
    print(f"{'step':<32}{'cli':<8}{'calls':>7}{'calls/mod':>11}{'tok/call':>11}{'s/call':>8}")
    print("-" * 118)
    for key in sorted(per_mod):
        counts = per_mod[key]
        calls = sum(counts.values())
        print(
            f"{key[0]:<32}{key[1]:<8}{calls:>7}{calls / len(counts):>11.1f}"
            f"{tok[key] / calls:>11,.0f}{secs[key] / calls:>8.0f}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paper-runs", type=Path, required=True)
    ap.add_argument("--artifacts", type=Path, default=None)
    args = ap.parse_args()

    root = args.paper_runs
    manifests = sorted(root.glob("**/run_manifest.json"))
    if not manifests:
        print(f"no run_manifest.json under {root}")
        return 1
    runs = [read_manifest(p, root) for p in manifests]
    runs, dropped = dedupe(runs)

    print(f"found {len(manifests)} manifests -> {len(runs)} distinct runs")
    for d in dropped:
        print(f"  dropped: {d}")

    report_runs(runs)
    report_consistency(runs)
    report_per_module(runs)
    report_roles(runs)
    report_timing(runs)

    if args.artifacts:
        report_usage_records(read_usage_records(args.artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
