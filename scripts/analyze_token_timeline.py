#!/usr/bin/env python3
"""Correlate the run's token throughput -- not its call count -- against the 429s.

Why this exists
---------------
`analyze_stream_timeline.py` counted *claude sessions in flight* and found no link
to the rate limits. That measurement was wrong on two counts:

1. Rate limits are priced in tokens, not requests. The IOCR run moved ~53.7M
   tokens in 4.3 hours (~208k/min sustained). Session count says nothing about it.
2. Codex streams carry no `"timestamp"`, so 80 codex sessions -- the ones
   averaging 189k input tokens each -- were invisible to that count entirely.

This script fixes both. `candidate_discovery/iterations.jsonl` records every
iteration's `agent`, `duration_s` and full token breakdown, and iterations run
strictly in sequence inside a module. So the claude iterations, which do carry
wall-clock timestamps, anchor the codex ones: place the anchored iteration on the
real clock, then walk the sequence forwards and backwards by `duration_s`.

The residual check is what makes this trustworthy -- when a module has two or
more anchors, the placement predicts the second anchor's start time, and the
error is reported rather than hidden.

Usage:
    python scripts/analyze_token_timeline.py <artifacts>/spotlights_manager
"""

from __future__ import annotations

import glob
import json
import os
import re
import statistics as st
import sys
from datetime import datetime

TS_RE = re.compile(rb'"timestamp":"([0-9T:.\-]+Z)"')
AGENT_DIR = {"claude_code": "claude_code", "claude": "claude_code", "codex": "codex"}
# Token fields as iterations.jsonl names them.
FRESH = ("input_tokens", "cache_create_tokens", "output_tokens")
CACHED = ("cache_read_tokens",)


def _ts(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def stream_window(path_glob: str) -> tuple[float, float] | None:
    """Observed first/last wall-clock stamp across a stream directory."""
    stamps: list[float] = []
    for path in glob.glob(path_glob, recursive=True):
        raw = open(path, "rb").read()
        stamps += [_ts(m.group(1).decode()) for m in TS_RE.finditer(raw)]
    return (min(stamps), max(stamps)) if stamps else None


def place_module(mod_dir: str) -> tuple[list[dict], list[float]]:
    """Every discovery iteration of one module, placed on the wall clock.

    Returns the placed iterations and the anchor residuals (seconds) so the
    caller can report how well the sequence assumption held.
    """
    jsonl = os.path.join(mod_dir, "candidate_discovery", "iterations.jsonl")
    if not os.path.exists(jsonl):
        return [], []
    rows: list[dict] = []
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["n"])
    if not rows:
        return [], []

    disc = os.path.join(mod_dir, "candidate_discovery")
    for r in rows:
        agent = AGENT_DIR.get(str(r.get("agent", "")), str(r.get("agent", "")))
        r["_dir"] = os.path.join(disc, f"iter_{r['n']}_{agent}")
        r["_observed"] = stream_window(os.path.join(r["_dir"], "**", "*"))
        r["_dur"] = float(r.get("duration_s") or 0.0)

    anchors = [r for r in rows if r["_observed"]]
    if not anchors:
        return [], []

    # Walk the sequence out from the first anchor: iteration k starts where k-1 ended.
    base = anchors[0]
    idx = rows.index(base)
    offsets: dict[int, float] = {base["n"]: base["_observed"][0]}
    for i in range(idx + 1, len(rows)):
        prev = rows[i - 1]
        offsets[rows[i]["n"]] = offsets[prev["n"]] + prev["_dur"]
    for i in range(idx - 1, -1, -1):
        nxt = rows[i + 1]
        offsets[rows[i]["n"]] = offsets[nxt["n"]] - rows[i]["_dur"]

    residuals = [
        abs(offsets[a["n"]] - a["_observed"][0]) for a in anchors[1:] if a["n"] in offsets
    ]

    placed = []
    for r in rows:
        start = offsets.get(r["n"])
        if start is None:
            continue
        placed.append(
            dict(
                mod=os.path.basename(mod_dir),
                n=r["n"],
                agent=r.get("agent"),
                t0=start,
                t1=start + r["_dur"],
                dur=r["_dur"],
                api_s=float(r.get("api_time_s") or 0.0),
                fresh=sum(int(r.get(k) or 0) for k in FRESH),
                cached=sum(int(r.get(k) or 0) for k in CACHED),
                anchored=bool(r["_observed"]),
            )
        )
    return placed, residuals


def rate_limits(root: str) -> list[float]:
    """Wall-clock moment of every 429 the streams recorded."""
    out: list[float] = []
    paths = glob.glob(os.path.join(root, "modules", "*", "**", "raw_stdout.log"), recursive=True)
    paths += glob.glob(os.path.join(root, "modules", "*", "**", "*.stdout"), recursive=True)
    for path in paths:
        seen: float | None = None
        for line in open(path, "rb"):
            found = TS_RE.search(line)
            if found:
                seen = _ts(found.group(1).decode())
            if b'"error_status":429' in line or b"429 Too Many Requests" in line:
                if seen:
                    out.append(seen)
    return sorted(out)


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0


def main(root: str) -> None:
    placed: list[dict] = []
    residuals: list[float] = []
    for mod_dir in sorted(glob.glob(os.path.join(root, "modules", "*"))):
        p, r = place_module(mod_dir)
        placed += p
        residuals += r
    if not placed:
        sys.exit(f"no placeable iterations under {root}")

    r429 = rate_limits(root)
    t0 = min(p["t0"] for p in placed)
    t1 = max(p["t1"] for p in placed)
    span_min = (t1 - t0) / 60

    print("=== placement quality ===")
    mods = len({p["mod"] for p in placed})
    print(f"iterations placed        : {len(placed)} across {mods} modules")
    print(f"  of which anchored      : {sum(1 for p in placed if p['anchored'])}")
    print(f"  codex (was invisible)  : {sum(1 for p in placed if p['agent'] == 'codex')}")
    if residuals:
        print(
            f"anchor residual (s)      : median {st.median(residuals):.0f}  "
            f"max {max(residuals):.0f}  n={len(residuals)}"
        )
        print("  (how far the sequence assumption misses a second known start)")
    else:
        print("anchor residual          : n/a (one anchor per module)")
    print(f"run span                 : {span_min:.0f} min")
    print(f"429s with a timestamp    : {len(r429)}")

    # --- per-minute series ----------------------------------------------------
    # Tokens are spread evenly across an iteration's duration. That understates
    # true peaks, since tokens actually move during api_time_s only -- so any
    # correlation found here is a floor, not a ceiling.
    minutes = int(span_min) + 1
    fresh = [0.0] * minutes
    cached = [0.0] * minutes
    calls = [0.0] * minutes
    for p in placed:
        a, b = (p["t0"] - t0) / 60, (p["t1"] - t0) / 60
        width = max(b - a, 1e-6)
        for m in range(int(a), min(int(b) + 1, minutes)):
            overlap = max(0.0, min(b, m + 1) - max(a, m))
            share = overlap / width
            fresh[m] += p["fresh"] * share
            cached[m] += p["cached"] * share
            calls[m] += 1 if overlap > 0 else 0
    hits = [0.0] * minutes
    for t in r429:
        m = int((t - t0) / 60)
        if 0 <= m < minutes:
            hits[m] += 1

    total = [f + c for f, c in zip(fresh, cached, strict=True)]
    print("\n=== token throughput, per minute ===")
    series_by_name = (
        ("fresh input+output", fresh),
        ("cache reads", cached),
        ("all tokens", total),
    )
    for name, series in series_by_name:
        nz = [v for v in series if v > 0]
        if nz:
            print(
                f"  {name:20s} mean {st.mean(nz):9,.0f}  median {st.median(nz):9,.0f}  "
                f"peak {max(nz):10,.0f}"
            )

    # --- the comparison that matters -----------------------------------------
    print("\n=== what predicts a 429: tokens, or call count? ===")
    print("  bucket |   all tokens |  fresh tokens |   call count")
    print("  -------+--------------+---------------+-------------")
    for width in (1, 2, 5, 10):
        def agg(series: list[float], w: int = width) -> list[float]:
            return [sum(series[i : i + w]) for i in range(0, minutes, w)]

        y = agg(hits)
        print(
            f"  {width:2d} min | {_pearson(agg(total), y):+12.3f} | "
            f"{_pearson(agg(fresh), y):+13.3f} | {_pearson(agg(calls), y):+12.3f}"
        )

    # --- dose response on tokens ---------------------------------------------
    print("\n=== dose-response: 429s per minute, by that minute's token load ===")
    live = [(total[m], hits[m]) for m in range(minutes) if total[m] > 0]
    live.sort()
    n = len(live)
    if n >= 5:
        for q in range(5):
            lo, hi = n * q // 5, n * (q + 1) // 5
            chunk = live[lo:hi]
            toks = [c[0] for c in chunk]
            h = [c[1] for c in chunk]
            print(
                f"  quintile {q + 1}: {st.mean(toks):10,.0f} tok/min  ->  "
                f"{st.mean(h):5.2f} rate limits/min   ({len(chunk)} minutes)"
            )

    # --- tokens in flight at the moment of each 429 --------------------------
    print("\n=== token load at the moment of a 429 vs a quiet minute ===")
    hit_min = {int((t - t0) / 60) for t in r429}
    busy = [total[m] for m in range(minutes) if m in hit_min and total[m] > 0]
    quiet = [total[m] for m in range(minutes) if m not in hit_min and total[m] > 0]
    if busy and quiet:
        print(f"  minutes with a 429 : {st.mean(busy):10,.0f} tok/min  (n={len(busy)})")
        print(f"  minutes without    : {st.mean(quiet):10,.0f} tok/min  (n={len(quiet)})")
        print(f"  ratio              : {st.mean(busy) / st.mean(quiet):.2f}x")

    print("\n=== biggest token minutes ===")
    top = sorted(range(minutes), key=lambda m: -total[m])[:8]
    for m in sorted(top):
        print(
            f"  +{m:3d} min: {total[m]:10,.0f} tok/min  "
            f"({calls[m]:.0f} calls)  {hits[m]:.0f} rate limits"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1].replace("\\", "/"))
