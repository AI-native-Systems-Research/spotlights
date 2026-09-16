#!/usr/bin/env python3
"""Reconstruct an archived run's agent-CLI timeline from the persisted streams.

Both CLIs write newline-delimited JSON, and claude's events carry a wall-clock
`"timestamp"`. That is the only per-call clock in the artifacts -- `run_manifest`
holds aggregates only, `manifest.json` holds none, and every module `status.json`
shares one plan-time `started_at`, so a module window is queue wait plus work and
must not be read as concurrency.

Timestamping each stream instead yields real intervals for every claude session,
and every `api_retry` event carries the exact moment it was rate limited. With
both, "did our own concurrency cause the 429s" becomes a measurement rather than
an assumption.

Usage:
    python scripts/analyze_stream_timeline.py <artifacts_dir>/spotlights_manager
"""

from __future__ import annotations

import glob
import json
import os
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict

TS_RE = re.compile(rb'"timestamp":"([0-9T:.\-]+Z)"')
DELAY_RE = re.compile(rb'"retry_delay_ms":(\d+)')
ATTEMPT_RE = re.compile(rb'"attempt":(\d+)')


def _parse_ts(text: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def collect(root: str) -> tuple[list[dict], list[dict]]:
    """Per-stream intervals and every failure event, both on a wall clock."""
    paths = [
        p.replace("\\", "/")
        for p in glob.glob(
            os.path.join(root, "modules", "*", "**", "raw_stdout.log"), recursive=True
        )
        + glob.glob(os.path.join(root, "modules", "*", "**", "*.stdout"), recursive=True)
    ]
    spans: list[dict] = []
    events: list[dict] = []
    for path in sorted(paths):
        rel = os.path.relpath(path, root).replace("\\", "/")
        raw = open(path, "rb").read()
        stamps = [_parse_ts(m.group(1).decode()) for m in TS_RE.finditer(raw)]
        step = "step5" if rel.endswith(".stdout") else "step2"
        if stamps:
            spans.append(
                dict(
                    rel=rel,
                    mod=rel.split("/")[1],
                    step=step,
                    t0=min(stamps),
                    t1=max(stamps),
                )
            )
        seen: float | None = None
        for line in raw.splitlines():
            line = line.strip()
            if not line.startswith(b"{"):
                continue
            found = TS_RE.search(line)
            if found:
                seen = _parse_ts(found.group(1).decode())
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            # codex --json nests events under "msg"; claude does not.
            for ev in (obj, obj.get("msg")):
                if not isinstance(ev, dict):
                    continue
                kind = status = reason = None
                if ev.get("type") == "system" and ev.get("subtype") == "api_retry":
                    kind = "api_retry"
                    status = ev.get("error_status")
                    reason = ev.get("error")
                elif ev.get("type") == "error":
                    kind, reason = "error", str(ev.get("message") or "")[:90]
                elif ev.get("type") == "turn.failed":
                    err = ev.get("error")
                    kind = "turn.failed"
                    reason = str(err.get("message") if isinstance(err, dict) else err)[:90]
                if kind:
                    events.append(
                        dict(
                            rel=rel,
                            mod=rel.split("/")[1],
                            step=step,
                            kind=kind,
                            status=status,
                            reason=reason,
                            t=seen,
                        )
                    )
    return spans, events


def _is_429(ev: dict) -> bool:
    return ev["status"] == 429 or "429" in (ev["reason"] or "")


def _permutation_p(a: list[float], b: list[float], trials: int = 20000) -> float:
    """One-sided p for mean(a) > mean(b), by shuffling the labels."""
    obs = st.mean(a) - st.mean(b)
    pool = a + b
    rnd = random.Random(0)
    ge = 0
    for _ in range(trials):
        rnd.shuffle(pool)
        if st.mean(pool[: len(a)]) - st.mean(pool[len(a) :]) >= obs:
            ge += 1
    return ge / trials


def main(root: str) -> None:
    spans, events = collect(root)
    if not spans:
        sys.exit(f"no timestamped streams under {root}")
    rate_limits = [e for e in events if _is_429(e)]
    timed = [e for e in rate_limits if e["t"]]
    hit = {e["rel"] for e in rate_limits}
    run0 = min(s["t0"] for s in spans)
    end = max(s["t1"] for s in spans)

    print("=== scale ===")
    print(f"timestamped claude streams        : {len(spans)}")
    print(
        "claude 429 retry events           : "
        f"{sum(1 for e in rate_limits if e['kind'] == 'api_retry')}"
    )
    print(
        "codex streams that gave up at 429 : "
        f"{len({e['rel'] for e in rate_limits if e['kind'] != 'api_retry'})}"
    )
    print(f"502 server_error retries          : {sum(1 for e in events if e['status'] == 502)}")
    n_hit = len(hit & {s["rel"] for s in spans})
    print(
        "claude calls hitting >=1 429      : "
        f"{n_hit}/{len(spans)} = {n_hit / len(spans) * 100:.0f}%"
    )

    grid = [run0 + i * 30 for i in range(int((end - run0) / 30))]
    conc = [sum(1 for s in spans if s["t0"] <= t <= s["t1"]) for t in grid]
    print(
        "\nclaude sessions in flight         : "
        f"mean {st.mean(conc):.1f} median {st.median(conc):.0f} max {max(conc)}"
    )

    # --- did our own pacing cause them? Four independent tests. ---------------
    print("\n=== did our own concurrency cause the 429s? ===")

    def bucket_corr(width_min: float) -> tuple[float, int]:
        n = int(((end - run0) / 60) // width_min) + 1
        xs, ys = [], []
        for b in range(n):
            lo = run0 + b * width_min * 60
            hi = run0 + (b + 1) * width_min * 60
            xs.append(sum(1 for s in spans if s["t0"] < hi and s["t1"] > lo))
            ys.append(sum(1 for e in timed if lo <= e["t"] < hi))
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        return (num / (dx * dy) if dx and dy else 0.0), n

    for width in (2.0, 5.0, 10.0):
        r, n = bucket_corr(width)
        print(f"  [1] corr(in flight, 429s) at {width:>4.0f}-min buckets (n={n:3d}) : {r:+.3f}")

    def inflight_at(t: float, exclude: str) -> int:
        return sum(1 for s in spans if s["rel"] != exclude and s["t0"] <= t <= s["t1"])

    a = [inflight_at(s["t0"], s["rel"]) for s in spans if s["rel"] in hit]
    b = [inflight_at(s["t0"], s["rel"]) for s in spans if s["rel"] not in hit]
    print(
        f"  [2] peers in flight at own launch  -- 429-hit {st.mean(a):.2f} vs "
        f"clean {st.mean(b):.2f}  p={_permutation_p(a, b):.3f}"
    )

    def near(t: float, w: float, exclude: str) -> int:
        return sum(1 for s in spans if s["rel"] != exclude and t - w <= s["t0"] <= t + w)

    for w in (30, 60, 120, 300):
        a = [near(s["t0"], w, s["rel"]) for s in spans if s["rel"] in hit]
        b = [near(s["t0"], w, s["rel"]) for s in spans if s["rel"] not in hit]
        print(
            f"  [3] launches within +/-{w:3d}s        -- 429-hit {st.mean(a):.2f} vs "
            f"clean {st.mean(b):.2f}  p={_permutation_p(a, b):.3f}"
        )

    per = Counter(e["rel"] for e in rate_limits)

    def mean_conc(s: dict) -> float:
        others = [o for o in spans if o["rel"] != s["rel"]]
        step = max((s["t1"] - s["t0"]) / 40, 1)
        samples, t = [], s["t0"]
        while t <= s["t1"]:
            samples.append(1 + sum(1 for o in others if o["t0"] <= t <= o["t1"]))
            t += step
        return st.mean(samples) if samples else 1.0

    print("  [4] dose-response, 429s per call by the concurrency it actually ran at:")
    for lo, hi in ((1, 2), (2, 3), (3, 4), (4, 5), (5, 99)):
        grp = [s for s in spans if lo <= mean_conc(s) < hi]
        if not grp:
            continue
        label = f"{lo}-{hi}" if hi < 99 else f"{lo}+"
        print(
            f"        {label:>5s} peers : {len(grp):3d} calls, "
            f"{sum(per.get(s['rel'], 0) for s in grp) / len(grp):.2f} per call"
        )

    # --- and what the CLIs own retry did about it ----------------------------
    delays: list[int] = []
    depth: list[int] = []
    for s in spans:
        raw = open(os.path.join(root, s["rel"]), "rb").read()
        delays += [int(m.group(1)) for m in DELAY_RE.finditer(raw)]
        attempts = [int(m.group(1)) for m in ATTEMPT_RE.finditer(raw)]
        if attempts:
            depth.append(max(attempts))
    print("\n=== what the agent CLIs did about it themselves ===")
    if delays:
        print(
            f"retry events {len(delays)}   median backoff {st.median(delays):.0f} ms   "
            f"max {max(delays)} ms"
        )
        print(f"total time every CLI spent backing off, whole run: {sum(delays) / 1000:.0f} s")
    if depth:
        print(
            f"deepest attempt reached: {max(depth)}   "
            f"distribution: {dict(sorted(Counter(depth).items()))}"
        )

    # --- where the spikes landed ---------------------------------------------
    print("\n=== the biggest 429 spikes: one stuck call, or many modules at once? ===")
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for e in timed:
        by_bucket[int(((e["t"] - run0) / 60) // 5)].append(e)
    for b, evs in sorted(by_bucket.items(), key=lambda kv: -len(kv[1]))[:5]:
        infl = sum(
            1 for s in spans if s["t0"] < run0 + (b + 1) * 300 and s["t1"] > run0 + b * 300
        )
        print(
            f"  +{b * 5:3d}-{b * 5 + 5:3d} min: {len(evs):3d} rate limits across "
            f"{len({e['mod'] for e in evs})} modules / {len({e['rel'] for e in evs})} streams, "
            f"{infl} calls in flight"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1].replace("\\", "/"))
