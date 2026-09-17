#!/usr/bin/env python3
"""Find where the gateway starts refusing us, and what it says when it does.

Why this exists
---------------
The IOCR run took 149 rate limits and the archived streams could not explain
them: measured from the stream timestamps that run averaged 2.9 concurrent agent
sessions and peaked at 6, and four independent tests found no link between our
own concurrency and the refusals. But it never went above 6, so it could never
show where the wall is -- only that it is not inside 1..6.

Two things this probe can learn that the archive cannot, and that our virtual
key is permitted to ask (the key is restricted to `llm_api_routes`, so
`/key/info` returns 403 and no administrative answer is available to us):

1. **What a 429 actually says.** The agent CLIs wait ~0.6 s on their own jittered
   guess and never surface a `retry-after`. Calling the API directly exposes
   every response header, so if the gateway states a wait figure we can obey it
   instead of inventing a backoff constant.
2. **Whether our own concurrency is what breaks it.** This ramps well past 6.

And one inference that substitutes for the traffic log we cannot get: a limit
attached to our own key is a *constant*, while a ceiling shared with the rest of
the team moves as other people start and stop working. So run this at several
times of day with `--label` and compare the breaking points. A stable knee means
the limit is ours and pacing fixes it; a knee that wanders means it is shared and
only waiting helps.

A call costs about $0.00005 (measured from `x-litellm-response-cost-original` on
a 1-token reply), so a full ramp is a couple of cents. Reads
ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN from the environment and prints
neither.

Usage (WSL):
    source ~/.config/coral-env.sh
    python3 scripts/ratelimit_probe.py --label midday --max-concurrency 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# Headers worth keeping. Everything else is dropped rather than filtered, so a
# future header carrying anything sensitive cannot leak into a transcript.
HEADER_PREFIXES = ("retry-after", "x-ratelimit", "ratelimit", "anthropic-ratelimit")
COST_HEADER = "x-litellm-response-cost-original"
NO_BODY = "(no body)"


def one_call(url: str, token: str, model: str, idx: int, t_zero: float) -> dict:
    """One minimal request. Never raises -- a refusal is the measurement."""
    payload = json.dumps(
        {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "authorization": f"Bearer {token}",
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        },
    )
    started = time.time()
    row: dict = {"idx": idx, "t": round(started - t_zero, 3)}
    headers = None
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            row["status"] = resp.status
            headers = resp.headers
    except urllib.error.HTTPError as exc:
        row["status"] = exc.code
        headers = exc.headers
        row["detail"] = exc.read()[:400].decode("utf-8", "replace")
    except Exception as exc:  # a network-level failure is data too
        row["status"] = 0
        row["detail"] = type(exc).__name__
    row["latency_s"] = round(time.time() - started, 3)
    if headers is not None:
        for name, value in headers.items():
            low = name.lower()
            if low.startswith(HEADER_PREFIXES):
                row.setdefault("limit_headers", {})[low] = value
            elif low == COST_HEADER:
                row["cost"] = value
    return row


def wave(url: str, token: str, model: str, n: int, t_zero: float) -> list[dict]:
    """Fire n requests as simultaneously as threads allow."""
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda i: one_call(url, token, model, i, t_zero), range(n)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run", help="tag this probe, e.g. midday / 3am")
    ap.add_argument("--max-concurrency", type=int, default=8, help="highest wave size")
    ap.add_argument(
        "--stop-after-429", type=int, default=20, help="stop once this many refusals are seen"
    )
    ap.add_argument("--max-calls", type=int, default=250, help="hard spend guard")
    ap.add_argument("--settle-s", type=float, default=5.0, help="pause between waves")
    ap.add_argument("--out-dir", default="artifacts/ratelimit_probe")
    args = ap.parse_args()

    base = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
    if not base or not token:
        sys.exit("ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN not set -- source coral-env.sh first")
    root = base.removesuffix("/v1")
    url = base if base.endswith("/v1/messages") else f"{root}/v1/messages"

    sizes = [n for n in (1, 2, 4, 8, 16, 32, 64) if n <= args.max_concurrency]
    os.makedirs(args.out_dir, exist_ok=True)
    # An unset shell variable arrives as an empty --label and would silently write
    # ".jsonl", overwriting the previous run -- fatal for the time-of-day
    # comparison this probe exists for, so always stamp the clock into the name.
    label = "".join(c for c in args.label if c.isalnum() or c in "-_") or "run"
    stamp = time.strftime("%Y%m%d-%H%M")
    out_path = os.path.join(args.out_dir, f"{label}-{stamp}.jsonl")

    print(f"label {label}   model {model}   waves {sizes}   budget {args.max_calls} calls")
    print("wave |  ok  429  5xx other | median latency | retry-after seen")
    print("-----+---------------------+----------------+-----------------")

    t_zero = time.time()
    rows: list[dict] = []
    refusals = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for n in sizes:
            if len(rows) + n > args.max_calls:
                print(f"stopping before wave {n}: would exceed --max-calls")
                break
            batch = wave(url, token, model, n, t_zero)
            for r in batch:
                r["wave"] = n
                r["label"] = label
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            rows += batch

            codes = Counter(r["status"] for r in batch)
            ok = codes.get(200, 0)
            r429 = codes.get(429, 0)
            r5xx = sum(v for k, v in codes.items() if isinstance(k, int) and 500 <= k < 600)
            other = n - ok - r429 - r5xx
            lat = st.median([r["latency_s"] for r in batch])
            says = any("retry-after" in (r.get("limit_headers") or {}) for r in batch)
            print(
                f"{n:4d} | {ok:3d} {r429:4d} {r5xx:4d} {other:5d} | "
                f"{lat:13.2f}s | {'YES' if says else 'no'}"
            )
            refusals += r429
            if refusals >= args.stop_after_429:
                print(f"stopping: {refusals} refusals captured, enough to read")
                break
            time.sleep(args.settle_s)

    # --- what the gateway actually told us -----------------------------------
    seen: dict[str, str] = {}
    for r in rows:
        for k, v in (r.get("limit_headers") or {}).items():
            seen.setdefault(k, v)
    print("\nrate-limit headers observed:")
    if seen:
        for k, v in sorted(seen.items()):
            print(f"  {k}: {v}")
    else:
        print("  none -- the gateway never states a limit or a wait, so any backoff")
        print("  we build has to be sized by measurement rather than by reading it")

    bad = [r for r in rows if r["status"] == 429]
    if bad:
        print(f"\nfirst refusal, verbatim ({len(bad)} total):")
        print(f"  {(bad[0].get('detail') or NO_BODY)[:300]}")
        by_wave = Counter(r["wave"] for r in bad)
        print(
            "  refusals by wave size: "
            + ", ".join(f"{k}->{v}" for k, v in sorted(by_wave.items()))
        )
    else:
        top = sizes[-1] if sizes else 0
        print(f"\nno refusals in {len(rows)} calls up to concurrency {top}.")
        print("  Our own key was not the constraint at this level, at this time of day.")

    costs = [float(r["cost"]) for r in rows if r.get("cost")]
    if costs:
        print(f"\nspend: {len(rows)} calls, ${sum(costs):.4f}")
    print(f"raw rows: {out_path}")


if __name__ == "__main__":
    main()
