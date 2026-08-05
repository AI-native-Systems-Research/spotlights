#!/usr/bin/env python3
"""Build a spotlights-style run_manifest.json from a Claude Code one-shot baseline.

Reads the `_stdout.json` sidecar produced by `claude -p --output-format json` plus
the run metadata (repo path, prompt path, output dir), and emits a
`run_manifest.json` that matches the Ophir schema used by spotlights-engine runs,
minus the `spotlights` block (there is no spotlights involvement).

Usage:
  cc_baseline_manifest.py \
      --stdout-json PATH \
      --run-dir     PATH \
      --repo-path   PATH \
      --repo-url    URL \
      --objective   TEXT \
      --prompt-path PATH \
      --wall-clock  SECONDS \
      --run-id      TEXT
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse engine internals so the schema stays identical by construction.
from spotlights_engine.costing.rates import compute_cost, load_rates
from spotlights_engine.costing.records import UsageRecord
from spotlights_engine.costing.manifest import (
    aggregate_models_used,
    build_run_manifest,
)


def _git_head(path: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except FileNotFoundError:
        pass
    return ""


def _count_candidates(candidates_dir: Path) -> int:
    if not candidates_dir.exists():
        return 0
    return sum(
        1
        for entry in candidates_dir.iterdir()
        if entry.is_file()
        and entry.suffix == ".md"
        and entry.name != "_ranking.md"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout-json", required=True, type=Path)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--repo-url", required=True)
    ap.add_argument("--objective", required=True)
    ap.add_argument("--prompt-path", required=True, type=Path)
    ap.add_argument("--wall-clock", required=True, type=float)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()

    _payload = json.loads(args.stdout_json.read_text())
    # Verbose stream-json produces an array of turn events; the final element
    # is the summary object (same shape as non-verbose output).
    if isinstance(_payload, list):
        summary = next((ev for ev in reversed(_payload) if isinstance(ev, dict) and ev.get("subtype") == "success"), None)
        if summary is None:
            summary = next((ev for ev in reversed(_payload) if isinstance(ev, dict) and "modelUsage" in ev), None)
        if summary is None:
            raise SystemExit("could not find summary object in verbose stream-json _stdout.json")
        sidecar = summary
    else:
        sidecar = _payload

    # Build one UsageRecord per model — CC reports per-model in `modelUsage`.
    records: list[UsageRecord] = []
    model_usage = sidecar.get("modelUsage") or {}
    if not model_usage:
        # Fallback: single record from top-level usage if modelUsage empty.
        top = sidecar.get("usage") or {}
        # No model id available; leave `model` empty so cost falls to CLI-family key.
        model_usage = {
            "": {
                "inputTokens": top.get("input_tokens", 0),
                "outputTokens": top.get("output_tokens", 0),
                "cacheReadInputTokens": top.get("cache_read_input_tokens", 0),
                "cacheCreationInputTokens": top.get("cache_creation_input_tokens", 0),
            }
        }

    duration_api_s = (sidecar.get("duration_api_ms") or 0) / 1000.0
    per_model_api_s = duration_api_s / max(1, len(model_usage))

    for model_id, u in model_usage.items():
        records.append(
            UsageRecord(
                step="agent_proposals",
                module_qualified_name="_one_shot",
                session_index=1,
                invocation_index=0,
                invocation_id=sidecar.get("session_id", ""),
                provider="anthropic",
                cli="claude",
                model=model_id or "",
                role="one_shot_baseline",
                input=int(u.get("inputTokens", 0)),
                output=int(u.get("outputTokens", 0)),
                cache_read=int(u.get("cacheReadInputTokens", 0)),
                cache_create=int(u.get("cacheCreationInputTokens", 0)),
                api_time_s=per_model_api_s,
                cli_reported_cost_usd=float(u.get("costUSD", 0.0)) or None,
            )
        )

    rates = load_rates()
    cost = compute_cost(records, rates)

    provenance = {
        "repo_url": args.repo_url,
        "target_commit_sha": _git_head(args.repo_path),
        # No spotlights involvement — leave empty so `build_run_manifest`
        # records "Spotlights commit unavailable" in `notes`. We strip the
        # `spotlights` block post-build.
        "spotlights_commit_sha": "",
    }

    # Config fingerprint: a small dict identifying the CC-baseline recipe,
    # analogous to spotlights' config_fingerprint. We keep it here temporarily
    # so build_run_manifest can be reused; then move it to a top-level `agent`
    # block after construction.
    config_fingerprint = {
        "agent": "claude-code",
        "prompt_path": str(args.prompt_path),
        "prompt_sha256_first16": _sha256_first16(args.prompt_path),
    }

    candidates_dir = args.run_dir / "candidates"
    num_candidates = _count_candidates(candidates_dir)

    manifest = build_run_manifest(
        run_id=args.run_id,
        date=datetime.now(timezone.utc).isoformat(),
        objective=args.objective,
        provenance=provenance,
        config_fingerprint=config_fingerprint,
        records=records,
        cost=cost,
        wall_clock_s=args.wall_clock,
        accumulated_duration_s=args.wall_clock,
        candidates_path=str(candidates_dir),
        num_candidates=num_candidates,
        module_status={"SUCCEEDED": 1 if not sidecar.get("is_error") else 0,
                       "FAILED":   1 if sidecar.get("is_error") else 0,
                       "DEGRADED": 0,
                       "SKIPPED":  0},
        notes=["one-shot Claude-Code baseline (no spotlights pipeline)"],
    )

    # Serialize, then remove the `spotlights` block and stamp `pipeline` +
    # `agent` at top level so the file reflects the CC-baseline shape.
    payload = manifest.model_dump(mode="json")
    payload.pop("spotlights", None)
    payload["pipeline"] = "one-shot-agent-baseline"
    payload["agent"] = {
        "name": "claude-code",
        "prompt_path": str(args.prompt_path),
        "prompt_sha256_first16": config_fingerprint["prompt_sha256_first16"],
        "cli_session_id": sidecar.get("session_id", ""),
        "num_turns": sidecar.get("num_turns", 0),
        "terminal_reason": sidecar.get("terminal_reason", ""),
        "stop_reason": sidecar.get("stop_reason", ""),
        "is_error": bool(sidecar.get("is_error", False)),
    }
    # Also stamp the CC-reported cost for cross-check (contracted-rate math is
    # what's authoritative in `cost.amount_usd`).
    payload["cost"]["amount_usd_cc_reported"] = float(sidecar.get("total_cost_usd") or 0.0)

    out_path = args.run_dir / "run_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out_path}", file=sys.stderr)
    return 0


def _sha256_first16(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
