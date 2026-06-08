"""Pick the target-repo snapshot SHA from the filtered view + raw scrape.

Deterministic given: the raw scrape and the view. No LLM calls.
Output is a `SnapshotPin` written to `<run_dir>/snapshot.json`.

Algorithm:

1. Load the view's PR numbers; join against raw to get `merged_at`.
2. Find the earliest `merged_at` across the view PRs. Every view PR
   is "future" relative to whatever we pin.
3. Compute cutoff = earliest_in_view_merged_at - buffer_hours.
4. From raw, find the latest PR whose `merged_at` is before the
   cutoff. Its `merge_sha` is the pin.

Edge cases:
- Empty view → ValueError (filter produced nothing).
- No raw PR before cutoff → ValueError (window too short or buffer
  too large; widen scrape or reduce buffer).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import ValidationError

from spotlights_engine.repo_bench.schemas import (
    SNAPSHOT_BUFFER_HOURS,
    RawPR,
    SnapshotPin,
)
from spotlights_engine.repo_bench.storage import (
    data_root,
    raw_dir,
    read_jsonl_lenient,
    write_json,
)

log = logging.getLogger(__name__)


def pick_snapshot(
    *,
    window_id: str,
    view_id: str,
    view_path: Path,
    run_dir: Path,
    data_root_override: Path | None = None,
    buffer_hours: int | None = None,
) -> SnapshotPin:
    """Compute and persist the snapshot pin for a filtered view.

    Writes `<run_dir>/snapshot.json`.

    `view_path` is `<run_dir>/view/prs.jsonl`. `data_root_override` is
    where to find the cached raw scrape (defaults to `data_root()`).
    """
    buf = buffer_hours if buffer_hours is not None else SNAPSHOT_BUFFER_HOURS

    if not view_path.exists():
        raise FileNotFoundError(
            f"no view at {view_path}; run filter step first"
        )

    raw_path = raw_dir(window_id, root=data_root_override) / "prs.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"no raw scrape at {raw_path}; cannot pick snapshot without it"
        )

    view_pr_numbers = _load_view_pr_numbers(view_path)
    if not view_pr_numbers:
        raise ValueError(
            f"empty view at {view_path}; nothing to anchor against"
        )

    raw_by_pr = {pr.pr_number: pr for pr in _load_raw(raw_path)}
    in_view: list[dict] = []
    for pr_n in view_pr_numbers:
        pr = raw_by_pr.get(pr_n)
        if pr is None:
            log.warning("snapshot: view PR #%d missing from raw; skipping", pr_n)
            continue
        in_view.append({"pr_number": pr_n, "merged_at": pr.merged_at})

    if not in_view:
        raise ValueError(
            f"no view PRs found in raw scrape; raw is stale relative to view"
        )

    earliest = min(in_view, key=lambda r: r["merged_at"])
    earliest_merged_at = earliest["merged_at"]
    log.info(
        "snapshot: %d view PRs, earliest is PR #%d at %s",
        len(in_view), earliest["pr_number"], earliest_merged_at.isoformat(),
    )

    cutoff = earliest_merged_at - timedelta(hours=buf)
    eligible = [pr for pr in raw_by_pr.values() if pr.merged_at < cutoff]
    if not eligible:
        raise ValueError(
            f"no PR in raw scrape merged before cutoff "
            f"{cutoff.isoformat()} (earliest in view: PR #{earliest['pr_number']} "
            f"at {earliest_merged_at.isoformat()}). Widen the scrape window "
            f"or reduce --snapshot-buffer-hours."
        )
    snapshot_pr = max(eligible, key=lambda pr: pr.merged_at)

    rationale = (
        f"Pinned to PR #{snapshot_pr.pr_number} (merged "
        f"{snapshot_pr.merged_at.isoformat()}). The earliest PR in the "
        f"filtered view is #{earliest['pr_number']} at "
        f"{earliest_merged_at.isoformat()}, so every one of the "
        f"{len(in_view)} view PRs is at least {buf}h future relative to "
        f"this SHA. Discovery must be blind to those changes."
    )

    pin = SnapshotPin(
        window_id=window_id,
        view_id=view_id,
        snapshot_sha=snapshot_pr.merge_sha,
        snapshot_pr_number=snapshot_pr.pr_number,
        snapshot_merged_at=snapshot_pr.merged_at,
        earliest_in_view_pr_number=earliest["pr_number"],
        earliest_in_view_merged_at=earliest_merged_at,
        buffer_hours=buf,
        n_view=len(in_view),
        rationale=rationale,
        pinned_at=datetime.now(timezone.utc),
    )

    out_path = run_dir / "snapshot.json"
    write_json(out_path, pin)
    log.info("snapshot: pinned SHA %s → %s", pin.snapshot_sha, out_path)

    return pin


# ── Internals ─────────────────────────────────────────────────────────


def _load_view_pr_numbers(view_path: Path) -> set[int]:
    """Read view's prs.jsonl, return pr_numbers."""
    out: set[int] = set()
    for line in view_path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            out.add(int(json.loads(line)["pr_number"]))
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
    return out


def _load_raw(raw_path: Path) -> list[RawPR]:
    out: list[RawPR] = []
    for d in read_jsonl_lenient(raw_path):
        try:
            out.append(RawPR.model_validate(d))
        except ValidationError:
            continue
    return out


__all__ = ["pick_snapshot"]
