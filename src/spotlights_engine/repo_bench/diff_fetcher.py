"""Fetch per-PR unified diffs for a view.

Diffs land at `raw/<window_id>/diffs/<pr_number>.diff` (raw bytes from
GitHub) with a sibling manifest recording each diff's `sha256`,
`fetched_at`, and byte size.

Run on demand during the bench-prep flow rather than at scrape time,
because:

  - Diff bodies are MB-scale; we'd download tens of GB during a 5K-PR
    scrape if we pulled them all.
  - Downstream stages work on a much smaller view, so we only fetch
    what we'll use.

We reuse `aggregation._GitHubClient` for auth + retry + secondary
rate-limit handling.

Default behavior is idempotent: skip any PR whose `<n>.diff` already
exists. `refresh=True` re-fetches every diff in the view; new bytes
overwrite atomically and the manifest's `sha256` updates so any
downstream cache keyed on `diff_sha256` automatically invalidates.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from spotlights_engine.repo_bench.aggregation import (
    DEFAULT_REPO,
    USER_AGENT,
    _GitHubClient,
)
from spotlights_engine.repo_bench.storage import (
    raw_dir,
    write_json,
)

log = logging.getLogger(__name__)

# GitHub's pulls endpoint returns the unified diff body when called with
# this Accept header. Note: this is a different content-type than the
# JSON `/pulls/{n}` we already use in aggregation; we hit the same URL
# with a different Accept and get raw text instead of JSON.
_DIFF_ACCEPT = "application/vnd.github.diff"


# ── Public API ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DiffFetchHandle:
    """What `fetch_diffs()` returns."""

    window_id: str
    diffs_dir: Path
    manifest_path: Path
    n_total: int          # PRs in the view
    n_fetched: int        # actually fetched this run (cache miss + refresh)
    n_skipped: int        # already present, not refreshed


def fetch_diffs(
    *,
    window_id: str,
    view_path: Path,
    token: str,
    repo: str = DEFAULT_REPO,
    refresh: bool = False,
    data_root_override: Path | None = None,
) -> DiffFetchHandle:
    """Fetch each PR's unified diff into `raw/<window_id>/diffs/<n>.diff`.

    Inputs:
        window_id: the raw scrape's window id (must already exist).
        view_path: path to the run's view file (`<run_dir>/view/prs.jsonl`).
        token:     GitHub PAT (same scope as `aggregation.scrape`).
        refresh:   if True, re-fetch every diff in the view; otherwise
                   skip PRs whose diff file already exists.
        data_root_override: data-root override for tests.

    Idempotent under refresh=False: re-running with the same view is
    a no-op.
    """
    pr_numbers = _read_view_pr_numbers(view_path)
    diffs_dir = raw_dir(window_id, root=data_root_override) / "diffs"
    diffs_dir.mkdir(parents=True, exist_ok=True)

    client = _GitHubClient(token=token, repo=repo)
    manifest = _read_manifest(diffs_dir)

    fetched = 0
    skipped = 0
    for pr_number in pr_numbers:
        diff_path = diffs_dir / f"{pr_number}.diff"
        if diff_path.exists() and not refresh:
            skipped += 1
            continue
        try:
            body = _fetch_one_diff(client, pr_number)
        except _DiffFetchError as e:
            log.warning("fetch_diffs: PR #%d: %s", pr_number, e)
            continue
        sha = hashlib.sha256(body).hexdigest()
        _atomic_write_bytes(diff_path, body)
        manifest[str(pr_number)] = {
            "sha256": sha,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "bytes": len(body),
        }
        # Persist manifest after each PR so a Ctrl+C mid-run leaves a
        # consistent state on disk: every diff that exists has its
        # entry in the manifest.
        _write_manifest(diffs_dir, manifest)
        fetched += 1
        if fetched % 10 == 0:
            log.info("fetch_diffs: %d/%d", fetched + skipped, len(pr_numbers))

    _write_manifest(diffs_dir, manifest)

    return DiffFetchHandle(
        window_id=window_id,
        diffs_dir=diffs_dir,
        manifest_path=diffs_dir / "manifest.json",
        n_total=len(pr_numbers),
        n_fetched=fetched,
        n_skipped=skipped,
    )


# ── Internals ─────────────────────────────────────────────────────────


class _DiffFetchError(Exception):
    """One PR's diff couldn't be fetched. Logged + skipped, not fatal."""


def _read_view_pr_numbers(view_path: Path) -> list[int]:
    """Read pr_numbers from a view's prs.jsonl."""
    if not view_path.exists():
        raise FileNotFoundError(
            f"no view at {view_path}; run filter step first"
        )
    pr_numbers: list[int] = []
    with view_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pr_numbers.append(int(row["pr_number"]))
    return pr_numbers


def _fetch_one_diff(client: _GitHubClient, pr_number: int) -> bytes:
    """GET /repos/<repo>/pulls/<n> with Accept: application/vnd.github.diff.

    We don't use `client.get` because the payload is raw bytes, not JSON.
    We do reuse the auth + URL build + (in spirit) the rate-limit retry
    behavior, but a much simpler version since the diff endpoint shares
    the `core` rate-limit bucket, which `client._maybe_throttle`
    already protects against on its other calls.
    """
    url = f"https://api.github.com/repos/{client.repo}/pulls/{pr_number}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": _DIFF_ACCEPT,
            "Authorization": f"Bearer {client.token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        raise _DiffFetchError(f"HTTP {e.code} from {url}") from e


def _read_manifest(diffs_dir: Path) -> dict[str, dict]:
    p = diffs_dir / "manifest.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("fetch_diffs: manifest unreadable, starting fresh")
        return {}
    return data if isinstance(data, dict) else {}


def _write_manifest(diffs_dir: Path, manifest: dict[str, dict]) -> None:
    write_json(diffs_dir / "manifest.json", manifest)


def _atomic_write_bytes(path: Path, body: bytes) -> None:
    """Write `body` to `path` atomically: temp file in same dir + rename.

    Same idiom as `storage.atomic_write_text` but for raw bytes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(body)
        os.replace(tmp, path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


__all__ = ["DiffFetchHandle", "fetch_diffs"]
