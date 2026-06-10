"""Pull merged PRs from GitHub for a given window.

Full-fidelity scrape: list endpoint + per-PR detail + per-PR files. Caller
gets every field needed for any future filter (per-file paths/additions,
parent_sha, labels, body) without ever needing to re-scrape.

Why stdlib `urllib` instead of `requests`/`httpx`: this module is the first
network caller in the repo, and adding a transitive dep just to GET JSON
feels heavier than the few lines of `urlopen` it would replace.

Usage:

    from spotlights_engine.repo_bench import aggregation
    handle = aggregation.scrape(
        window_start="2025-12-03",
        window_end="2026-06-03",
        token=os.environ["GITHUB_TOKEN"],
    )
    print(handle.window_id, handle.prs_path, handle.manifest_path)

For the CLI entry point see `cli.py`.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from spotlights_engine.repo_bench.schemas import (
    AggregationManifest,
    FileChange,
    RawPR,
)
from spotlights_engine.repo_bench.storage import (
    append_jsonl,
    raw_dir,
    read_jsonl_lenient,
    window_id_for,
    write_json,
    write_jsonl,
)

log = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
DEFAULT_REPO = "vllm-project/vllm"
USER_AGENT = "spotlights-repo-bench/0.1"

# GitHub's search API caps hard at 1000 results regardless of paging. We use
# the issues endpoint to discover PR numbers and the pulls endpoint to fetch
# each one's detail; the issues endpoint paginates without that 1000 cap.
PER_PAGE = 100


# ── public API ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScrapeHandle:
    window_id: str
    out_dir: Path
    prs_path: Path
    manifest_path: Path
    total_prs: int


def scrape(
    *,
    window_start: date | datetime | str,
    window_end: date | datetime | str,
    token: str,
    repo: str = DEFAULT_REPO,
    out_root: Path | None = None,
    progress: bool = True,
) -> ScrapeHandle:
    """Scrape merged PRs in [window_start, window_end] into raw/<window_id>/.

    Resumable. On startup, looks for `prs.jsonl.partial` and `pr_numbers.json`
    in the target directory; if present, picks up where the prior run left
    off. Per-PR fetches are appended incrementally with fsync, so a crash
    leaves the partial file intact and ready for resume.

    Final output (`prs.jsonl` + `manifest.json`) is written atomically only
    once every PR has been fetched — so the presence of `prs.jsonl` means
    the scrape completed.
    """
    start = _to_date(window_start)
    end = _to_date(window_end)
    if end < start:
        raise ValueError(f"window_end {end} is before window_start {start}")

    wid = window_id_for(start, end)
    out_dir = raw_dir(wid, root=out_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    prs_path = out_dir / "prs.jsonl"
    manifest_path = out_dir / "manifest.json"
    partial_path = out_dir / "prs.jsonl.partial"
    pr_numbers_path = out_dir / "pr_numbers.json"

    if prs_path.exists() and manifest_path.exists():
        log.info("scrape: %s already complete — re-running will rebuild it", wid)

    client = _GitHubClient(token=token, repo=repo)
    query = (
        f"repo:{repo} is:pr is:merged "
        f"merged:{start.isoformat()}..{end.isoformat()}"
    )
    log.info("scrape: window=%s..%s repo=%s", start, end, repo)

    # Phase A — discover PR numbers (or load from prior run)
    pr_numbers = _load_or_discover_numbers(
        client, start, end, pr_numbers_path, progress=progress
    )

    # Phase B — incremental fetch with checkpointing
    already: dict[int, dict] = {row["pr_number"]: row for row in read_jsonl_lenient(partial_path)}
    if already:
        log.info("scrape: resuming — %d PRs already in partial", len(already))

    todo = [n for n in pr_numbers if n not in already]
    log.info("scrape: %d PRs to fetch (%d cached)", len(todo), len(already))

    fetched_this_run = 0
    for n in todo:
        try:
            pr = _fetch_pr_full(client, n)
        except _SkipPR as e:
            log.warning("scrape: skipping PR #%d: %s", n, e)
            continue
        append_jsonl(partial_path, pr)
        already[n] = json.loads(pr.model_dump_json())
        fetched_this_run += 1
        if progress and (fetched_this_run % 25 == 0):
            log.info(
                "scrape: fetched %d/%d this run (%d/%d total)",
                fetched_this_run, len(todo), len(already), len(pr_numbers),
            )

    # Phase C — finalize: sort partial, atomic-rename to final output
    rows = sorted(already.values(), key=lambda r: r["pr_number"])
    write_jsonl(prs_path, rows)
    manifest = AggregationManifest(
        window_start=datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc),
        window_end=datetime.combine(end, datetime.min.time(), tzinfo=timezone.utc),
        window_days=(end - start).days,
        github_query=query,
        fetched_at=datetime.now(timezone.utc),
        total_prs_returned=len(rows),
        repo=repo,
    )
    write_json(manifest_path, manifest)

    # Drop the resume artifacts now that the canonical prs.jsonl is on disk.
    # They are byte-equivalent (partial) or small caches (pr_numbers); keeping
    # them around just doubled disk usage. If a future scrape is interrupted,
    # the resume mechanism rebuilds them from scratch.
    for stale in (partial_path, pr_numbers_path):
        try:
            stale.unlink(missing_ok=True)
        except OSError as e:  # noqa: PERF203 — non-fatal cleanup
            log.warning("scrape: could not remove %s: %s", stale, e)

    log.info("scrape: complete — %d PRs in %s", len(rows), prs_path)
    return ScrapeHandle(
        window_id=wid,
        out_dir=out_dir,
        prs_path=prs_path,
        manifest_path=manifest_path,
        total_prs=len(rows),
    )


@dataclass(frozen=True)
class MergeHandle:
    window_id: str
    out_dir: Path
    prs_path: Path
    manifest_path: Path
    total_prs: int


def merge_windows(
    *,
    window_ids: list[str],
    target_window_id: str | None = None,
    out_root: Path | None = None,
) -> MergeHandle:
    """Merge multiple raw windows into one, deduplicating by pr_number.

    Later windows in the list take precedence on duplicates. The merged
    output lands in raw/<target_window_id>/ (computed from the combined
    date span if not given explicitly).
    """
    if len(window_ids) < 2:
        raise ValueError("merge_windows requires at least two window_ids")

    manifests: list[AggregationManifest] = []
    all_prs: dict[int, dict] = {}

    for wid in window_ids:
        wdir = raw_dir(wid, root=out_root)
        prs_path = wdir / "prs.jsonl"
        manifest_path = wdir / "manifest.json"
        if not prs_path.exists():
            raise FileNotFoundError(f"No prs.jsonl in window {wid} ({prs_path})")
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.json in window {wid} ({manifest_path})")

        manifest = AggregationManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        manifests.append(manifest)

        for row in read_jsonl_lenient(prs_path):
            all_prs[row["pr_number"]] = row

    repos = {m.repo for m in manifests}
    if len(repos) > 1:
        raise ValueError(f"Cannot merge windows from different repos: {repos}")

    starts = [m.window_start for m in manifests]
    ends = [m.window_end for m in manifests]
    merged_start = min(starts)
    merged_end = max(ends)

    if target_window_id is None:
        target_window_id = window_id_for(merged_start, merged_end)

    rows = sorted(all_prs.values(), key=lambda r: r["pr_number"])

    out_dir = raw_dir(target_window_id, root=out_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    prs_out = out_dir / "prs.jsonl"
    manifest_out = out_dir / "manifest.json"

    write_jsonl(prs_out, rows)

    merged_manifest = AggregationManifest(
        window_start=merged_start,
        window_end=merged_end,
        window_days=(merged_end.date() - merged_start.date()).days
        if hasattr(merged_end, "date")
        else (merged_end - merged_start).days,
        github_query=" | ".join(m.github_query for m in manifests),
        fetched_at=datetime.now(timezone.utc),
        total_prs_returned=len(rows),
        repo=next(iter(repos)),
    )
    write_json(manifest_out, merged_manifest)

    log.info(
        "merge_windows: %d unique PRs from %d windows → %s",
        len(rows), len(window_ids), prs_out,
    )
    return MergeHandle(
        window_id=target_window_id,
        out_dir=out_dir,
        prs_path=prs_out,
        manifest_path=manifest_out,
        total_prs=len(rows),
    )


def _load_or_discover_numbers(
    client: "_GitHubClient",
    start: date,
    end: date,
    cache_path: Path,
    *,
    progress: bool,
) -> list[int]:
    """Return PR numbers in window, reading from cache if present.

    The discovery phase costs ~50 search calls and is the part most likely
    to bump into the 30/min search cap. Caching it means resume after a
    crash skips straight to per-PR fetching.
    """
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, list):
                log.info("scrape: loaded %d cached pr-numbers from %s", len(cached), cache_path)
                return [int(n) for n in cached]
        except (json.JSONDecodeError, ValueError) as e:
            log.warning("scrape: pr-numbers cache unreadable (%s) — re-discovering", e)

    nums = list(_iter_merged_pr_numbers(client, start, end, progress=progress))
    log.info("scrape: discovered %d merged PRs in window", len(nums))
    write_json(cache_path, nums)
    return nums


# ── internals ────────────────────────────────────────────────────────────


def _to_date(d: date | datetime | str) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(d)


class _SkipPR(Exception):
    """Raised when a single PR can't be assembled — logged and skipped."""


class _GitHubClient:
    """Minimal stdlib REST client. Handles auth, paging, secondary rate-limit."""

    def __init__(self, token: str, repo: str) -> None:
        if not token:
            raise ValueError("GitHub token required (env GITHUB_TOKEN)")
        self.token = token
        self.repo = repo

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = self._build_url(path, params)
        return self._get_json(url)

    def paged(self, path: str, params: dict[str, Any] | None = None) -> Iterator[Any]:
        params = dict(params or {})
        params.setdefault("per_page", PER_PAGE)
        params["page"] = 1
        while True:
            page = self.get(path, params)
            if not page:
                return
            yield from page
            if len(page) < PER_PAGE:
                return
            params["page"] += 1

    def _build_url(self, path: str, params: dict[str, Any] | None) -> str:
        url = f"{GITHUB_API}{path}"
        if params:
            url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
        return url

    def _get_json(self, url: str, *, attempt: int = 1) -> Any:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self.token}",
                "User-Agent": USER_AGENT,
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                self._maybe_throttle(resp.headers)
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt <= 5:
                wait = self._retry_wait(e.headers, attempt)
                log.warning(
                    "github %s on %s — sleeping %.1fs (attempt %d/5)",
                    e.code, url, wait, attempt,
                )
                time.sleep(wait)
                return self._get_json(url, attempt=attempt + 1)
            if e.code in (502, 503, 504) and attempt <= 5:
                wait = min(2 ** attempt, 30)
                log.warning(
                    "github transient %s on %s — sleeping %.1fs (attempt %d/5)",
                    e.code, url, wait, attempt,
                )
                time.sleep(wait)
                return self._get_json(url, attempt=attempt + 1)
            raise
        except (
            http.client.IncompleteRead,
            urllib.error.URLError,
            socket.timeout,
            ConnectionError,
        ) as e:
            # Transient network glitches — retry with bounded backoff.
            # Distinct from HTTP 403/429 (handled above as rate-limit).
            if attempt <= 5:
                wait = min(2 ** attempt, 30)
                log.warning(
                    "github transient %s on %s — sleeping %.1fs (attempt %d/5)",
                    type(e).__name__, url, wait, attempt,
                )
                time.sleep(wait)
                return self._get_json(url, attempt=attempt + 1)
            raise

    @staticmethod
    def _maybe_throttle(headers: Any) -> None:
        """Preemptive sleep only when the *current* bucket is nearly empty.

        Resource matters: `core` has 5000/hr, `search` has 30/min. Both report
        through the same `X-RateLimit-*` headers, so we use the bucket-aware
        `X-RateLimit-Resource` to decide the threshold — otherwise every
        single search call (always under 30 remaining) would trigger a 59s
        sleep. The real protection against exhaustion is the 403/429 retry
        path, which honors `Retry-After`; this is just a soft pre-empt.
        """
        remaining = headers.get("X-RateLimit-Remaining")
        if remaining is None:
            return
        try:
            r = int(remaining)
        except ValueError:
            return
        resource = (headers.get("X-RateLimit-Resource") or "").lower()
        # search bucket is small by design (30/min); only sleep if truly empty
        threshold = 1 if resource == "search" else 50
        if r > threshold:
            return
        reset = headers.get("X-RateLimit-Reset")
        try:
            reset_at = int(reset) if reset else 0
        except ValueError:
            reset_at = 0
        wait = max(1, reset_at - int(time.time())) if reset_at else 5
        log.warning(
            "github %s rate-limit near empty (%s left) — sleeping %ds",
            resource or "?", r, wait,
        )
        time.sleep(min(wait, 120))

    @staticmethod
    def _retry_wait(headers: Any, attempt: int) -> float:
        retry_after = headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        reset = headers.get("X-RateLimit-Reset")
        if reset:
            try:
                return max(1.0, int(reset) - time.time())
            except ValueError:
                pass
        return min(60.0, 2 ** attempt)


def _iter_merged_pr_numbers(
    client: _GitHubClient,
    start: date,
    end: date,
    *,
    progress: bool = True,
) -> Iterator[int]:
    """Discover PR numbers via the search API.

    Search caps at 1000 results per query. For 6 months of vLLM that may be
    tight; we shard by month to stay safely under the cap and merge the
    results.
    """
    seen: set[int] = set()
    for shard_start, shard_end in _month_shards(start, end):
        q = (
            f"repo:{client.repo} is:pr is:merged "
            f"merged:{shard_start.isoformat()}..{shard_end.isoformat()}"
        )
        page_num = 0
        for page in _search_paged(client, q):
            page_num += 1
            for item in page:
                n = item.get("number")
                if isinstance(n, int) and n not in seen:
                    seen.add(n)
                    yield n
            if progress:
                log.info(
                    "search: %s..%s page=%d cumulative=%d",
                    shard_start, shard_end, page_num, len(seen),
                )


def _month_shards(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield (start, end) date ranges, one per calendar month chunk."""
    cur = start
    while cur <= end:
        # last day of cur's month, capped at `end`
        if cur.month == 12:
            next_first = date(cur.year + 1, 1, 1)
        else:
            next_first = date(cur.year, cur.month + 1, 1)
        chunk_end = min(end, date.fromordinal(next_first.toordinal() - 1))
        yield cur, chunk_end
        cur = next_first


def _search_paged(client: _GitHubClient, q: str) -> Iterator[list[dict[str, Any]]]:
    page = 1
    while True:
        result = client.get(
            "/search/issues",
            {"q": q, "per_page": PER_PAGE, "page": page, "sort": "created", "order": "asc"},
        )
        items = result.get("items", []) if isinstance(result, dict) else []
        if not items:
            return
        yield items
        if len(items) < PER_PAGE:
            return
        page += 1
        if page > 10:  # search caps at 1000 = 10 pages of 100; bail out
            return


def _fetch_pr_full(client: _GitHubClient, pr_number: int) -> RawPR:
    """Pull PR detail + files and assemble a RawPR. Skips on missing core fields."""
    detail = client.get(f"/repos/{client.repo}/pulls/{pr_number}")
    if not isinstance(detail, dict):
        raise _SkipPR("detail not a dict")
    merged_at = detail.get("merged_at")
    merge_sha = detail.get("merge_commit_sha")
    if not merged_at or not merge_sha:
        raise _SkipPR("not actually merged")

    base = detail.get("base") or {}
    parent_sha = base.get("sha") or ""
    if not parent_sha:
        # Rare: closed-and-reopened PRs sometimes lack base.sha. Fall back to
        # the merge-commit's first parent via the commits endpoint.
        commit = client.get(f"/repos/{client.repo}/commits/{merge_sha}")
        parents = commit.get("parents", []) if isinstance(commit, dict) else []
        parent_sha = parents[0]["sha"] if parents else ""

    user = detail.get("user") or {}
    labels = [
        lab.get("name", "") for lab in detail.get("labels", []) if isinstance(lab, dict)
    ]

    files = list(_iter_pr_files(client, pr_number))

    return RawPR(
        pr_number=pr_number,
        title=detail.get("title", ""),
        body=detail.get("body") or "",
        merged_at=_parse_iso(merged_at),
        merge_sha=merge_sha,
        parent_sha=parent_sha,
        author=user.get("login", ""),
        labels=labels,
        files_changed=files,
        additions_total=int(detail.get("additions") or 0),
        deletions_total=int(detail.get("deletions") or 0),
        commits_count=int(detail.get("commits") or 0),
        url=detail.get("html_url", f"https://github.com/{client.repo}/pull/{pr_number}"),
    )


def _iter_pr_files(client: _GitHubClient, pr_number: int) -> Iterator[FileChange]:
    for f in client.paged(f"/repos/{client.repo}/pulls/{pr_number}/files"):
        if not isinstance(f, dict):
            continue
        yield FileChange(
            path=f.get("filename", ""),
            additions=int(f.get("additions") or 0),
            deletions=int(f.get("deletions") or 0),
            status=f.get("status", ""),
        )


def _parse_iso(s: str) -> datetime:
    # GitHub returns Z-suffixed UTC timestamps; fromisoformat doesn't accept
    # the Z literal until 3.11+, but we require 3.11 anyway.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)
