"""Aggregation — fixture-backed unit tests, no network.

The GitHub client is replaced by a fake that returns canned responses keyed
by URL path. Tests cover: month sharding, PR-detail assembly, and the full
scrape orchestration end-to-end.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from spotlights_engine.repo_bench import aggregation, storage
from spotlights_engine.repo_bench.schemas import RawPR


# ── helpers ──────────────────────────────────────────────────────────────


def _detail(n: int, **overrides: Any) -> dict:
    base = {
        "number": n,
        "title": f"PR {n}",
        "body": f"body of {n}",
        "merged_at": "2026-01-15T12:00:00Z",
        "merge_commit_sha": f"merge{n:08d}",
        "base": {"sha": f"parent{n:08d}"},
        "user": {"login": "alice"},
        "labels": [{"name": "perf"}, {"name": "v1"}],
        "additions": 10,
        "deletions": 3,
        "commits": 2,
        "html_url": f"https://github.com/vllm-project/vllm/pull/{n}",
    }
    base.update(overrides)
    return base


def _files(n: int) -> list[dict]:
    return [
        {"filename": f"vllm/foo_{n}.py", "additions": 5, "deletions": 1, "status": "modified"},
        {"filename": f"tests/test_foo_{n}.py", "additions": 5, "deletions": 2, "status": "modified"},
    ]


class _FakeClient:
    """Stand-in for `_GitHubClient` — same surface, in-memory responses."""

    def __init__(self, repo: str, *, search_items: list[dict], details: dict, files: dict):
        self.repo = repo
        self._search_items = search_items
        self._details = details
        self._files = files
        self.calls: list[str] = []

    def get(self, path: str, params: dict | None = None) -> Any:
        self.calls.append(path)
        if path == "/search/issues":
            page = (params or {}).get("page", 1)
            per_page = (params or {}).get("per_page", 100)
            start = (page - 1) * per_page
            end = start + per_page
            return {"items": self._search_items[start:end]}
        if path.startswith(f"/repos/{self.repo}/pulls/") and path.endswith("/files"):
            n = int(path.split("/")[-2])
            return self._files.get(n, [])
        if path.startswith(f"/repos/{self.repo}/pulls/"):
            n = int(path.rstrip("/").split("/")[-1])
            return self._details[n]
        if path.startswith(f"/repos/{self.repo}/commits/"):
            return {"parents": [{"sha": "fallback-parent"}]}
        raise AssertionError(f"unexpected path {path}")

    def paged(self, path: str, params: dict | None = None):
        # Files endpoint: small per-PR list, single page is fine.
        result = self.get(path, params)
        if isinstance(result, list):
            yield from result


# ── _month_shards ────────────────────────────────────────────────────────


def test_month_shards_within_one_month():
    shards = list(aggregation._month_shards(date(2026, 1, 5), date(2026, 1, 20)))
    assert shards == [(date(2026, 1, 5), date(2026, 1, 20))]


def test_month_shards_across_months():
    shards = list(aggregation._month_shards(date(2025, 12, 15), date(2026, 2, 10)))
    assert shards == [
        (date(2025, 12, 15), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 31)),
        (date(2026, 2, 1), date(2026, 2, 10)),
    ]


def test_month_shards_year_boundary():
    shards = list(aggregation._month_shards(date(2025, 12, 30), date(2026, 1, 2)))
    assert shards == [
        (date(2025, 12, 30), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 1, 2)),
    ]


# ── _fetch_pr_full ───────────────────────────────────────────────────────


def test_fetch_pr_full_assembles_raw_pr():
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={42: _detail(42)},
        files={42: _files(42)},
    )
    pr = aggregation._fetch_pr_full(fake, 42)
    assert isinstance(pr, RawPR)
    assert pr.pr_number == 42
    assert pr.merge_sha == "merge00000042"
    assert pr.parent_sha == "parent00000042"
    assert [f.path for f in pr.files_changed] == [
        "vllm/foo_42.py", "tests/test_foo_42.py",
    ]
    assert pr.labels == ["perf", "v1"]
    assert pr.merged_at == datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def test_fetch_pr_full_skips_unmerged():
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={1: _detail(1, merged_at=None)},
        files={1: []},
    )
    with pytest.raises(aggregation._SkipPR):
        aggregation._fetch_pr_full(fake, 1)


def test_fetch_pr_full_falls_back_when_base_sha_missing():
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={7: _detail(7, base={})},
        files={7: _files(7)},
    )
    pr = aggregation._fetch_pr_full(fake, 7)
    assert pr.parent_sha == "fallback-parent"


# ── scrape end-to-end ────────────────────────────────────────────────────


def test_scrape_writes_jsonl_and_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))

    search_items = [{"number": 100}, {"number": 101}, {"number": 102}]
    details = {n: _detail(n) for n in (100, 101, 102)}
    files = {n: _files(n) for n in (100, 101, 102)}
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=search_items,
        details=details,
        files=files,
    )

    monkeypatch.setattr(
        aggregation, "_GitHubClient", lambda **kw: fake,
    )

    handle = aggregation.scrape(
        window_start="2026-01-01",
        window_end="2026-01-31",
        token="fake-token",
    )

    assert handle.window_id == "2026-01-01__2026-01-31"
    assert handle.total_prs == 3

    rows = [
        json.loads(l) for l in handle.prs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["pr_number"] for r in rows] == [100, 101, 102]
    assert all("merge_sha" in r and "parent_sha" in r for r in rows)

    manifest = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_prs_returned"] == 3
    assert manifest["window_days"] == 30
    assert "merged:2026-01-01..2026-01-31" in manifest["github_query"]


def test_scrape_rejects_inverted_window(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(
        aggregation, "_GitHubClient",
        lambda **kw: _FakeClient(
            repo="vllm-project/vllm", search_items=[], details={}, files={}
        ),
    )
    with pytest.raises(ValueError, match="before"):
        aggregation.scrape(
            window_start="2026-06-03",
            window_end="2026-01-01",
            token="fake",
        )


def test_scrape_dedupes_pr_numbers_across_shards(monkeypatch, tmp_path: Path):
    """Same PR number returned in two month shards (rare but possible)
    should still produce one row."""
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))

    # Both shards return PR 50; scrape should dedupe.
    search_items = [{"number": 50}, {"number": 50}]
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=search_items,
        details={50: _detail(50)},
        files={50: _files(50)},
    )
    monkeypatch.setattr(aggregation, "_GitHubClient", lambda **kw: fake)

    handle = aggregation.scrape(
        window_start="2026-01-01",
        window_end="2026-01-15",
        token="fake-token",
    )
    assert handle.total_prs == 1


# ── checkpoint / resume ──────────────────────────────────────────────────


def test_scrape_cleans_up_partial_and_pr_numbers_on_finalize(
    monkeypatch, tmp_path: Path
):
    """On a successful scrape, the partial + pr_numbers cache are removed.

    Earlier behavior kept them around for "audit value", but they're
    byte-equivalent (partial) or trivially rederivable (pr_numbers), so
    they just doubled disk usage. A future interrupted scrape rebuilds
    them via the resume path.
    """
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    nums = [200, 201, 202]
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[{"number": n} for n in nums],
        details={n: _detail(n) for n in nums},
        files={n: _files(n) for n in nums},
    )
    monkeypatch.setattr(aggregation, "_GitHubClient", lambda **kw: fake)

    handle = aggregation.scrape(
        window_start="2026-02-01",
        window_end="2026-02-28",
        token="fake-token",
    )
    # Final artifacts present
    assert (handle.out_dir / "prs.jsonl").exists()
    assert (handle.out_dir / "manifest.json").exists()
    # Resume artifacts gone
    assert not (handle.out_dir / "pr_numbers.json").exists()
    assert not (handle.out_dir / "prs.jsonl.partial").exists()


def test_scrape_resume_skips_already_fetched(monkeypatch, tmp_path: Path):
    """Pre-seed partial with PR 100; scrape should fetch only 101 and 102."""
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    out = tmp_path / "raw" / "2026-03-01__2026-03-31"
    out.mkdir(parents=True)

    # Seed: pr_numbers cache + partial with PR 100 already done
    (out / "pr_numbers.json").write_text(
        json.dumps([100, 101, 102]), encoding="utf-8"
    )
    seeded = _detail(100)
    seeded_pr = aggregation._fetch_pr_full(
        _FakeClient(
            repo="vllm-project/vllm",
            search_items=[],
            details={100: seeded},
            files={100: _files(100)},
        ),
        100,
    )
    (out / "prs.jsonl.partial").write_text(
        seeded_pr.model_dump_json() + "\n", encoding="utf-8"
    )

    # Now run scrape: client has 101+102 only — if it tried to fetch 100 again
    # the FakeClient's missing entry would raise KeyError.
    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={101: _detail(101), 102: _detail(102)},
        files={101: _files(101), 102: _files(102)},
    )
    monkeypatch.setattr(aggregation, "_GitHubClient", lambda **kw: fake)

    handle = aggregation.scrape(
        window_start="2026-03-01",
        window_end="2026-03-31",
        token="fake-token",
    )
    assert handle.total_prs == 3
    rows = [
        json.loads(l)
        for l in handle.prs_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [r["pr_number"] for r in rows] == [100, 101, 102]
    # search endpoint should not have been hit on resume
    assert not any(c == "/search/issues" for c in fake.calls)


def test_scrape_torn_partial_line_is_tolerated(monkeypatch, tmp_path: Path):
    """A truncated final line (simulating a crash mid-write) is dropped on resume."""
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    out = tmp_path / "raw" / "2026-04-01__2026-04-30"
    out.mkdir(parents=True)

    nums = [300, 301]
    (out / "pr_numbers.json").write_text(json.dumps(nums), encoding="utf-8")

    # Seed partial with one good line + one truncated line
    fake_seed = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={300: _detail(300)},
        files={300: _files(300)},
    )
    good = aggregation._fetch_pr_full(fake_seed, 300)
    (out / "prs.jsonl.partial").write_text(
        good.model_dump_json() + "\n" + '{"pr_number": 301, "title": "tru',
        encoding="utf-8",
    )

    fake = _FakeClient(
        repo="vllm-project/vllm",
        search_items=[],
        details={301: _detail(301)},
        files={301: _files(301)},
    )
    monkeypatch.setattr(aggregation, "_GitHubClient", lambda **kw: fake)

    handle = aggregation.scrape(
        window_start="2026-04-01",
        window_end="2026-04-30",
        token="fake-token",
    )
    assert handle.total_prs == 2  # 300 from partial, 301 re-fetched
