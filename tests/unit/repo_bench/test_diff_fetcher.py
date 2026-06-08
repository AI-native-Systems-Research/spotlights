"""Diff fetcher — fetch per-PR diffs into raw/<window_id>/diffs/."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from spotlights_engine.repo_bench import diff_fetcher, filtering, storage
from spotlights_engine.repo_bench.schemas import RawPR


# ── Fixtures ──────────────────────────────────────────────────────────


def _pr(pr_number: int, title: str = "x", **kw) -> RawPR:
    return RawPR(
        pr_number=pr_number,
        title=title,
        body="",
        merged_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        merge_sha="a" * 40,
        parent_sha="b" * 40,
        author=kw.get("author", "alice"),
        labels=[],
        files_changed=kw.get("files_changed") or [
            {"path": "vllm/foo.py", "additions": 1, "deletions": 0, "status": "modified"}
        ],
        additions_total=kw.get("additions_total", 1),
        deletions_total=kw.get("deletions_total", 0),
        commits_count=1,
        url=f"https://github.com/x/y/pull/{pr_number}",
    )


@pytest.fixture
def fixture_window(tmp_path: Path, monkeypatch) -> tuple[Path, str, Path]:
    """Build a small raw + view; return (root, window_id, view_path)."""
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    win = "2026-06-01__2026-06-03"
    raw = storage.raw_dir(win, root=tmp_path)
    raw.mkdir(parents=True)

    # PR 1 ranks above PR 2 (higher claimed %); the view file is ordered by
    # score, so PR 1 is fetched first. Tests rely on that order.
    rows = [
        _pr(1, "[Perf] X, 20.0% throughput improvement"),
        _pr(2, "[Perf] Y, 5.0% latency improvement"),
        _pr(3, "Refactor scheduler"),  # filtered out
    ]
    storage.write_jsonl(raw / "prs.jsonl", rows)

    run_dir = tmp_path / 'run'
    handle = filtering.derive(
        win,
        rules=[filtering.TitleStrictPerfClaim(), filtering.RankBySpecificityAndMagnitude()],
        run_dir=run_dir,
        data_root_override=tmp_path,
    )
    return tmp_path, win, run_dir / 'view' / 'prs.jsonl' 


# ── _fetch_one_diff stub ──────────────────────────────────────────────


@pytest.fixture
def stub_fetch_one(monkeypatch):
    """Replace _fetch_one_diff with a stub that returns canned bytes."""
    calls: list[int] = []
    bodies: dict[int, bytes] = {}

    def _stub(client, pr_number: int) -> bytes:
        calls.append(pr_number)
        return bodies.get(pr_number, f"diff for #{pr_number}\n".encode("utf-8"))

    monkeypatch.setattr(diff_fetcher, "_fetch_one_diff", _stub)
    return calls, bodies


# ── Tests ──────────────────────────────────────────────────────────────


def test_fetch_diffs_writes_per_pr_files(fixture_window, stub_fetch_one):
    root, win, vp = fixture_window
    calls, _ = stub_fetch_one

    handle = diff_fetcher.fetch_diffs(
        window_id=win, view_path=vp, token="t", data_root_override=root
    )

    assert handle.n_total == 2  # only the two perf PRs survived the filter
    assert handle.n_fetched == 2
    assert handle.n_skipped == 0
    assert sorted(calls) == [1, 2]

    diffs = list(handle.diffs_dir.glob("*.diff"))
    assert {p.name for p in diffs} == {"1.diff", "2.diff"}
    assert (handle.diffs_dir / "1.diff").read_bytes() == b"diff for #1\n"


def test_fetch_diffs_manifest_records_sha256(fixture_window, stub_fetch_one):
    root, win, vp = fixture_window
    handle = diff_fetcher.fetch_diffs(
        window_id=win, view_path=vp, token="t", data_root_override=root
    )

    manifest = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest.keys()) == {"1", "2"}
    expected_sha = hashlib.sha256(b"diff for #1\n").hexdigest()
    assert manifest["1"]["sha256"] == expected_sha
    assert manifest["1"]["bytes"] == len(b"diff for #1\n")
    # fetched_at parses as ISO datetime
    datetime.fromisoformat(manifest["1"]["fetched_at"])


def test_fetch_diffs_idempotent_default(fixture_window, stub_fetch_one):
    root, win, vp = fixture_window
    calls, _ = stub_fetch_one

    diff_fetcher.fetch_diffs(window_id=win, view_path=vp, token="t", data_root_override=root)
    assert sorted(calls) == [1, 2]

    # Second call: both already on disk, no fetches.
    handle2 = diff_fetcher.fetch_diffs(
        window_id=win, view_path=vp, token="t", data_root_override=root
    )
    assert handle2.n_fetched == 0
    assert handle2.n_skipped == 2
    assert sorted(calls) == [1, 2]  # unchanged


def test_fetch_diffs_refresh_overwrites(fixture_window, stub_fetch_one):
    root, win, vp = fixture_window
    calls, bodies = stub_fetch_one

    diff_fetcher.fetch_diffs(window_id=win, view_path=vp, token="t", data_root_override=root)
    first_sha = json.loads(
        (storage.raw_dir(win, root=root) / "diffs" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )["1"]["sha256"]

    # Stub returns different bytes for #1 next time
    bodies[1] = b"fresh diff for #1\n"

    handle = diff_fetcher.fetch_diffs(
        window_id=win, view_path=vp, token="t", data_root_override=root, refresh=True
    )
    assert handle.n_fetched == 2
    assert handle.n_skipped == 0
    assert sorted(calls) == [1, 1, 2, 2]  # both fetched twice (once each run)
    new_sha = hashlib.sha256(b"fresh diff for #1\n").hexdigest()
    assert new_sha != first_sha
    manifest = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    assert manifest["1"]["sha256"] == new_sha
    assert (handle.diffs_dir / "1.diff").read_bytes() == b"fresh diff for #1\n"


def test_fetch_diffs_skips_failed_diff_keeps_going(fixture_window, monkeypatch):
    root, win, vp = fixture_window

    def _flaky(client, pr_number: int) -> bytes:
        if pr_number == 1:
            raise diff_fetcher._DiffFetchError("simulated 404")
        return f"diff for #{pr_number}\n".encode("utf-8")

    monkeypatch.setattr(diff_fetcher, "_fetch_one_diff", _flaky)

    handle = diff_fetcher.fetch_diffs(
        window_id=win, view_path=vp, token="t", data_root_override=root
    )
    # PR 2 succeeds; PR 1 logged + skipped.
    assert handle.n_fetched == 1
    assert (handle.diffs_dir / "2.diff").exists()
    assert not (handle.diffs_dir / "1.diff").exists()
    manifest = json.loads(handle.manifest_path.read_text(encoding="utf-8"))
    assert "1" not in manifest
    assert "2" in manifest


def test_fetch_diffs_missing_view_raises(fixture_window, tmp_path):
    root, win, _ = fixture_window
    with pytest.raises(FileNotFoundError, match="no view at"):
        diff_fetcher.fetch_diffs(
            window_id=win, view_path=tmp_path / "nonexistent.jsonl",
            token="t", data_root_override=root,
        )


def test_fetch_diffs_writes_atomically(fixture_window, monkeypatch):
    """A crash mid-write must not leave a `.tmp-*` file at the diff path."""
    root, win, vp = fixture_window

    crashed_at = []

    def _crashing(client, pr_number: int) -> bytes:
        if pr_number == 2:
            crashed_at.append(pr_number)
            raise RuntimeError("boom")
        return f"diff for #{pr_number}\n".encode("utf-8")

    monkeypatch.setattr(diff_fetcher, "_fetch_one_diff", _crashing)

    with pytest.raises(RuntimeError, match="boom"):
        diff_fetcher.fetch_diffs(
            window_id=win, view_path=vp, token="t", data_root_override=root
        )

    assert crashed_at == [2]
    diffs_dir = storage.raw_dir(win, root=root) / "diffs"
    # PR 1 succeeded; its diff is on disk under the canonical name.
    assert (diffs_dir / "1.diff").read_bytes() == b"diff for #1\n"
    # No leftover temp files for PR 2.
    leftovers = list(diffs_dir.glob("2.diff.tmp-*"))
    assert leftovers == []
