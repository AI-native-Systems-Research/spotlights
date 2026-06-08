"""Filter rules + derive + view_id canonicalization."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from spotlights_engine.repo_bench import filtering, schemas, storage
from spotlights_engine.repo_bench.schemas import RawPR


# ── Fixture builder ────────────────────────────────────────────────────


def _pr(
    *,
    pr_number: int,
    title: str,
    body: str = "",
    author: str = "alice",
    files: list[dict[str, Any]] | None = None,
    additions_total: int = 10,
    deletions_total: int = 5,
) -> RawPR:
    return RawPR(
        pr_number=pr_number,
        title=title,
        body=body,
        merged_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
        merge_sha="a" * 40,
        parent_sha="b" * 40,
        author=author,
        labels=[],
        files_changed=files
        or [{"path": "vllm/foo.py", "additions": 10, "deletions": 5, "status": "modified"}],
        additions_total=additions_total,
        deletions_total=deletions_total,
        commits_count=1,
        url=f"https://github.com/vllm-project/vllm/pull/{pr_number}",
    )


# ── Predicates ─────────────────────────────────────────────────────────


def test_not_bot_keeps_humans():
    rule = filtering.NotBot()
    assert rule.keep(_pr(pr_number=1, title="x", author="alice"))
    assert rule.keep(_pr(pr_number=2, title="x", author="DarkLight1337"))


def test_not_bot_drops_known_bots():
    rule = filtering.NotBot()
    assert not rule.keep(_pr(pr_number=1, title="x", author="dependabot[bot]"))
    assert not rule.keep(_pr(pr_number=2, title="x", author="pre-commit-ci[bot]"))
    assert not rule.keep(_pr(pr_number=3, title="x", author="github-actions[bot]"))


def test_not_bot_drops_suffix_pattern():
    rule = filtering.NotBot()
    assert not rule.keep(_pr(pr_number=1, title="x", author="someone[bot]"))
    assert not rule.keep(_pr(pr_number=2, title="x", author="myname-bot"))
    assert not rule.keep(_pr(pr_number=3, title="x", author="bot-runner"))


def test_not_revert_keeps_normal_titles():
    rule = filtering.NotRevert()
    assert rule.keep(_pr(pr_number=1, title="[Perf] Speed up X by 5%"))
    assert rule.keep(_pr(pr_number=2, title="reverberation tuning"))  # `revert\b` boundary


def test_not_revert_drops_revert_titles():
    rule = filtering.NotRevert()
    assert not rule.keep(_pr(pr_number=1, title='Revert "[Foo] bar"'))
    assert not rule.keep(_pr(pr_number=2, title="[Revert] flaky test"))
    assert not rule.keep(_pr(pr_number=3, title="  revert this"))  # leading whitespace tolerated


def test_title_strict_perf_claim_matches_real_titles():
    rule = filtering.TitleStrictPerfClaim()
    # Real example shapes from the corpus.
    assert rule.keep(_pr(pr_number=1, title="[Perf] Optimize maxsim, 13.9% E2E throughput improvement"))
    assert rule.keep(_pr(pr_number=2, title="[Perf] 4.3% TTFT improvement from foo"))
    assert rule.keep(_pr(pr_number=3, title="2x speedup on Y"))


def test_title_strict_perf_claim_rejects_no_number():
    rule = filtering.TitleStrictPerfClaim()
    assert not rule.keep(_pr(pr_number=1, title="[Perf] Optimize maxsim"))
    assert not rule.keep(_pr(pr_number=2, title="Speed up X"))


def test_title_strict_perf_claim_rejects_naked_percent():
    rule = filtering.TitleStrictPerfClaim()
    # number% but not bound to a perf noun
    assert not rule.keep(_pr(pr_number=1, title="Round to 5% accuracy"))


# ── Ranker ─────────────────────────────────────────────────────────────


def test_rank_by_specificity_and_magnitude():
    rule = filtering.RankBySpecificityAndMagnitude()
    # 13.5% in 1 file → 13.5
    one_file = _pr(
        pr_number=1,
        title="[Perf] X, 13.5% TTFT improvement",
        files=[{"path": "vllm/a.py", "additions": 5, "deletions": 2, "status": "modified"}],
    )
    # 30% in 4 files → 7.5
    four_files = _pr(
        pr_number=2,
        title="[Perf] Y, 30% throughput improvement",
        files=[
            {"path": f"vllm/{n}.py", "additions": 1, "deletions": 0, "status": "modified"}
            for n in range(4)
        ],
    )
    s1 = rule.score(one_file)
    s2 = rule.score(four_files)
    assert s1 == pytest.approx(13.5)
    assert s2 == pytest.approx(7.5)
    assert s1 > s2  # narrow + big beats wider + bigger


def test_rank_returns_zero_when_no_extractable_pct():
    rule = filtering.RankBySpecificityAndMagnitude()
    pr = _pr(pr_number=1, title="2x speedup", body="")  # x-ratio, not %
    assert rule.score(pr) == 0.0


def test_rank_inputs_surfaced_for_view():
    rule = filtering.RankBySpecificityAndMagnitude()
    pr = _pr(
        pr_number=1,
        title="[Perf] X, 13.5% TTFT improvement",
        files=[
            {"path": "vllm/a.py", "additions": 5, "deletions": 2, "status": "modified"}
        ],
        additions_total=5,
        deletions_total=2,
    )
    ri = rule.rank_inputs(pr)
    assert ri == {"pct": 13.5, "files": 1, "loc": 7}


# ── view_id canonicalization ──────────────────────────────────────────


def test_view_id_predicate_order_canonical():
    """Predicates are commutative; same set → same view_id."""
    a = filtering.view_id_for(
        [filtering.NotBot(), filtering.NotRevert(), filtering.TitleStrictPerfClaim()]
    )
    b = filtering.view_id_for(
        [filtering.TitleStrictPerfClaim(), filtering.NotBot(), filtering.NotRevert()]
    )
    c = filtering.view_id_for(
        [filtering.NotRevert(), filtering.NotBot(), filtering.TitleStrictPerfClaim()]
    )
    assert a == b == c


def test_view_id_changes_with_predicate_set():
    full = filtering.view_id_for(
        [filtering.NotBot(), filtering.NotRevert(), filtering.TitleStrictPerfClaim()]
    )
    without_revert = filtering.view_id_for(
        [filtering.NotBot(), filtering.TitleStrictPerfClaim()]
    )
    assert full != without_revert


def test_view_id_changes_with_ranker_presence():
    no_ranker = filtering.view_id_for([filtering.TitleStrictPerfClaim()])
    with_ranker = filtering.view_id_for(
        [filtering.TitleStrictPerfClaim(), filtering.RankBySpecificityAndMagnitude()]
    )
    assert no_ranker != with_ranker


def test_view_id_is_short_hex():
    vid = filtering.view_id_for([filtering.TitleStrictPerfClaim()])
    assert len(vid) == 12
    int(vid, 16)  # hex


# ── Spec params ────────────────────────────────────────────────────────


def test_specs_include_regex_so_pattern_change_invalidates_view_id(monkeypatch):
    """If the bot suffix regex pattern changes, view_id must change.

    We don't actually mutate the regex (it's module-private); we assert
    the spec exposes the pattern in `params`, which `artifact_hash` will
    pick up.
    """
    spec = filtering.NotBot().to_spec()
    assert "suffix_pattern" in spec.params
    assert spec.params["suffix_pattern"]
    assert "known_bots" in spec.params
    # known_bots is sorted (deterministic across Python runs)
    assert spec.params["known_bots"] == sorted(spec.params["known_bots"])


def test_perf_claim_spec_includes_pattern():
    spec = filtering.TitleStrictPerfClaim().to_spec()
    assert "pattern" in spec.params
    assert "false_positive_strip" in spec.params
    assert spec.params["scope"] == "title-only"


# ── derive (end-to-end on a fixture window) ───────────────────────────


@pytest.fixture
def fixture_root(tmp_path: Path, monkeypatch) -> Path:
    """Set up a fake data root with a tiny raw scrape and override env."""
    monkeypatch.setenv(storage.DATA_ROOT_ENV, str(tmp_path))
    win = "2026-06-01__2026-06-03"
    raw = storage.raw_dir(win, root=tmp_path)
    raw.mkdir(parents=True)

    # Six-PR fixture mixing every case.
    rows = [
        # Bot — should drop
        _pr(pr_number=1, title="bump x", author="dependabot[bot]"),
        # Revert — should drop
        _pr(pr_number=2, title='Revert "[Foo] bar"'),
        # Perf with claim — should keep
        _pr(
            pr_number=3,
            title="[Perf] X, 13.5% TTFT improvement",
            files=[{"path": "vllm/a.py", "additions": 5, "deletions": 2, "status": "modified"}],
            additions_total=5, deletions_total=2,
        ),
        # Perf with bigger claim, more files — should rank below #3
        _pr(
            pr_number=4,
            title="[Perf] Y, 30% throughput improvement",
            files=[
                {"path": f"vllm/{n}.py", "additions": 1, "deletions": 0, "status": "modified"}
                for n in range(4)
            ],
            additions_total=4, deletions_total=0,
        ),
        # Non-perf — should drop (TitleStrictPerfClaim)
        _pr(pr_number=5, title="Refactor scheduler"),
        # Perf in body only, not title — should drop (title-only filter)
        _pr(pr_number=6, title="Optimize foo", body="13.9% E2E throughput improvement"),
    ]
    storage.write_jsonl(raw / "prs.jsonl", rows)
    return tmp_path


def test_derive_end_to_end_on_fixture(fixture_root: Path, tmp_path: Path):
    win = "2026-06-01__2026-06-03"
    handle = filtering.derive(
        win,
        rules=[
            filtering.NotBot(),
            filtering.NotRevert(),
            filtering.TitleStrictPerfClaim(),
            filtering.RankBySpecificityAndMagnitude(),
        ],
        run_dir=tmp_path / "run", data_root_override=fixture_root,
    )

    assert handle.n_kept == 2  # PRs 3 and 4
    assert handle.n_dropped == 4
    assert (handle.path / "prs.jsonl").exists()
    assert (handle.path / "manifest.json").exists()

    # Top-ranked is #3 (13.5 score), then #4 (7.5 score).
    rows = [json.loads(l) for l in (handle.path / "prs.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert [r["pr_number"] for r in rows] == [3, 4]
    assert rows[0]["score"] == pytest.approx(13.5)
    assert rows[1]["score"] == pytest.approx(7.5)
    assert rows[0]["rank_inputs"] == {"pct": 13.5, "files": 1, "loc": 7}
    assert rows[0]["raw_pr_ref"] == {"window_id": win, "pr_number": 3}


def test_derive_byte_identical_on_rerun(fixture_root: Path, tmp_path: Path):
    win = "2026-06-01__2026-06-03"
    rules = [
        filtering.NotBot(),
        filtering.NotRevert(),
        filtering.TitleStrictPerfClaim(),
        filtering.RankBySpecificityAndMagnitude(),
    ]

    h1 = filtering.derive(win, rules, run_dir=tmp_path / "run", data_root_override=fixture_root)
    bytes_1 = (h1.path / "prs.jsonl").read_bytes()

    h2 = filtering.derive(win, rules, run_dir=tmp_path / "run", data_root_override=fixture_root)
    bytes_2 = (h2.path / "prs.jsonl").read_bytes()

    assert h1.view_id == h2.view_id
    assert bytes_1 == bytes_2  # deterministic re-derive


def test_derive_predicate_order_lands_in_same_view(fixture_root: Path, tmp_path: Path):
    """Reordering predicates writes to the SAME view_id directory."""
    win = "2026-06-01__2026-06-03"
    h1 = filtering.derive(
        win,
        [
            filtering.NotBot(),
            filtering.NotRevert(),
            filtering.TitleStrictPerfClaim(),
            filtering.RankBySpecificityAndMagnitude(),
        ],
        run_dir=tmp_path / "run", data_root_override=fixture_root,
    )
    h2 = filtering.derive(
        win,
        [
            filtering.TitleStrictPerfClaim(),
            filtering.NotBot(),
            filtering.NotRevert(),
            filtering.RankBySpecificityAndMagnitude(),
        ],
        run_dir=tmp_path / "run", data_root_override=fixture_root,
    )
    assert h1.view_id == h2.view_id
    assert h1.path == h2.path


def test_derive_rejects_two_rankers(fixture_root: Path, tmp_path: Path):
    win = "2026-06-01__2026-06-03"
    with pytest.raises(ValueError, match="at most one ranker"):
        filtering.derive(
            win,
            [
                filtering.RankBySpecificityAndMagnitude(),
                filtering.RankBySpecificityAndMagnitude(),
            ],
            run_dir=tmp_path / "run", data_root_override=fixture_root,
        )


def test_derive_rejects_empty_rules(fixture_root: Path, tmp_path: Path):
    win = "2026-06-01__2026-06-03"
    with pytest.raises(ValueError, match="at least one rule"):
        filtering.derive(win, [], run_dir=tmp_path / "run", data_root_override=fixture_root)


def test_derive_missing_raw_raises(fixture_root: Path, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="no raw scrape"):
        filtering.derive(
            "9999-01-01__9999-12-31",
            [filtering.NotBot()],
            run_dir=tmp_path / "run", data_root_override=fixture_root,
        )


def test_derive_manifest_records_canonical_specs(fixture_root: Path, tmp_path: Path):
    win = "2026-06-01__2026-06-03"
    handle = filtering.derive(
        win,
        [
            filtering.NotRevert(),  # alphabetically after not-bot
            filtering.NotBot(),
            filtering.TitleStrictPerfClaim(),
        ],
        run_dir=tmp_path / "run", data_root_override=fixture_root,
    )
    manifest = json.loads((handle.path / "manifest.json").read_text(encoding="utf-8"))
    rule_names = [r["name"] for r in manifest["rules"]]
    # Predicates should be sorted by name in the manifest (canonical order).
    assert rule_names == ["not-bot", "not-revert", "title-strict-perf-claim"]
