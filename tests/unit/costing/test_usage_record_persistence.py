"""Resume-safety mechanics for durable usage records.

The run manifest is a pure aggregation over on-disk `UsageRecord`s, so the
plan (design/manifest_plan.md §5) treats these persistence properties as
first-class: session-scoped clearing must not double-count or lose prior
sessions, partial `.tmp` writes must be tolerated, and provenance must be
pinned once and read back unchanged on resume.
"""

from __future__ import annotations

from pathlib import Path

from spotlights_engine.costing.records import UsageRecord
from spotlights_engine.schemas.common import SpotlightContext
from spotlights_engine.spotlights_manager import persistence as P


def _record(*, session_index: int, invocation_index: int, cli: str) -> UsageRecord:
    return UsageRecord(
        step="candidate_discovery",
        module_qualified_name="pkg/a",
        session_index=session_index,
        invocation_index=invocation_index,
        provider="anthropic" if cli == "claude" else "openai",
        cli=cli,  # type: ignore[arg-type]
        model="m1",
        role="candidate_discovery",
        input=10,
        output=5,
    )


def test_write_read_roundtrip(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    rec = _record(session_index=1, invocation_index=0, cli="claude")
    P.write_usage_record(mp, rec)

    records, notes = P.read_usage_records(mp)
    assert notes == []
    assert len(records) == 1
    assert records[0] == rec


def test_filename_encodes_idempotency_key_so_rewrite_replaces(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    rec = _record(session_index=1, invocation_index=0, cli="codex")
    P.write_usage_record(mp, rec)
    # Same key, larger counts (e.g. a re-run producing the final total): the
    # file is replaced, not duplicated.
    P.write_usage_record(
        mp, rec.model_copy(update={"input": 999})
    )

    records, _ = P.read_usage_records(mp)
    assert len(records) == 1
    assert records[0].input == 999


def test_session_scoped_clear_preserves_other_sessions(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    P.write_usage_record(mp, _record(session_index=1, invocation_index=0, cli="claude"))
    P.write_usage_record(mp, _record(session_index=2, invocation_index=0, cli="claude"))

    P.clear_usage_records(mp, "candidate_discovery", session_index=2)

    records, _ = P.read_usage_records(mp)
    # Session 1's spend must survive — aggregation sums across sessions to
    # reflect the run's real cost.
    assert [r.session_index for r in records] == [1]


def test_session_prefix_is_not_confused_by_multi_digit_sessions(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    P.write_usage_record(mp, _record(session_index=1, invocation_index=0, cli="claude"))
    P.write_usage_record(mp, _record(session_index=11, invocation_index=0, cli="claude"))

    # Clearing "s1." must not also nuke "s11." (trailing dot disambiguates).
    P.clear_usage_records(mp, "candidate_discovery", session_index=1)

    records, _ = P.read_usage_records(mp)
    assert [r.session_index for r in records] == [11]


def test_clear_without_session_wipes_whole_step(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    P.write_usage_record(mp, _record(session_index=1, invocation_index=0, cli="claude"))
    P.write_usage_record(mp, _record(session_index=2, invocation_index=0, cli="claude"))

    P.clear_usage_records(mp, "candidate_discovery", session_index=None)

    records, _ = P.read_usage_records(mp)
    assert records == []


def test_partial_and_unreadable_files_are_tolerated_and_noted(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    P.write_usage_record(mp, _record(session_index=1, invocation_index=0, cli="claude"))
    usage_dir = mp.usage_dir("candidate_discovery")
    # An interrupted atomic write leaves a `.tmp` sibling; a corrupt file may
    # also survive a crash. Neither must break aggregation.
    (usage_dir / "s1.i0001.codex.json.tmp").write_text("{partial", encoding="utf-8")
    (usage_dir / "s1.i0002.codex.json").write_text("not json", encoding="utf-8")

    records, notes = P.read_usage_records(mp)
    assert len(records) == 1
    assert any("partial" in n for n in notes)
    assert any("unreadable" in n for n in notes)


def test_clear_discovery_artifacts_clears_usage_session_scoped(tmp_path: Path) -> None:
    mp = P.ModulePaths(tmp_path)
    P.write_usage_record(mp, _record(session_index=1, invocation_index=0, cli="claude"))
    P.write_usage_record(mp, _record(session_index=2, invocation_index=0, cli="claude"))

    # The orchestrator redoes a step for the current session only; the step's
    # clear hook must drop that session's records in lockstep with its outputs.
    P.clear_discovery_artifacts(mp, session_index=2)

    records, _ = P.read_usage_records(mp)
    assert [r.session_index for r in records] == [1]


def test_provenance_pinned_in_manifest_and_read_back_unchanged(tmp_path: Path) -> None:
    paths = P.ManagerPaths(tmp_path)
    provenance = {
        "repo_url": "https://example.test/repo.git",
        "target_commit_sha": "abc123",
        "spotlights_commit_sha": "def456",
    }
    P.init_manifest(
        paths,
        input_fingerprint={"a": 1},
        config_fingerprint={"b": 2},
        context=SpotlightContext(objective="reduce latency"),
        provenance=provenance,
    )

    manifest = P.read_manifest(paths)
    assert manifest is not None
    # Resume reads provenance back rather than recomputing it, so a resume from
    # a different working tree can't silently rewrite it.
    assert manifest["provenance"] == provenance
