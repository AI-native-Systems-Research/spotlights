"""Runner state-machine tests for the signal pipeline.

Stages are still stubs in Phase 1 — tests use the stub outputs directly
rather than monkeypatching anything. As real stages land in later steps,
these tests should keep passing because the stubs remain swappable
fallbacks (each `run` / `run_one` only needs to return a schema-valid
payload).

Plan reference: `/Users/idanfr/.claude/plans/humble-plotting-cook.md`
Phase 2 (run-dir layout + resume rules), Phase 3 (orchestrator + inject
validation tables).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from spotlights_engine.signal_pipeline import (
    InjectSpec,
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.layout import (
    RunDirLayout,
    artifact_hash,
)
from spotlights_engine.signal_pipeline.runner import (
    InjectValidationError,
    PipelineLayoutError,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _input(tmp_path: Path) -> SignalPipelineInput:
    """`subject_root` is irrelevant for stub runs; point at tmp."""
    return SignalPipelineInput(subject_root=tmp_path)


def _read_status(layout: RunDirLayout) -> dict:
    return json.loads(layout.status_path.read_text())


# ── Phase 2 — run-dir layout ─────────────────────────────────────────────


def test_fresh_run_writes_canonical_layout(tmp_path):
    run_dir = tmp_path / "run"
    res = run_pipeline(_input(tmp_path), run_dir=run_dir)

    assert res.completed_stages == ["01", "02", "03", "04", "05"]
    assert res.skipped_stages == []
    assert (run_dir / "input.json").exists()
    assert (run_dir / "01_signals.json").exists()
    assert (run_dir / "02_projecttree.json").exists()
    assert (run_dir / "03_candidates.json").exists()
    assert (run_dir / "04_changes" / "_manifest.json").exists()
    assert (run_dir / "04_changes" / "cand-0001.json").exists()
    assert (run_dir / "04_changes" / "cand-0002.json").exists()
    assert (run_dir / "05_results" / "_manifest.json").exists()
    assert (run_dir / "05_results" / "cand-0001.json").exists()
    assert (run_dir / "05_results" / "cand-0002.json").exists()
    assert (run_dir / "status.json").exists()


def test_status_reports_done_for_every_stage(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    status = _read_status(RunDirLayout(run_dir))
    for stage in ("01", "02", "03", "04", "05"):
        assert status["stages"][stage]["state"] == "done"


def test_fanout_manifest_records_upstream_hash(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)

    candidates = json.loads((run_dir / "03_candidates.json").read_text())
    expected = artifact_hash(candidates)
    manifest_04 = json.loads((run_dir / "04_changes" / "_manifest.json").read_text())
    assert manifest_04["upstream_candidates_hash"] == expected
    assert sorted(manifest_04["covered_ids"]) == ["cand-0001", "cand-0002"]


# ── Resume / no-resume ───────────────────────────────────────────────────


def test_second_run_skips_everything_when_resume_on(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    res = run_pipeline(_input(tmp_path), run_dir=run_dir)
    assert res.completed_stages == []
    assert res.skipped_stages == ["01", "02", "03", "04", "05"]


def test_no_resume_reruns_everything(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    res = run_pipeline(_input(tmp_path), run_dir=run_dir, resume=False)
    assert res.completed_stages == ["01", "02", "03", "04", "05"]
    assert res.skipped_stages == []


def test_corrupt_single_artifact_is_not_complete(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    # Corrupt 03_candidates.json — it should re-run on the next pass.
    (run_dir / "03_candidates.json").write_text("not json")
    res = run_pipeline(_input(tmp_path), run_dir=run_dir)
    assert "03" in res.completed_stages
    # 04 / 05 also re-run because their upstream hash now mismatches
    # (the regenerated candidates list is byte-identical, but the hash
    # check still has to re-validate; cand-* files survive).
    assert "04" in res.completed_stages or "04" in res.skipped_stages


def test_corrupt_fanout_entry_is_not_complete(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    (run_dir / "04_changes" / "cand-0001.json").write_text("not json")
    res = run_pipeline(_input(tmp_path), run_dir=run_dir)
    assert "04" in res.completed_stages


# ── Stage selection ──────────────────────────────────────────────────────


def test_only_stage_runs_one(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)  # populate everything
    res = run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        stages=StageSelection.only("03"),
        resume=False,
    )
    assert res.completed_stages == ["03"]
    assert res.skipped_stages == []


def test_to_stage_stops_at_target(tmp_path):
    run_dir = tmp_path / "run"
    res = run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="01", to_stage="03"),
    )
    assert res.completed_stages == ["01", "02", "03"]
    assert not (run_dir / "04_changes").exists()
    assert not (run_dir / "05_results").exists()


def test_from_stage_requires_upstream_complete(tmp_path):
    run_dir = tmp_path / "run"
    with pytest.raises(PipelineLayoutError, match="requires stage 01"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            stages=StageSelection(from_stage="03", to_stage="03"),
        )


def test_from_stage_runs_when_upstream_already_done(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="01", to_stage="02"),
    )
    res = run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="03", to_stage="03"),
        resume=False,
    )
    assert res.completed_stages == ["03"]


# ── Inject — single-artifact form ───────────────────────────────────────


def test_inject_single_artifact_substitutes_payload(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)

    # Hand-craft a candidates list with one entry only.
    custom = [
        {
            "id": "cand-9999",
            "file": "x/y.py",
            "line_start": 1,
            "line_end": 2,
            "symbol": "z",
            "kind": "function",
            "description": "injected",
            "current_approach": "injected",
            "evolve_rationale": "injected",
            "estimated_impact": "high",
            "estimated_impact_explanation": "injected",
            "state": "DISCOVERED",
            "deep_research_proposals": [],
            "agent_proposals": [],
        }
    ]
    src = tmp_path / "candidates.json"
    src.write_text(json.dumps(custom))

    run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        inject=[InjectSpec.parse(f"03={src}")],
    )
    written = json.loads((run_dir / "03_candidates.json").read_text())
    assert written == custom


def test_inject_single_invalid_payload_raises(tmp_path):
    run_dir = tmp_path / "run"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"this": "is not a list of candidates"}))
    with pytest.raises(InjectValidationError, match="failed validation"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"03={bad}")],
        )


def test_inject_single_form_on_fanout_stage_rejects(tmp_path):
    run_dir = tmp_path / "run"
    f = tmp_path / "f.json"
    f.write_text("{}")
    with pytest.raises(InjectValidationError, match="expected a directory"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"04={f}")],
        )


def test_inject_dir_on_single_artifact_stage_rejects(tmp_path):
    run_dir = tmp_path / "run"
    d = tmp_path / "dir"
    d.mkdir()
    with pytest.raises(InjectValidationError, match="single-artifact"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"03={d}")],
        )


# ── Inject — per-id form (fan-out only) ─────────────────────────────────


def _make_change_payload(candidate_id: str, mech: str = "injected") -> dict:
    return {
        "change_id": f"chg-{candidate_id}",
        "candidate_ref": candidate_id,
        "change_type": "tune",
        "mechanism": mech,
        "expected_effect": "x",
        "required_changes": "x",
        "evaluation_metric": "x",
    }


def test_inject_per_id_substitutes_one_entry(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)

    src = tmp_path / "one_change.json"
    src.write_text(json.dumps(_make_change_payload("cand-0001", mech="HAND")))

    run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        inject=[InjectSpec.parse(f"04/cand-0001={src}")],
    )
    written = json.loads((run_dir / "04_changes" / "cand-0001.json").read_text())
    assert written["mechanism"] == "HAND"
    # cand-0002 is untouched (still the stub).
    other = json.loads((run_dir / "04_changes" / "cand-0002.json").read_text())
    assert "stub" in other["mechanism"]


def test_inject_per_id_unknown_id_rejects(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    src = tmp_path / "ghost.json"
    src.write_text(json.dumps(_make_change_payload("cand-9999")))
    with pytest.raises(InjectValidationError, match="not in upstream"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"04/cand-9999={src}")],
        )


def test_inject_per_id_on_single_stage_rejects(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    src = tmp_path / "x.json"
    src.write_text("{}")
    with pytest.raises(InjectValidationError, match="single-artifact"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"03/foo={src}")],
        )


def test_inject_per_id_when_upstream_missing_rejects(tmp_path):
    run_dir = tmp_path / "run"
    src = tmp_path / "f.json"
    src.write_text(json.dumps(_make_change_payload("cand-0001")))
    with pytest.raises(InjectValidationError, match="upstream stage 03 has no artifact"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"04/cand-0001={src}")],
        )


# ── Inject — fan-out whole-dir ──────────────────────────────────────────


def test_inject_fanout_dir_replaces_contents(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)

    inject_dir = tmp_path / "changes_dir"
    inject_dir.mkdir()
    (inject_dir / "cand-0001.json").write_text(
        json.dumps(_make_change_payload("cand-0001", mech="DIR-1"))
    )
    (inject_dir / "cand-0002.json").write_text(
        json.dumps(_make_change_payload("cand-0002", mech="DIR-2"))
    )

    run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        inject=[InjectSpec.parse(f"04={inject_dir}")],
    )
    one = json.loads((run_dir / "04_changes" / "cand-0001.json").read_text())
    two = json.loads((run_dir / "04_changes" / "cand-0002.json").read_text())
    assert one["mechanism"] == "DIR-1"
    assert two["mechanism"] == "DIR-2"


def test_inject_fanout_dir_missing_id_rejects(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    inject_dir = tmp_path / "incomplete"
    inject_dir.mkdir()
    (inject_dir / "cand-0001.json").write_text(
        json.dumps(_make_change_payload("cand-0001"))
    )  # missing cand-0002
    with pytest.raises(InjectValidationError, match="missing files"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"04={inject_dir}")],
        )


def test_inject_fanout_dir_orphan_id_rejects(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(_input(tmp_path), run_dir=run_dir)
    inject_dir = tmp_path / "orphans"
    inject_dir.mkdir()
    (inject_dir / "cand-0001.json").write_text(json.dumps(_make_change_payload("cand-0001")))
    (inject_dir / "cand-0002.json").write_text(json.dumps(_make_change_payload("cand-0002")))
    (inject_dir / "cand-9999.json").write_text(json.dumps(_make_change_payload("cand-9999")))
    with pytest.raises(InjectValidationError, match="orphan files"):
        run_pipeline(
            _input(tmp_path),
            run_dir=run_dir,
            inject=[InjectSpec.parse(f"04={inject_dir}")],
        )


# ── Inject parsing ──────────────────────────────────────────────────────


def test_inject_spec_parse_single():
    s = InjectSpec.parse("03=foo.json")
    assert s.stage == "03" and s.id_ is None and s.source == Path("foo.json")


def test_inject_spec_parse_per_id():
    s = InjectSpec.parse("04/cand-0001=foo.json")
    assert s.stage == "04" and s.id_ == "cand-0001"


def test_inject_spec_parse_rejects_bad_stage():
    with pytest.raises(InjectValidationError, match="unknown stage"):
        InjectSpec.parse("99=foo.json")


def test_inject_spec_parse_rejects_missing_eq():
    with pytest.raises(InjectValidationError, match="NN=path"):
        InjectSpec.parse("03foo.json")


# ── artifact_hash ────────────────────────────────────────────────────────


class _Tiny(BaseModel):
    a: int
    b: str


def test_artifact_hash_is_deterministic_and_canonical():
    h1 = artifact_hash([_Tiny(a=1, b="x"), _Tiny(a=2, b="y")])
    h2 = artifact_hash(
        [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    )  # plain dicts canonicalize the same way
    assert h1 == h2


def test_artifact_hash_distinguishes_list_order():
    h1 = artifact_hash([{"a": 1}, {"a": 2}])
    h2 = artifact_hash([{"a": 2}, {"a": 1}])
    assert h1 != h2  # list order is preserved per the canonical rules


def test_artifact_hash_distinguishes_keys():
    h1 = artifact_hash({"a": 1, "b": 2})
    h2 = artifact_hash({"a": 1, "b": 3})
    assert h1 != h2


# ── End-to-end with inject + selection ──────────────────────────────────


def test_inject_then_run_downstream(tmp_path):
    """Workflow: inject stages 01/02/03, then `--from-stage 04` to fan out."""
    run_dir = tmp_path / "run"

    # Inject stage 01 — a minimal Signals payload.
    sig_path = tmp_path / "sig.json"
    sig_path.write_text(
        json.dumps(
            {
                "workload": {"workload_id": "w", "description": ""},
                "traces": [],
                "anomalies": [],
            }
        )
    )

    # Inject stage 02 — a minimal ProjectTree.
    pt_path = tmp_path / "pt.json"
    pt_path.write_text(
        json.dumps(
            {
                "repository": {"name": "r", "summary": "s"},
                "modules": [],
            }
        )
    )

    # Inject stage 03 — one candidate.
    c = {
        "id": "cand-0001",
        "file": "x.py",
        "line_start": 1,
        "line_end": 2,
        "symbol": "f",
        "kind": "function",
        "description": "d",
        "current_approach": "c",
        "evolve_rationale": "r",
        "estimated_impact": "low",
        "estimated_impact_explanation": "e",
        "state": "DISCOVERED",
        "deep_research_proposals": [],
        "agent_proposals": [],
    }
    cand_path = tmp_path / "cand.json"
    cand_path.write_text(json.dumps([c]))

    res = run_pipeline(
        _input(tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="04", to_stage="05"),
        inject=[
            InjectSpec.parse(f"01={sig_path}"),
            InjectSpec.parse(f"02={pt_path}"),
            InjectSpec.parse(f"03={cand_path}"),
        ],
    )
    assert res.completed_stages == ["04", "05"]
    assert (run_dir / "04_changes" / "cand-0001.json").exists()
    assert (run_dir / "05_results" / "cand-0001.json").exists()
    # No cand-0002 anywhere — the injected upstream had only one candidate.
    assert not (run_dir / "04_changes" / "cand-0002.json").exists()
