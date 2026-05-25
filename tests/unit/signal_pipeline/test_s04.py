"""Stage 04 — change generation wiring.

Per-candidate fan-out: tests cover the prompt assembly, the runner-owned
fields (`change_id` / `candidate_ref`) being filled deterministically
regardless of LLM output, and error propagation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.signal_pipeline import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.schemas import Change
from spotlights_engine.signal_pipeline.stages import s04_change_generation
from spotlights_engine.signal_pipeline.stages.s04_change_generation import (
    ChangeGenerationError,
    _ChangeProposal,
)


def _candidate(id_: str = "cand-0001") -> Candidate:
    return Candidate(
        id=id_,
        file="m/x.py",
        line_start=10,
        line_end=20,
        symbol="hot_path",
        kind="function",
        description="d",
        current_approach="c",
        evolve_rationale="r",
        estimated_impact="medium",
        estimated_impact_explanation="e",
    )


# ── Prompt + schema ─────────────────────────────────────────────────────


def test_build_prompt_substitutes_inputs(tmp_path):
    prompt = s04_change_generation._build_prompt(_candidate(), tmp_path)
    assert "cand-0001" in prompt
    assert "hot_path" in prompt
    assert str(tmp_path) in prompt
    # No leftover format markers.
    assert "{candidate_json}" not in prompt
    assert "{subject_root}" not in prompt


def test_proposal_schema_excludes_runner_owned_fields():
    """`change_id` and `candidate_ref` must NOT be in the LLM-output
    schema — the runner sets those, and exposing them invites id
    mismatches that corrupt the fan-out manifest."""
    schema = _ChangeProposal.model_json_schema()
    assert "change_id" not in schema["properties"]
    assert "candidate_ref" not in schema["properties"]
    assert schema["additionalProperties"] is False


# ── _generate_change wiring ─────────────────────────────────────────────


@pytest.mark.no_stub_stages
def test_generate_change_fills_runner_owned_fields(monkeypatch, tmp_path):
    """LLM proposes mechanism/effect/etc.; the runner stamps
    `change_id` and `candidate_ref` deterministically from the input
    candidate. This holds even if the LLM (counterfactually) tried to
    pick its own ids — the schema doesn't expose those fields."""

    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output={
                "change_type": "tune",
                "mechanism": "lower the watermark",
                "expected_effect": "p95 latency ↓",
                "required_changes": "scheduler.py:Scheduler",
                "evaluation_metric": "agentic-mix latency vs baseline",
            },
            duration_s=0.05,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    change = s04_change_generation._generate_change(
        _candidate("cand-0042"), tmp_path, tmp_path / "log"
    )
    assert isinstance(change, Change)
    assert change.candidate_ref == "cand-0042"
    assert change.change_id == "chg-cand-0042"
    assert change.change_type == "tune"
    assert change.mechanism == "lower the watermark"


@pytest.mark.no_stub_stages
def test_generate_change_propagates_claude_error(monkeypatch, tmp_path):
    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output=None, duration_s=1.0, error="claude exit=2"
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    with pytest.raises(ChangeGenerationError, match="cand-0001.*exit=2"):
        s04_change_generation._generate_change(
            _candidate(), tmp_path, tmp_path / "log"
        )


def test_generate_change_uses_per_candidate_log_dir(monkeypatch, tmp_path):
    """The runner passes `ctx.log_dir / candidate_id` so per-candidate
    raw streams don't collide. Verify _generate_change is called with
    a candidate-specific path when invoked through run_one.

    Conftest stubs for stages 01/02/03 stay active (the autouse fixture
    runs); we only override stage 04's stub for this test so the runner
    threads candidates through to our capture."""
    seen: list[Path] = []

    def fake_generate_change(candidate, subject_root, log_dir):
        seen.append(log_dir)
        return Change(
            change_id=f"chg-{candidate.id}",
            candidate_ref=candidate.id,
            change_type="other",
            mechanism="m",
            expected_effect="e",
            required_changes="r",
            evaluation_metric="v",
        )

    monkeypatch.setattr(
        s04_change_generation, "_generate_change", fake_generate_change
    )

    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="01", to_stage="04"),
    )
    # Two candidates from the conftest stub → two log dirs, each named
    # by candidate id.
    log_root = run_dir.resolve() / "_logs" / "04_changes"
    assert sorted(p.name for p in seen) == ["cand-0001", "cand-0002"]
    assert all(p.parent == log_root for p in seen)


# ── End-to-end via the runner ───────────────────────────────────────────


def test_run_pipeline_writes_per_candidate_change_files(tmp_path):
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="01", to_stage="04"),
    )
    fanout = run_dir / "04_changes"
    assert (fanout / "cand-0001.json").exists()
    assert (fanout / "cand-0002.json").exists()
    one = json.loads((fanout / "cand-0001.json").read_text())
    assert one["candidate_ref"] == "cand-0001"
    assert one["change_id"].startswith("chg-cand-0001")


# ── Safety order ────────────────────────────────────────────────────────


def test_safety_order_no_top_level_main_imports():
    forbidden = {"extract", "ExtractorConfig", "ModulesExtractorInput", "run_claude"}
    bound = set(dir(s04_change_generation)) & forbidden
    assert bound == set(), (
        f"safety-order regression: {bound} bound at module top of s04_change_generation"
    )
