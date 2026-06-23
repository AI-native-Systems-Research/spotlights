"""Stage 03 — candidate generation wiring.

State-machine tests live in `test_runner.py`; this file covers the
delegation to `claude_subprocess.run_claude`: prompt assembly, schema
construction, error propagation, and the safety-order invariant.

The `--from-stage 03 …` smoke against a real subject repo is a manual
verification step (per the plan); not in here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline,
)
from spotlights_engine.signal_pipeline.schemas import (
    AnomalyLite,
    CandidateDraft,
    Signals,
    TraceSummaryLite,
    WorkloadProfileLite,
)
from spotlights_engine.signal_pipeline.stages import s03_candidate_generation
from spotlights_engine.signal_pipeline.stages.s03_candidate_generation import (
    CandidateGenerationError,
    CandidateList,
)


def _signals() -> Signals:
    return Signals(
        workload=WorkloadProfileLite(workload_id="w-test"),
        traces=[TraceSummaryLite(trace_id="t1", summary="x")],
        anomalies=[AnomalyLite(anomaly_id="a1", type="latency", description="x")],
    )


def _tree() -> ProjectTree:
    return ProjectTree(
        repository=Repository(name="r", summary="s"),
        modules=[Module(name="m", path="m", description="d")],
    )


def _candidate(id_: str = "cand-0001") -> CandidateDraft:
    return CandidateDraft(
        id=id_,
        file="m/x.py",
        line_start=1,
        line_end=2,
        symbol="f",
        kind="function",
        description="d",
        current_approach="c",
        evolve_rationale="r",
        estimated_impact="medium",
        estimated_impact_explanation="e",
    )


# ── Prompt assembly ──────────────────────────────────────────────────────


def test_build_prompt_substitutes_inputs(tmp_path):
    prompt = s03_candidate_generation._build_prompt(_signals(), _tree(), tmp_path / "subject")
    # Inputs are substituted (their JSON strings appear) and the literal
    # `{` placeholders for the output example block survive `.format()`.
    assert "w-test" in prompt
    assert "r" in prompt and "summary" in prompt
    assert str(tmp_path / "subject") in prompt
    # Output-shape example uses {{ }} → emits literal { } after format.
    assert '"candidates"' in prompt
    # No raw format-field markers left over.
    assert "{signals_json}" not in prompt
    assert "{project_tree_json}" not in prompt
    assert "{subject_root}" not in prompt
    assert "{max_candidates_clause}" not in prompt


def test_build_prompt_default_says_multiple_allowed(tmp_path):
    """Without max_candidates, the prompt keeps the open-ended language."""
    prompt = s03_candidate_generation._build_prompt(_signals(), _tree(), tmp_path / "subject")
    assert "Multiple candidates are allowed" in prompt
    assert "at most" not in prompt.split("## Output")[0]


def test_build_prompt_caps_count_when_max_candidates_set(tmp_path):
    """max_candidates=N injects an `at most N` instruction into the prompt."""
    prompt = s03_candidate_generation._build_prompt(
        _signals(), _tree(), tmp_path / "subject", max_candidates=5
    )
    assert "at most 5 candidates" in prompt
    # The default open-ended sentence should be replaced, not duplicated.
    assert "Multiple candidates are allowed" not in prompt


def test_candidate_list_schema_is_closed():
    """The envelope's JSON schema must mark `additionalProperties: false`
    on every object level — claude's structured-output endpoint requires
    it. Regression guard for future schema edits."""
    schema = CandidateList.model_json_schema()
    assert schema["additionalProperties"] is False
    # Spot-check the inner CandidateDraft definition (referenced via $defs).
    cand_def = schema.get("$defs", {}).get("CandidateDraft")
    assert cand_def is not None
    assert cand_def.get("additionalProperties") is False


# ── _run_candidate_generation wiring ────────────────────────────────────


@pytest.mark.no_stub_stages
def test_run_calls_claude_with_expected_args(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_run_claude(*, prompt, log_dir, json_schema, cwd, max_turns,
                       timeout_s, permission_mode, allowed_tools, **kwargs):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        captured.update(
            prompt=prompt,
            log_dir=log_dir,
            json_schema=json_schema,
            cwd=cwd,
            max_turns=max_turns,
            timeout_s=timeout_s,
            permission_mode=permission_mode,
            allowed_tools=tuple(allowed_tools),
        )
        return ClaudeRunResult(
            structured_output={"candidates": [_candidate().model_dump(mode="json")]},
            duration_s=0.1,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    subject = tmp_path / "subject"
    subject.mkdir()
    log_dir = tmp_path / "log"

    out = s03_candidate_generation._run_candidate_generation(
        signals=_signals(),
        project_tree=_tree(),
        subject_root=subject,
        log_dir=log_dir,
    )

    assert isinstance(out, list) and len(out) == 1
    assert isinstance(out[0], CandidateDraft) and out[0].id == "cand-0001"
    assert captured["cwd"] == subject
    assert captured["permission_mode"] == "plan"
    assert captured["allowed_tools"] == ("Read",)
    assert captured["log_dir"] == log_dir
    # Schema text is the closed envelope schema.
    parsed_schema = json.loads(captured["json_schema"])
    assert parsed_schema["additionalProperties"] is False


@pytest.mark.no_stub_stages
def test_run_propagates_claude_error(monkeypatch, tmp_path):
    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output=None,
            duration_s=2.5,
            error="claude exit=1: stderr='boom'",
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    with pytest.raises(CandidateGenerationError, match="exit=1"):
        s03_candidate_generation._run_candidate_generation(
            signals=_signals(),
            project_tree=_tree(),
            subject_root=tmp_path,
            log_dir=tmp_path / "log",
        )


@pytest.mark.no_stub_stages
def test_run_rejects_invalid_envelope_payload(monkeypatch, tmp_path):
    def fake_run_claude(**_kw):
        from spotlights_engine.signal_pipeline.claude_subprocess import ClaudeRunResult

        return ClaudeRunResult(
            structured_output={"wrong_key": []},
            duration_s=0.1,
        )

    import spotlights_engine.signal_pipeline.claude_subprocess as cs

    monkeypatch.setattr(cs, "run_claude", fake_run_claude)

    # Pydantic ValidationError surfaces — caller can wrap as needed.
    with pytest.raises(Exception):
        s03_candidate_generation._run_candidate_generation(
            signals=_signals(),
            project_tree=_tree(),
            subject_root=tmp_path,
            log_dir=tmp_path / "log",
        )


# ── End-to-end via the runner ───────────────────────────────────────────


def test_run_pipeline_invokes_stage_03_via_conftest_stub(tmp_path):
    """Sanity: the conftest stub is wired so a default run produces real
    Candidates on disk. This guards against a future refactor that
    forgets to update the conftest stub when changing the stage 03
    monkeypatch target."""
    run_dir = tmp_path / "run"
    run_pipeline(
        SignalPipelineInput(subject_root=tmp_path),
        run_dir=run_dir,
        stages=StageSelection(from_stage="01", to_stage="03"),
    )
    written = json.loads((run_dir / "03_candidates.json").read_text())
    assert isinstance(written, list) and len(written) == 2
    assert {c["id"] for c in written} == {"cand-0001", "cand-0002"}


# ── Safety order ────────────────────────────────────────────────────────


def test_safety_order_no_top_level_main_imports():
    forbidden = {"extract", "ExtractorConfig", "ModulesExtractorInput"}
    bound = set(dir(s03_candidate_generation)) & forbidden
    assert bound == set(), (
        f"safety-order regression: {bound} bound at module top of s03_candidate_generation"
    )


def test_safety_order_no_top_level_claude_subprocess_import():
    """`claude_subprocess` itself is part of this package, so importing
    it isn't a safety-order issue per the plan; but we don't want
    `run_claude` *bound at module top* either, since the test seam
    monkeypatches it via the `claude_subprocess` module attribute."""
    assert "run_claude" not in dir(s03_candidate_generation)
