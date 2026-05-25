"""Stage 03 — candidate generation (Bundle C).

Real implementation. Wraps the Candidate-list output in a small envelope
model so we can hand a closed `additionalProperties: false` JSON schema
to claude (the structured-output endpoint requires it). Stage's `run`
delegates to `_run_candidate_generation`, which is the test-monkeypatch
seam.

The prompt template lives in `signal_pipeline/prompts/candidate_generation.md`
so iterating on the prompt doesn't require a code change. The user said
they're still tuning prompts manually — this is the seam.

`Read` is the only tool granted to claude. `get_raw_trace` is *not*
exposed: the prompt instructs the model to treat raw-trace drills as
unavailable and rely on TraceSummary + source drills.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.signal_pipeline.schemas import Signals
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "candidate_generation.md"
)

# Subprocess budget. Mirrors the modules_extractor defaults.
_MAX_TURNS = 60
_TIMEOUT_S = 1800


class CandidateList(BaseModel):
    """Output envelope for stage 03's claude session.

    Wrapping `list[Candidate]` in a closed object satisfies claude's
    structured-output requirement (`additionalProperties: false` at every
    object level). The runner unwraps `.candidates` before persisting.
    """

    model_config = ConfigDict(extra="forbid")

    candidates: list[Candidate] = Field(default_factory=list)


class CandidateGenerationError(RuntimeError):
    pass


def parse_artifact(raw: Any) -> list[Candidate]:
    if not isinstance(raw, list):
        raise TypeError(
            f"03_candidates.json must be a JSON list of Candidates; got {type(raw).__name__}"
        )
    return [Candidate.model_validate(item) for item in raw]


def _build_prompt(signals: Signals, project_tree: ProjectTree, subject_root: Path) -> str:
    template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        signals_json=signals.model_dump_json(indent=2),
        project_tree_json=project_tree.model_dump_json(indent=2),
        subject_root=str(subject_root),
    )


def _run_candidate_generation(
    signals: Signals,
    project_tree: ProjectTree,
    subject_root: Path,
    log_dir: Path,
) -> list[Candidate]:
    """Real path. Test-monkeypatch seam.

    Imports `claude_subprocess` lazily so the package can be imported
    without `claude` on PATH (the `_check_layout()` precondition is the
    canonical fail site).
    """
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _build_prompt(signals, project_tree, subject_root)
    schema_text = json.dumps(CandidateList.model_json_schema())

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=subject_root,
        max_turns=_MAX_TURNS,
        timeout_s=_TIMEOUT_S,
        permission_mode="plan",
        allowed_tools=("Read",),
    )
    if result.error is not None or result.structured_output is None:
        raise CandidateGenerationError(
            f"stage 03 (candidate generation) failed after {result.duration_s:.1f}s: "
            f"{result.error}"
        )

    envelope = CandidateList.model_validate(result.structured_output)
    return envelope.candidates


def run(ctx: StageContext) -> list[Candidate]:
    signals = ctx.upstream["01"]
    project_tree = ctx.upstream["02"]
    subject_root = ctx.signal_input.subject_root.resolve()
    return _run_candidate_generation(
        signals=signals,
        project_tree=project_tree,
        subject_root=subject_root,
        log_dir=ctx.log_dir,
    )


SPEC = StageSpec(
    stage_id="03",
    name="candidates",
    shape="single",
    upstream=("01", "02"),
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC", "CandidateList", "CandidateGenerationError"]
