"""Stage 04 — change generation (Bundle D part 1).

Real implementation. Per-candidate fan-out: one `claude -p` session per
`Candidate` produces one `Change` spec. Spec only — no file edits at
this stage. Stage 05 hands the spec to an execution backend.

The LLM produces only the *creative* fields (`change_type`, `mechanism`,
`expected_effect`, `required_changes`, `evaluation_metric`). The runner
fills `change_id` and `candidate_ref` deterministically — keeping them
out of the LLM's hands prevents id mismatches that would corrupt the
fan-out manifest.

Test seam: `_generate_change(candidate, subject_root, log_dir)`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.signal_pipeline.schemas import Change, ChangeType
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "change_generation.md"
)

_MAX_TURNS = 30
_TIMEOUT_S = 900


class _ChangeProposal(BaseModel):
    """LLM-output schema for stage 04 — only the fields the LLM owns.

    `change_id` and `candidate_ref` are runner-owned; keeping them out
    of this schema prevents the LLM from picking ids that don't match
    the candidate it was briefed with."""

    model_config = ConfigDict(extra="forbid")

    change_type: ChangeType
    mechanism: str = Field(min_length=1)
    expected_effect: str = Field(min_length=1)
    required_changes: str = Field(min_length=1)
    evaluation_metric: str = Field(min_length=1)


class ChangeGenerationError(RuntimeError):
    pass


def parse_artifact(raw: Any) -> Any:
    return raw


def parse_item(raw: Any) -> Change:
    return Change.model_validate(raw)


def ids_from_upstream(candidates: Any) -> list[str]:
    if not isinstance(candidates, list):
        raise TypeError(
            f"upstream of stage 04 must be list[Candidate]; got {type(candidates).__name__}"
        )
    return [c.id if isinstance(c, Candidate) else c["id"] for c in candidates]


def _build_prompt(candidate: Candidate, subject_root: Path) -> str:
    template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        candidate_json=candidate.model_dump_json(indent=2),
        subject_root=str(subject_root),
    )


def _generate_change(candidate: Candidate, subject_root: Path, log_dir: Path) -> Change:
    """Real path. Test-monkeypatch seam.

    `claude_subprocess` is lazy-imported to keep the safety order
    consistent with stages 01–03."""
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _build_prompt(candidate, subject_root)
    schema_text = json.dumps(_ChangeProposal.model_json_schema())

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=subject_root,
        max_turns=_MAX_TURNS,
        timeout_s=_TIMEOUT_S,
        permission_mode="plan",  # spec only — no edits
        allowed_tools=("Read",),
    )
    if result.error is not None or result.structured_output is None:
        raise ChangeGenerationError(
            f"stage 04 (change generation) for {candidate.id} failed after "
            f"{result.duration_s:.1f}s: {result.error}"
        )

    proposal = _ChangeProposal.model_validate(result.structured_output)
    return Change(
        change_id=f"chg-{candidate.id}",
        candidate_ref=candidate.id,
        change_type=proposal.change_type,
        mechanism=proposal.mechanism,
        expected_effect=proposal.expected_effect,
        required_changes=proposal.required_changes,
        evaluation_metric=proposal.evaluation_metric,
    )


def run_one(ctx: StageContext, id_: str) -> Change:
    # Upstream is `list[Candidate]` (parsed by s03's parse_artifact).
    candidates: list[Candidate] = ctx.upstream["03"]
    by_id = {c.id: c for c in candidates}
    candidate = by_id.get(id_)
    if candidate is None:
        # Should never happen — the runner derives the id list from
        # `ids_from_upstream(candidates)` above. Defensive.
        raise ChangeGenerationError(
            f"stage 04: no candidate with id {id_!r} in upstream"
        )
    subject_root = ctx.signal_input.subject_root.resolve()
    # One log dir per candidate, so per-candidate raw streams don't collide.
    per_id_log_dir = ctx.log_dir / id_
    return _generate_change(candidate, subject_root, per_id_log_dir)


SPEC = StageSpec(
    stage_id="04",
    name="changes",
    shape="fanout",
    upstream=("03",),
    parse_artifact=parse_artifact,
    parse_item=parse_item,
    upstream_for_hash="03",
    upstream_hash_field="upstream_candidates_hash",
    ids_from_upstream=ids_from_upstream,
    run_one=run_one,
)


__all__ = ["SPEC", "ChangeGenerationError"]
