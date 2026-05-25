"""Stage 04 — change generation (Bundle D part 1).

**Phase 1 stub.** Emits one placeholder `Change` per upstream `Candidate`.
Step 6 of the implementation order replaces `run_one` with a `claude -p`
call that takes the candidate as the brief and returns a structured
`Change`.

Fan-out stage. Per-id files at `04_changes/<candidate_id>.json`,
`_manifest.json` records `upstream_candidates_hash` and `covered_ids`.
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.signal_pipeline.schemas import Change
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> Any:
    # Single-stage parse_artifact is unused for fan-out stages, but the
    # field is required on StageSpec. Returning the raw dict {id: payload}
    # is the convention the runner uses when assembling the fan-out view.
    return raw


def parse_item(raw: Any) -> Change:
    return Change.model_validate(raw)


def ids_from_upstream(candidates: Any) -> list[str]:
    # Upstream is the parsed `list[Candidate]` from stage 03.
    if not isinstance(candidates, list):
        raise TypeError(
            f"upstream of stage 04 must be list[Candidate]; got {type(candidates).__name__}"
        )
    return [c.id if isinstance(c, Candidate) else c["id"] for c in candidates]


def run_one(ctx: StageContext, id_: str) -> Change:
    # Phase 1 stub.
    return Change(
        change_id=f"chg-{id_}",
        candidate_ref=id_,
        change_type="other",
        mechanism="Phase 1 stub change — replace in step 6.",
        expected_effect="Stubbed.",
        required_changes="Stubbed.",
        evaluation_metric="Stubbed.",
    )


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


__all__ = ["SPEC"]
