"""Stage 05 — execution backend (Bundle D part 2).

**Phase 1 stub.** Emits one placeholder `ExecutionResult` per upstream
`Change`. Step 7 of the implementation order replaces `run_one` with a
`claude -p` invocation that has file-edit permissions, briefed with the
`Change` and `cwd=signal_input.subject_root`.

Fan-out stage. Per-id files at `05_results/<candidate_id>.json`,
`_manifest.json` records `upstream_changes_hash` and `covered_ids`.
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.signal_pipeline.schemas import Change, ExecutionResult
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> Any:
    # See note in s04 — fan-out stages parse via `parse_item` per entry.
    return raw


def parse_item(raw: Any) -> ExecutionResult:
    return ExecutionResult.model_validate(raw)


def ids_from_upstream(changes: Any) -> list[str]:
    # Upstream of stage 05 is stage 04 — the runner assembles this as
    # `dict[candidate_id, Change]` (sorted by id). We key fan-out 05 on the
    # same candidate ids so 04/05 stay aligned 1:1.
    if not isinstance(changes, dict):
        raise TypeError(
            f"upstream of stage 05 must be dict[id, Change]; got {type(changes).__name__}"
        )
    return list(changes.keys())


def run_one(ctx: StageContext, id_: str) -> ExecutionResult:
    upstream_change = ctx.upstream["04"][id_]
    if isinstance(upstream_change, Change):
        change_id = upstream_change.change_id
    else:
        change_id = upstream_change["change_id"]
    # Phase 1 stub.
    return ExecutionResult(
        result_id=f"res-{id_}",
        change_ref=change_id,
        backend_id=ctx.signal_input.backend_id,
        status="applied",
        file_edits=[],
        rationale="Phase 1 stub result — replace in step 7.",
        artifacts={},
    )


SPEC = StageSpec(
    stage_id="05",
    name="results",
    shape="fanout",
    upstream=("04",),
    parse_artifact=parse_artifact,
    parse_item=parse_item,
    upstream_for_hash="04",
    upstream_hash_field="upstream_changes_hash",
    ids_from_upstream=ids_from_upstream,
    run_one=run_one,
)


__all__ = ["SPEC"]
