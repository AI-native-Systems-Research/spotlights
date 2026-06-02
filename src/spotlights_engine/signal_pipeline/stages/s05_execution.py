"""Stage 05 — execution backend (Bundle D part 2).

Per-change fan-out: one `claude -p` session per `Change` applies the
modification in `subject_root`, then we capture which files were touched
and read them back into `ExecutionResult.file_edits`.

⚠ This stage runs claude with `permission_mode="bypassPermissions"` so
edits land without per-file prompts. The cwd is the user's
`--subject-root` directory — the user authorized this when invoking the
pipeline. Real-world usage should run this against a clean git checkout
(or a worktree / sandbox) so reverting / inspecting diffs is easy.

The LLM produces a small summary (status, files_touched, rationale).
The runner — not the LLM — assembles the `ExecutionResult` by reading
each touched file from disk and packing it as `format="after_content"`.
This avoids forcing the model to dump full file contents twice (once
via Edit/Write, once into the JSON response).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spotlights_engine.signal_pipeline.schemas import (
    Change,
    ExecutionResult,
    ExecutionStatus,
    FileEdit,
)
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


_PROMPT_TEMPLATE_PATH = (
    Path(__file__).parent.parent / "prompts" / "execution_brief.md"
)

_MAX_TURNS = 80
_TIMEOUT_S = 1800

# Conservative cap on per-file size we'll inline into ExecutionResult.
# Anything larger gets truncated with a marker — file_edits is meant for
# review, not for archival.
_MAX_FILE_BYTES = 256 * 1024


class _ExecutionSummary(BaseModel):
    """LLM-output schema for stage 05.

    Path discipline: `files_touched` entries must be subject-root-relative.
    Empty list is allowed (signal: nothing got applied) but `status` should
    then be `failed`.
    """

    model_config = ConfigDict(extra="forbid")

    status: ExecutionStatus
    files_touched: list[str] = Field(default_factory=list)
    rationale: str = ""


class ExecutionError(RuntimeError):
    pass


def parse_artifact(raw: Any) -> Any:
    return raw


def parse_item(raw: Any) -> ExecutionResult:
    return ExecutionResult.model_validate(raw)


def ids_from_upstream(changes: Any) -> list[str]:
    if not isinstance(changes, dict):
        raise TypeError(
            f"upstream of stage 05 must be dict[id, Change]; got {type(changes).__name__}"
        )
    return list(changes.keys())


def _build_prompt(change: Change, subject_root: Path) -> str:
    template = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.format(
        change_json=change.model_dump_json(indent=2),
        subject_root=str(subject_root),
    )


def _read_file_edit(subject_root: Path, rel_path: str) -> FileEdit:
    """Read a touched file from disk for the ExecutionResult.

    `rel_path` must stay inside `subject_root` (no `..` traversal).
    Files larger than `_MAX_FILE_BYTES` are truncated with a marker so
    the JSON artifact stays bounded.
    """
    target = (subject_root / rel_path).resolve()
    if not str(target).startswith(str(subject_root.resolve())):
        raise ExecutionError(
            f"file_edit path escapes subject_root: {rel_path!r}"
        )
    if not target.exists():
        # The LLM said it touched this file but it isn't on disk —
        # probably deleted, possibly hallucinated. Record the claim as
        # an after_content with an empty payload so the post-mortem
        # surfaces the discrepancy rather than burying it.
        return FileEdit(
            path=rel_path,
            format="after_content",
            payload="<<file not found on disk after stage 05; LLM may have deleted or hallucinated>>",
        )
    raw = target.read_bytes()
    if len(raw) > _MAX_FILE_BYTES:
        truncated = raw[:_MAX_FILE_BYTES].decode("utf-8", errors="replace")
        return FileEdit(
            path=rel_path,
            format="after_content",
            payload=(
                truncated
                + f"\n<<truncated: {len(raw)} bytes total, only first {_MAX_FILE_BYTES} kept>>"
            ),
        )
    return FileEdit(
        path=rel_path,
        format="after_content",
        payload=raw.decode("utf-8", errors="replace"),
    )


def _execute_change(
    change: Change,
    subject_root: Path,
    log_dir: Path,
    backend_id: str,
    on_event=None,
) -> ExecutionResult:
    """Real path. Test-monkeypatch seam.

    `claude_subprocess` is lazy-imported to keep the safety-order
    invariant consistent with stages 02–04."""
    from spotlights_engine.signal_pipeline.claude_subprocess import run_claude

    prompt = _build_prompt(change, subject_root)
    schema_text = json.dumps(_ExecutionSummary.model_json_schema())

    result = run_claude(
        prompt=prompt,
        log_dir=log_dir,
        json_schema=schema_text,
        cwd=subject_root,
        max_turns=_MAX_TURNS,
        timeout_s=_TIMEOUT_S,
        # Edits land without per-file prompts — see module docstring's
        # caveat about running on a clean git checkout / worktree.
        permission_mode="bypassPermissions",
        allowed_tools=("Read", "Write", "Edit", "Bash"),
        on_event=on_event,
    )
    if result.error is not None or result.structured_output is None:
        raise ExecutionError(
            f"stage 05 (execution) for {change.candidate_ref} failed after "
            f"{result.duration_s:.1f}s: {result.error}"
        )

    summary = _ExecutionSummary.model_validate(result.structured_output)
    file_edits = [_read_file_edit(subject_root, p) for p in summary.files_touched]

    return ExecutionResult(
        result_id=f"res-{change.candidate_ref}",
        change_ref=change.change_id,
        backend_id=backend_id,
        status=summary.status,
        file_edits=file_edits,
        rationale=summary.rationale,
        artifacts={},
    )


def run_one(ctx: StageContext, id_: str) -> ExecutionResult:
    # Upstream is `dict[id, Change]` (assembled by the runner from
    # 04_changes/<id>.json files via s04's parse_item).
    changes: dict[str, Change] = ctx.upstream["04"]
    change = changes.get(id_)
    if change is None:
        raise ExecutionError(
            f"stage 05: no change with candidate_ref {id_!r} in upstream"
        )
    if not isinstance(change, Change):
        # Defensive: parse_item should have produced Change instances,
        # but the runner could conceivably hand a dict.
        change = Change.model_validate(change)
    subject_root = ctx.signal_input.subject_root.resolve()
    per_id_log_dir = ctx.log_dir / id_
    return _execute_change(
        change=change,
        subject_root=subject_root,
        log_dir=per_id_log_dir,
        backend_id=ctx.signal_input.backend_id,
        on_event=ctx.on_event,
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


__all__ = ["SPEC", "ExecutionError"]
