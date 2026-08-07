"""Thin wrapper around a centralized Claude session for the per-pair step-4 agent.

Reuses `spotlights_engine.llm_session` for the spawn + parse + env scrubbing +
transport retry, and maps the shared `SessionResult` into the step's
`PairRunResult` shape (which unwraps the schema's `{"proposals": [...]}`).
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.llm_session import (
    ClaudeSession,
    ClaudeSessionOptions,
    run_with_retry,
)
from spotlights_engine.proposal_from_finding_creator.errors import (
    ProposalFromFindingSetupError,
)


@dataclass
class PairRunResult:
    """Outcome of running one `(candidate, finding)` Claude session.

    Exactly one of `structured_output` (success) or `error` (failure) is set.
    `duration_s` is wall-clock from launch to completion or timeout.
    `stdout`/`stderr` carry the raw streams for debug-file persistence on
    failure. On success the streams are dropped — the validated payload is
    enough.
    """

    pair_key: str
    duration_s: float
    structured_output: list | None = None
    error: str | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    usage: AgentUsage | None = None


def ensure_claude_available() -> None:
    """Setup-time check: the `claude` CLI must be on PATH."""
    if shutil.which("claude") is None:
        raise ProposalFromFindingSetupError(
            "required CLI not on PATH: claude",
            executable="claude",
        )


def run_pair(
    *,
    pair_key: str,
    prompt: str,
    schema_text: str,
    repo_path: Path,
    max_turns: int,
    wallclock_s: int,
) -> PairRunResult:
    """Run one Claude session and return the parsed structured output."""
    session = ClaudeSession(
        ClaudeSessionOptions(
            json_schema=schema_text,
            permission_mode="plan",
            max_turns=max_turns,
            timeout_s=wallclock_s,
        )
    )
    result = run_with_retry(
        session,
        prompt,
        cwd=repo_path,
        on_event=None,
        label="proposal_from_finding_creator",
    )

    if result.error is not None:
        return PairRunResult(
            pair_key=pair_key,
            duration_s=result.duration_s,
            error=result.error,
            stdout=result.stdout,
            stderr=result.stderr,
            usage=result.usage,
        )

    unwrapped = _unwrap_proposals(result.structured_output)
    if unwrapped is not None:
        return PairRunResult(
            pair_key=pair_key,
            duration_s=result.duration_s,
            structured_output=unwrapped,
            usage=result.usage,
        )

    return PairRunResult(
        pair_key=pair_key,
        duration_s=result.duration_s,
        error="claude produced no structured_output and no usable fallback text",
        stdout=result.stdout,
        stderr=result.stderr,
        usage=result.usage,
    )


def _unwrap_proposals(payload: object) -> list | None:
    """Return the `proposals` array from the schema's wrapper object.

    The per-pair JSON schema (see `agent_schema.build_per_pair_schema_text`)
    is a top-level object — `{"proposals": [...]}` — because the Anthropic
    tool API requires `input_schema.type == "object"`. The rest of step 4
    works on the inner list, so unwrap here. A bare list is also accepted so a
    future schema change or alternative agent can keep working.
    """
    if isinstance(payload, dict):
        proposals = payload.get("proposals")
        if isinstance(proposals, list):
            return proposals
        return None
    if isinstance(payload, list):
        return payload
    return None


__all__ = ["PairRunResult", "ensure_claude_available", "run_pair"]
