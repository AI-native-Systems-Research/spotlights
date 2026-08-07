"""Thin wrapper around a centralized Claude session for the per-candidate pass.

Reuses `spotlights_engine.llm_session` for spawn + parse + env scrubbing +
transport retry, and maps the shared `SessionResult` into
`CandidateAgentRunResult`. Kept self-contained so step 5 does not couple to
step 4's runtime contract.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.llm_session import (
    ClaudeSession,
    ClaudeSessionOptions,
    run_with_retry,
)


@dataclass
class CandidateAgentRunResult:
    """Outcome of one per-candidate-per-agent session.

    Exactly one of `structured_output` (success) or `error` (failure) is set.
    `duration_s` is wall-clock from launch to completion or timeout.
    `stdout`/`stderr` carry the raw streams for debug-file persistence on
    failure. On success the streams are dropped — the validated payload is
    enough.
    """

    candidate_id: str
    duration_s: float
    structured_output: dict | list | None = None
    error: str | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    usage: AgentUsage | None = None


def ensure_claude_available() -> None:
    """Setup-time check: the `claude` CLI must be on PATH."""
    if shutil.which("claude") is None:
        raise AgentProposalsSetupError(
            "required CLI not on PATH: claude",
            executable="claude",
        )


def run_candidate_claude(
    *,
    candidate_id: str,
    prompt: str,
    schema_text: str,
    repo_path: Path,
    max_turns: int,
    wallclock_s: int,
) -> CandidateAgentRunResult:
    """Run one Claude session for a single candidate."""
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
        label="agent_proposals.claude",
    )

    if result.error is not None:
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=result.duration_s,
            error=result.error,
            stdout=result.stdout,
            stderr=result.stderr,
            usage=result.usage,
        )

    structured = result.structured_output
    if isinstance(structured, (dict, list)):
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=result.duration_s,
            structured_output=structured,
            usage=result.usage,
        )

    return CandidateAgentRunResult(
        candidate_id=candidate_id,
        duration_s=result.duration_s,
        error="claude produced no structured_output and no usable fallback text",
        stdout=result.stdout,
        stderr=result.stderr,
        usage=result.usage,
    )


__all__ = [
    "CandidateAgentRunResult",
    "ensure_claude_available",
    "run_candidate_claude",
]
