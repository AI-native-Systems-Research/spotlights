"""Thin wrapper around a centralized Codex session for the per-candidate pass.

Reuses `spotlights_engine.llm_session.CodexSession` for `--output-schema`
enforcement + spawn/parse/env-scrub/transport-retry, and maps the shared
`SessionResult` into the same `CandidateAgentRunResult` shape as the Claude
runner so step 5's orchestration stays agent-agnostic.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from spotlights_engine.agent_proposals.claude_exec import CandidateAgentRunResult
from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.llm_session import (
    CodexSession,
    CodexSessionOptions,
    run_with_retry,
)


def ensure_codex_available() -> None:
    """Setup-time check: the `codex` CLI must be on PATH."""
    if shutil.which("codex") is None:
        raise AgentProposalsSetupError(
            "required CLI not on PATH: codex",
            executable="codex",
        )


def run_candidate_codex(
    *,
    candidate_id: str,
    prompt: str,
    schema_text: str,
    repo_path: Path,
    wallclock_s: int,
    last_message_path: Path,
    schema_path: Path,
    codex_model: str | None = None,
    codex_reasoning_effort: str | None = None,
) -> CandidateAgentRunResult:
    """Run one Codex session for a single candidate.

    Writes the supplied `schema_text` to `schema_path` (under the per-module
    artifacts dir) so codex can mmap it. Reads the validated JSON from
    `last_message_path` on success.
    """
    session = CodexSession(
        CodexSessionOptions(
            repo_path=repo_path,
            output_last_message=last_message_path,
            output_schema=schema_path,
            schema_text=schema_text,
            model=codex_model,
            reasoning_effort=codex_reasoning_effort,
            sandbox="read-only",
            timeout_s=wallclock_s,
        )
    )
    result = run_with_retry(
        session,
        prompt,
        cwd=repo_path,
        on_event=None,
        label="agent_proposals.codex",
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

    return CandidateAgentRunResult(
        candidate_id=candidate_id,
        duration_s=result.duration_s,
        structured_output=result.structured_output,
        usage=result.usage,
    )


__all__ = ["ensure_codex_available", "run_candidate_codex"]
