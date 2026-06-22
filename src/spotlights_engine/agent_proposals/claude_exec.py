"""Thin subprocess wrapper around `claude -p` for the per-candidate Claude pass.

Mirrors `proposal_from_finding_creator.claude_exec` but parameterised on the
per-candidate schema and returns a `CandidateAgentRunResult`. Kept self-
contained so step 5 does not couple to step 4's runtime contract.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.claude_env import build_claude_env, claude_model_args


def _clean_env() -> dict[str, str]:
    return build_claude_env()


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
    # Resolve via shutil.which so Windows finds the .CMD shim. Bare
    # "claude" → FileNotFoundError because subprocess on Windows doesn't
    # follow PATHEXT for unqualified argv[0].
    claude_resolved = shutil.which("claude") or "claude"
    argv = [
        claude_resolved,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--json-schema",
        schema_text,
        "--permission-mode",
        "plan",
        "--max-turns",
        str(max_turns),
        *claude_model_args(),
    ]
    env = _clean_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            input=prompt.encode("utf-8"),
            capture_output=True,
            env=env,
            cwd=str(repo_path),
            timeout=wallclock_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude timed out after {duration:.1f}s",
            stdout=exc.stdout or b"",
            stderr=exc.stderr or b"",
        )
    duration = time.monotonic() - start

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude exit={completed.returncode}: stderr={stderr_tail!r}",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    try:
        result_event = _extract_result_event(completed.stdout)
    except _ResultEventError as exc:
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=str(exc),
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    if result_event is None:
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error="claude stream-json had no terminal result event",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    structured = result_event.get("structured_output")
    if isinstance(structured, dict | list):
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            structured_output=structured,
        )

    # Fallback: older CLI versions may put the JSON-encoded payload on `result`.
    fallback = result_event.get("result")
    if isinstance(fallback, str) and fallback.strip():
        try:
            parsed = json.loads(fallback)
        except json.JSONDecodeError as exc:
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=duration,
                error=f"claude result text not JSON: {exc}",
                stdout=completed.stdout or b"",
                stderr=completed.stderr or b"",
            )
        if isinstance(parsed, dict | list):
            return CandidateAgentRunResult(
                candidate_id=candidate_id,
                duration_s=duration,
                structured_output=parsed,
            )

    return CandidateAgentRunResult(
        candidate_id=candidate_id,
        duration_s=duration,
        error="claude produced no structured_output and no usable fallback text",
        stdout=completed.stdout or b"",
        stderr=completed.stderr or b"",
    )


class _ResultEventError(Exception):
    pass


def _extract_result_event(stdout: bytes) -> dict | None:
    last_event: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise _ResultEventError(
                f"claude stream-json line not JSON: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            raise _ResultEventError("claude stream-json line was not an object")
        last_event = obj
    if last_event is None or last_event.get("type") != "result":
        return None
    return last_event


__all__ = [
    "CandidateAgentRunResult",
    "ensure_claude_available",
    "run_candidate_claude",
]
