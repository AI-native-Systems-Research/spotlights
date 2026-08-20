"""Thin subprocess wrapper around `claude -p` for the one-shot fix session.

Modeled on `agent_proposals.claude_exec` — same env scrubbing, same
`shutil.which` resolution so a Windows `.CMD` shim is found, same stream-json
usage capture for costing. Two deliberate differences:

- **No `--json-schema`.** The deliverable is edited files in the worktree, not
  a structured payload, so there is nothing to validate against a schema. A
  stream that never reached its terminal `result` event costs us the token
  numbers, not the fix: `usage` degrades to `None` and the run still succeeds.
- **`--permission-mode acceptEdits`, not `plan`.** `agent_proposals` only
  proposes; this stage implements. The blast radius is a throwaway detached
  worktree under a temp dir, never the user's repo.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError
from spotlights_engine.costing.usage import AgentUsage, claude_usage_from_stream

_DROP_EXACT = frozenset(
    {
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "ANTHROPIC_BASE_URL",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")


def _clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key in _DROP_EXACT or key.startswith(_DROP_PREFIX):
            env.pop(key)
    return env


@dataclass
class FixRunResult:
    """Outcome of one fix session.

    `error is None` means the session completed; it does **not** mean the
    worktree changed. An agent that concluded the change could not be made in
    scope exits cleanly and leaves an empty diff — a legitimate outcome the
    caller records as "no patch produced", not as a failure.
    """

    candidate_id: str
    duration_s: float
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


def run_fix_claude(
    *,
    candidate_id: str,
    prompt: str,
    worktree: Path,
    max_turns: int,
    wallclock_s: int,
) -> FixRunResult:
    """Run one `claude -p` fix session with `worktree` as the working directory."""
    # Resolve via shutil.which so Windows finds the .CMD shim. Bare
    # "claude" → FileNotFoundError because subprocess on Windows doesn't
    # follow PATHEXT for unqualified argv[0].
    claude_resolved = shutil.which("claude") or "claude"
    argv = [
        claude_resolved,
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--max-turns",
        str(max_turns),
    ]
    env = _clean_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            env=env,
            cwd=str(worktree),
            timeout=wallclock_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        stdout = exc.stdout or b""
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude timed out after {duration:.1f}s",
            stdout=stdout,
            stderr=exc.stderr or b"",
            usage=claude_usage_from_stream(stdout),
        )
    except OSError as exc:
        duration = time.monotonic() - start
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"could not launch claude: {exc}",
        )

    duration = time.monotonic() - start
    stdout = completed.stdout or b""
    stderr = completed.stderr or b""
    usage = claude_usage_from_stream(stdout)

    if completed.returncode != 0:
        stderr_tail = stderr[-500:].decode("utf-8", "replace")
        return FixRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"claude exit={completed.returncode}: stderr={stderr_tail!r}",
            stdout=stdout,
            stderr=stderr,
            usage=usage,
        )

    return FixRunResult(
        candidate_id=candidate_id,
        duration_s=duration,
        stdout=stdout,
        stderr=stderr,
        usage=usage,
    )


__all__ = [
    "FixRunResult",
    "ensure_claude_available",
    "run_fix_claude",
]
