"""Self-contained subprocess wrapper around `codex exec` for the per-candidate
Codex pass.

Modelled on `candidate_discovery.CodexRunner` because we want
`--output-schema` enforcement, which `module_deep_research.CodexExecClient`
does not expose. Returns the same `CandidateAgentRunResult` shape as the
Claude runner so step 5's orchestration can stay agent-agnostic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from spotlights_engine.agent_proposals.claude_exec import CandidateAgentRunResult
from spotlights_engine.agent_proposals.errors import AgentProposalsSetupError


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
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.write_text(schema_text, encoding="utf-8")
    last_message_path.parent.mkdir(parents=True, exist_ok=True)
    if last_message_path.exists():
        try:
            last_message_path.unlink()
        except OSError:
            pass

    argv: list[str] = [
        "codex",
        "exec",
        "-",
        "--json",
        "--output-last-message",
        str(last_message_path),
        "--output-schema",
        str(schema_path),
        "--sandbox",
        "read-only",
        "-C",
        str(repo_path),
    ]
    if codex_model is not None:
        argv += ["-c", f'model="{codex_model}"']
    if codex_reasoning_effort is not None:
        argv += ["-c", f'model_reasoning_effort="{codex_reasoning_effort}"']

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
            error=f"codex timed out after {duration:.1f}s",
            stdout=exc.stdout or b"",
            stderr=exc.stderr or b"",
        )
    duration = time.monotonic() - start

    if completed.returncode != 0:
        stderr_tail = (completed.stderr or b"")[-500:].decode("utf-8", "replace")
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"codex exit={completed.returncode}: stderr={stderr_tail!r}",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    if not last_message_path.exists():
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error="codex output_last_message file missing",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )
    text = last_message_path.read_text(encoding="utf-8")
    if not text.strip():
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error="codex output_last_message empty",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=f"codex output_last_message not JSON: {exc}",
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )
    if not isinstance(parsed, (dict, list)):
        return CandidateAgentRunResult(
            candidate_id=candidate_id,
            duration_s=duration,
            error=(
                "codex output_last_message JSON was not an object or array: "
                f"got {type(parsed).__name__}"
            ),
            stdout=completed.stdout or b"",
            stderr=completed.stderr or b"",
        )

    return CandidateAgentRunResult(
        candidate_id=candidate_id,
        duration_s=duration,
        structured_output=parsed,
    )


__all__ = ["ensure_codex_available", "run_candidate_codex"]
