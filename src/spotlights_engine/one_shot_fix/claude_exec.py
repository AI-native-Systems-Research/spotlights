"""Thin subprocess wrapper around `claude -p` for the one-shot fix session.

Modeled on `agent_proposals.claude_exec` — prompt on stdin, `shutil.which`
resolution so a Windows `.CMD` shim is found, stream-json usage capture for
costing. Four deliberate differences:

- **`ANTHROPIC_AUTH_TOKEN` is scrubbed** in addition to the env vars
  `agent_proposals` drops (see `_DROP_EXACT`).

- **No `--json-schema`.** The deliverable is edited files in the worktree, not
  a structured payload, so there is nothing to validate against a schema. A
  stream that never reached its terminal `result` event costs us the token
  numbers, not the fix: `usage` degrades to `None` and the run still succeeds.
- **`--permission-mode acceptEdits`, not `plan`.** `agent_proposals` only
  proposes; this stage implements. The blast radius is a throwaway detached
  worktree under a temp dir, never the user's repo.
- **`--disallowedTools` denies git write commands.** File edits are contained
  by `cwd` (the throwaway worktree), but git is not: a worktree checkout
  shares the main repository's object database and refs namespace (and, for
  `git worktree`, its administrative metadata), so writes issued from inside
  the worktree can land in the main repo and outlive
  `git worktree remove --force` + `git worktree prune` — `git stash` puts an
  entry in the main repo's `refs/stash`, `git branch`/`git tag` create refs
  there, `git push` reaches a remote entirely outside the worktree, and
  `git worktree` itself edits the shared worktree-admin state. `git commit`
  and `git checkout` are milder (worktree-local) but still undermine the
  contract another way: they move the agent's work out of the working tree,
  so `collect_patch`'s `git diff` sees nothing and `FIX-NOTES.md` falsely
  reports no patch was produced. The prompt in `prompts.py` already asks the
  agent not to do this; `--disallowedTools` is what actually enforces it,
  since whether the prose is even reachable depends on the invoking user's
  Bash allowlist — including a `.claude/settings.json` the target repo may
  ship, which is present inside the worktree because the worktree is a
  checkout of that repo. `git reset` is deliberately NOT denied: on a
  detached worktree it only rewrites that worktree's own, private HEAD/index
  (cleaned up by `git worktree remove`), so it cannot escape — it can only
  destroy the agent's own uncommitted edits, a quality risk, not a blast-
  radius one.
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
        # Dropped for the reason `signal_pipeline/claude_subprocess.py` records:
        # inside a Claude Code session this token is set in the environment, and
        # inherited by a spawned `claude` it overrides the keychain credentials
        # and the session fails with `401 Invalid bearer token`. `fix` is
        # *designed* to be launched from such a session (that is what
        # `/spotlights-fix-candidate` does), so this is the module where the
        # leak is most likely, not least.
        "ANTHROPIC_AUTH_TOKEN",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "VIRTUAL_ENV",
    }
)
_DROP_PREFIX = ("VSCODE_", "OPTQUEST_", "SPOTLIGHTS_")

# Git write commands whose effects can escape the throwaway worktree: they
# either land in the main repository's shared refs/admin state (surviving
# `git worktree remove --force` + `git worktree prune`) or move the agent's
# work out of the working tree where `collect_patch` can no longer see it.
# See the module docstring for the per-command reasoning, including why
# `git reset` is deliberately absent.
_DISALLOWED_GIT_WRITES = (
    "Bash(git commit:*)",
    "Bash(git stash:*)",
    "Bash(git branch:*)",
    "Bash(git checkout:*)",
    "Bash(git push:*)",
    "Bash(git tag:*)",
    "Bash(git worktree:*)",
)


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
    # The prompt goes on **stdin**, not argv — same as
    # `agent_proposals.claude_exec`. A fix prompt embeds the whole findings
    # digest (candidate spec, code excerpt, scope list, oracles) and is
    # unbounded in principle. Linux caps a single argv element at
    # `MAX_ARG_STRLEN` = 128 KiB regardless of `ARG_MAX`, and macOS caps env +
    # argv together at 1 MiB (measured: `execve` of a ~1 MiB single argument
    # fails with `OSError: [Errno 7] Argument list too long`). On argv, a large
    # candidate therefore fails at process launch — surfacing as
    # `could not launch claude: ...` after a worktree was already created and
    # validated, with nothing about the message pointing at prompt size. Stdin
    # has no such limit.
    argv = [
        claude_resolved,
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--permission-mode",
        "acceptEdits",
        "--disallowedTools",
        ",".join(_DISALLOWED_GIT_WRITES),
        "--max-turns",
        str(max_turns),
    ]
    env = _clean_env()
    start = time.monotonic()
    try:
        completed = subprocess.run(
            argv,
            input=prompt.encode("utf-8"),
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
