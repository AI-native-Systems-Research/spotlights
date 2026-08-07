"""Shared subprocess result and runner protocol for module deep research."""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from spotlights_engine.costing.usage import AgentUsage
from spotlights_engine.llm_session.backoff import (
    RetryReason,
    budget_for,
    classify_transport,
    next_delay,
)
from spotlights_engine.llm_session.config import RetryPolicy, get_default_retry_policy
from spotlights_engine.llm_session.limiter import get_global_limiter
from spotlights_engine.llm_session.retry_log import get_retry_logger

WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION = os.name == "nt"


class AgentExecResult(BaseModel):
    """Captured result from one non-interactive research-agent invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    final_message: str | None = None
    # Token usage parsed from the CLI output while the raw stream is still
    # available; None when the runner emitted no parseable usage (e.g. a
    # timeout before any usage event was written).
    usage: AgentUsage | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self) -> None:
        if self.returncode != 0:
            raise RuntimeError(
                f"agent failed with exit code {self.returncode}\n"
                f"COMMAND: {' '.join(self.command)}\n"
                f"STDOUT:\n{self.stdout}\nSTDERR:\n{self.stderr}"
            )


class ModuleResearchRunner(Protocol):
    """Minimal interface shared by Codex, Claude, and test fakes."""

    name: str

    def run(self, prompt: str, *, check: bool = True) -> AgentExecResult: ...


def guarded_run(
    inner: Callable[[], AgentExecResult],
    *,
    cli: str,
    label: str,
    policy: RetryPolicy | None = None,
) -> AgentExecResult:
    """Run one CLI invocation inside the global limiter + transport retry.

    `inner` performs the actual spawn and returns an `AgentExecResult` (it may
    raise `subprocess.TimeoutExpired` on a wall-clock kill). This wraps it in
    the process-wide `GlobalCLILimiter` slot and the shared exponential-backoff
    classifier so module deep research shares the same "too many concurrent
    requests" handling as every other session. Each retry is appended to the
    dedicated retry log.

    Retries only on transient (rate-limit / overloaded / 429 / 529 / 503) exits
    and on timeouts (smaller budget); every other nonzero exit is returned as-is
    so the orchestrator's existing recoverable-issue handling still fires.
    """
    import subprocess

    policy = policy or get_default_retry_policy()
    limiter = get_global_limiter()
    attempt = 0
    while True:
        try:
            with limiter.sync_slot():
                result = inner()
        except subprocess.TimeoutExpired as exc:
            reason = RetryReason.TIMEOUT
            attempt += 1
            budget = budget_for(reason, policy)
            if attempt >= budget:
                raise
            delay = next_delay(attempt - 1, policy)
            get_retry_logger().record(
                cli=cli, label=label, attempt=attempt, budget=budget,
                reason=reason.value, delay_s=delay, error=str(exc),
            )
            time.sleep(delay)
            continue

        haystack = (result.stderr or "") + "\n" + (result.stdout or "")
        reason = classify_transport(ok=result.ok, timed_out=False, text=haystack)
        if reason is RetryReason.NONE:
            return result
        attempt += 1
        budget = budget_for(reason, policy)
        if attempt >= budget:
            return result
        delay = next_delay(attempt - 1, policy)
        get_retry_logger().record(
            cli=cli, label=label, attempt=attempt, budget=budget,
            reason=reason.value, delay_s=delay,
            error=f"exit={result.returncode}: {haystack[-300:]}",
        )
        time.sleep(delay)


def resolve_cli_executable(executable: str) -> str:
    """Resolve CLI shims on Windows while preserving POSIX command behavior.

    npm-installed CLIs are commonly exposed as `.cmd` shims on Windows. Passing a
    bare command name to `subprocess.run(..., shell=False)` can fail with
    `FileNotFoundError` there, even when the command works in `cmd.exe` or
    PowerShell. `shutil.which` applies Windows `PATHEXT` lookup and returns the
    concrete shim path. POSIX callers keep the configured command unchanged so
    command shapes remain stable in logs and tests.
    """
    if not WINDOWS_SUBPROCESS_NEEDS_SHIM_RESOLUTION:
        return executable
    return shutil.which(executable) or executable


__all__ = [
    "AgentExecResult",
    "ModuleResearchRunner",
    "guarded_run",
    "resolve_cli_executable",
]
