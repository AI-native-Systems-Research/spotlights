"""Retry + backoff + limiter wrapper — the core new behaviour.

Every CLI process is born inside `run_with_retry` / `arun_with_retry`, which:

- acquires the process-wide `GlobalCLILimiter` slot at the leaf (never held
  across a fan-out or across the backoff sleep),
- runs the (blocking) `session.run` — on a worker thread in the async path,
- classifies the outcome, and on a retryable transient/timeout failure sleeps
  a jittered exponential backoff and retries, up to the reason's budget,
- appends every retry attempt to the dedicated retry log for post-run
  diagnosis.

Write-mode sessions (`permission_mode` in {acceptEdits, bypassPermissions})
default to a single attempt so a partially-applied edit is never re-applied.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from spotlights_engine.llm_session.backoff import (
    RetryReason,
    budget_for,
    classify,
    next_delay,
)
from spotlights_engine.llm_session.config import (
    RetryPolicy,
    get_default_retry_policy,
    write_mode_policy,
)
from spotlights_engine.llm_session.limiter import GlobalCLILimiter, get_global_limiter
from spotlights_engine.llm_session.result import SessionResult
from spotlights_engine.llm_session.retry_log import get_retry_logger

_WRITE_MODES = frozenset({"acceptEdits", "bypassPermissions"})


def _effective_policy(session, policy: RetryPolicy | None) -> RetryPolicy:
    """Explicit policy wins; else derive from the process-wide default.

    Write-mode sessions collapse to a single attempt (never re-apply a
    partially-applied edit) while preserving the configured delays/jitter.
    """
    if policy is not None:
        return policy
    mode = getattr(getattr(session, "options", None), "permission_mode", "plan")
    if mode in _WRITE_MODES:
        return write_mode_policy()
    return get_default_retry_policy()


def _label_for(session, label: str | None) -> str:
    if label:
        return label
    return getattr(session, "name", type(session).__name__)


def _log_retry(
    *, session, label: str | None, attempt: int, budget: int,
    reason: RetryReason, delay_s: float, result: SessionResult,
) -> None:
    get_retry_logger().record(
        cli=getattr(session, "cli", getattr(session, "name", "?")),
        label=_label_for(session, label),
        attempt=attempt,
        budget=budget,
        reason=reason.value,
        delay_s=delay_s,
        error=result.error,
    )


def run_with_retry(
    session,
    prompt: str,
    *,
    cwd: Path | str | None,
    on_event: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    policy: RetryPolicy | None = None,
    limiter: GlobalCLILimiter | None = None,
    label: str | None = None,
) -> SessionResult:
    """Synchronous retry wrapper (ThreadPool / `to_thread` callers)."""
    policy = _effective_policy(session, policy)
    limiter = limiter or get_global_limiter()
    last: SessionResult | None = None
    attempt = 0
    while True:
        with limiter.sync_slot():
            res = _invoke(session, prompt, cwd=cwd, on_event=on_event, log_dir=log_dir)
        reason = classify(res)
        if res.ok or reason is RetryReason.NONE:
            return res
        last = res
        attempt += 1
        budget = budget_for(reason, policy)
        if attempt >= budget:
            return last
        delay = next_delay(attempt - 1, policy)
        _log_retry(
            session=session, label=label, attempt=attempt, budget=budget,
            reason=reason, delay_s=delay, result=res,
        )
        time.sleep(delay)


async def arun_with_retry(
    session,
    prompt: str,
    *,
    cwd: Path | str | None,
    on_event: Callable[[str], None] | None = None,
    log_dir: Path | None = None,
    policy: RetryPolicy | None = None,
    limiter: GlobalCLILimiter | None = None,
    label: str | None = None,
) -> SessionResult:
    """Async retry wrapper. Acquires the slot INSIDE the worker thread (the
    recommended, leak-free variant: one submission does acquire→run→release)."""
    policy = _effective_policy(session, policy)
    limiter = limiter or get_global_limiter()
    last: SessionResult | None = None
    attempt = 0
    while True:
        res = await asyncio.to_thread(
            _run_in_slot, limiter, session, prompt, cwd, on_event, log_dir
        )
        reason = classify(res)
        if res.ok or reason is RetryReason.NONE:
            return res
        last = res
        attempt += 1
        budget = budget_for(reason, policy)
        if attempt >= budget:
            return last
        delay = next_delay(attempt - 1, policy)
        _log_retry(
            session=session, label=label, attempt=attempt, budget=budget,
            reason=reason, delay_s=delay, result=res,
        )
        await asyncio.sleep(delay)


def _run_in_slot(limiter, session, prompt, cwd, on_event, log_dir) -> SessionResult:
    with limiter.sync_slot():
        return _invoke(session, prompt, cwd=cwd, on_event=on_event, log_dir=log_dir)


def _invoke(session, prompt, *, cwd, on_event, log_dir) -> SessionResult:
    """Call `session.run`. Both ClaudeSession and CodexSession accept
    `on_event` and `log_dir` (Codex ignores them), so one uniform call."""
    return session.run(prompt, cwd=cwd, on_event=on_event, log_dir=log_dir)


__all__ = ["arun_with_retry", "run_with_retry"]
