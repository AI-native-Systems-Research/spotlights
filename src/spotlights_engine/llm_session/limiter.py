"""Process-wide concurrency limiter for live Claude/Codex CLI processes.

The bug this fixes: today each fan-out layer bounds its own concurrency
independently (`asyncio.Semaphore` in some steps, `ThreadPoolExecutor` in
others) and the manager runs modules in parallel on top of that — so the true
number of simultaneous CLI processes is the *product* of the per-layer limits,
which is what produces "too many concurrent requests" errors.

`GlobalCLILimiter` is one process-wide cap. A single
`threading.BoundedSemaphore` is the source of truth — the only primitive safe
to acquire from BOTH plain threads (the ThreadPoolExecutor fan-outs) and from
asyncio worker threads. We deliberately do NOT use an `asyncio.Semaphore` for
the async path: that would be a *second, independent* counter and silently
double the effective cap.

Per the design's recommended (simpler) variant, `session.run` is always
blocking and always ends up on a worker thread, so the async path acquires the
slot **inside that same worker** via `sync_slot`. One thread submission does
acquire→run→release, so there is no cross-executor dependency (the
executor-exhaustion deadlock class disappears) and cancellation is leak-free by
construction: cancelling the `to_thread` future does not stop the running
worker, which still hits its `finally` and releases.
"""

from __future__ import annotations

import contextlib
import os
import threading


class GlobalCLILimiter:
    """Bounds simultaneous Claude/Codex processes across ALL modules/steps."""

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        self._sem = threading.BoundedSemaphore(max_concurrency)

    @contextlib.contextmanager
    def sync_slot(self):
        """Acquire one slot for the duration of the `with` body.

        Safe for ThreadPool / `to_thread` callers, and for async callers when
        acquired inside the worker thread that runs `session.run`.
        """
        self._sem.acquire()
        try:
            yield
        finally:
            self._sem.release()


# The process-wide limiter is a lazily-initialized singleton behind an accessor,
# NOT a module-level constant constructed at import. A BoundedSemaphore cannot be
# resized, and the manager sets the budget from its config at startup (not known
# at import time) — so construction is deferred.
_GLOBAL: GlobalCLILimiter | None = None
_GLOBAL_LOCK = threading.Lock()

_DEFAULT_MAX_CONCURRENCY = 6
_ENV_VAR = "SPOTLIGHTS_MAX_CLI_CONCURRENCY"


def _default_from_env() -> int:
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        return _DEFAULT_MAX_CONCURRENCY
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_MAX_CONCURRENCY
    return value if value >= 1 else _DEFAULT_MAX_CONCURRENCY


def configure_global_limiter(max_concurrency: int) -> GlobalCLILimiter:
    """Set the process-wide budget once, before any slot is taken.

    Raises if the limiter already exists (fail loud rather than silently
    ignore a late reconfigure). Callers needing a different budget construct
    their own `GlobalCLILimiter` and pass it as `limiter=`.
    """
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            raise RuntimeError("global CLI limiter already initialized")
        _GLOBAL = GlobalCLILimiter(max_concurrency)
        return _GLOBAL


def get_global_limiter() -> GlobalCLILimiter:
    """Return the process-wide limiter, lazily creating it from env/default."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            _GLOBAL = GlobalCLILimiter(_default_from_env())
        return _GLOBAL


def reset_global_limiter_for_tests() -> None:
    """Test-only: drop the singleton so a fresh budget can be configured."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        _GLOBAL = None


__all__ = [
    "GlobalCLILimiter",
    "configure_global_limiter",
    "get_global_limiter",
    "reset_global_limiter_for_tests",
]
