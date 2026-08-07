"""Dedicated, human-readable retry log for post-run diagnosis.

Every retry attempt is appended here with the reason (rate-limit /
concurrent-request / timeout / other), the attempt number, the backoff delay
chosen, and which session/module triggered it — so the "too many concurrent
requests" problem can be diagnosed after a run.

Where it lands: the manager configures the log into the run's artifacts folder,
following the existing convention — `ManagerPaths(artifacts_dir).root` =
`<artifacts_dir>/spotlights_manager/retries.log`. When no artifacts folder is
configured/available, the logger degrades gracefully to standard logging
(`logging.getLogger("spotlights_engine.llm_session.retry")`) rather than
crashing.

The line format is greppable and fixed:

    <iso-timestamp> retry cli=<claude|codex> label=<module/step> attempt=<n>/<budget> \
        reason=<transient|timeout|none> delay=<seconds>s error=<snippet>
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

_log = logging.getLogger("spotlights_engine.llm_session.retry")


def _now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _snippet(text: str | None, limit: int = 300) -> str:
    if not text:
        return ""
    flat = " ".join(text.split())
    if len(flat) > limit:
        flat = flat[:limit] + "…"
    return flat


class RetryLogger:
    """Appends one line per retry attempt to a file, with fallback logging.

    Robust by design: a file-write failure never propagates to the retry loop
    (it falls back to the module logger), and if no path was configured every
    record goes to standard logging.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._lock = threading.Lock()
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:  # pragma: no cover - defensive
                _log.warning("retry log dir unusable (%s); falling back to logging", exc)
                self._path = None

    @property
    def path(self) -> Path | None:
        return self._path

    def record(
        self,
        *,
        cli: str,
        label: str,
        attempt: int,
        budget: int,
        reason: str,
        delay_s: float,
        error: str | None,
    ) -> None:
        """Append one retry-attempt line (or emit via standard logging)."""
        line = (
            f"{_now_iso()} retry cli={cli} label={label} "
            f"attempt={attempt}/{budget} reason={reason} "
            f"delay={delay_s:.2f}s error={_snippet(error)}"
        )
        if self._path is None:
            _log.warning("%s", line)
            return
        try:
            with self._lock, self._path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:  # pragma: no cover - defensive
            _log.warning("retry log write failed (%s); %s", exc, line)


# Process-wide accessor, mirroring the limiter singleton. Defaults to a
# fallback-logging instance until the manager configures a path.
_RETRY_LOGGER = RetryLogger(None)
_RETRY_LOCK = threading.Lock()


def configure_retry_log(path: Path) -> RetryLogger:
    """Point the process-wide retry logger at `path` (idempotent last-wins)."""
    global _RETRY_LOGGER
    with _RETRY_LOCK:
        _RETRY_LOGGER = RetryLogger(path)
        return _RETRY_LOGGER


def get_retry_logger() -> RetryLogger:
    with _RETRY_LOCK:
        return _RETRY_LOGGER


def reset_retry_log_for_tests() -> None:
    """Test-only: restore the fallback-logging instance."""
    global _RETRY_LOGGER
    with _RETRY_LOCK:
        _RETRY_LOGGER = RetryLogger(None)


__all__ = [
    "RetryLogger",
    "configure_retry_log",
    "get_retry_logger",
    "reset_retry_log_for_tests",
]
