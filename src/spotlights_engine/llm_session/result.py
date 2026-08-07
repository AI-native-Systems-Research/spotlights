"""The single result type returned by every session, plus its errors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from spotlights_engine.costing.usage import AgentUsage


class LLMTransientError(RuntimeError):
    """A retryable transport failure (rate-limit / overloaded / 429 / 529 …)."""


class LLMSessionTimeout(RuntimeError):
    """A wall-clock timeout of the CLI process (its own, smaller retry budget)."""


@dataclass
class SessionResult:
    """Outcome of one Claude/Codex session invocation.

    On success `structured_output` and/or `final_message` are populated and
    `error is None`. On failure `error` carries a human-readable message; the
    raw `stdout`/`stderr` are retained for post-mortem persistence.
    """

    cli: Literal["claude", "codex"]
    returncode: int
    duration_s: float
    structured_output: dict | list | None = None
    final_message: str | None = None
    error: str | None = None
    usage: AgentUsage | None = None
    model: str | None = None
    cost_usd: float | None = None
    num_turns: int | None = None
    session_id: str | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    # True when the failure was a wall-clock timeout (drives the timeout retry
    # budget in the classifier). Never set on success.
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.returncode == 0

    def raise_for_status(self) -> None:
        """Raise if the session did not succeed (for callers that prefer raising)."""
        if self.ok:
            return
        if self.timed_out:
            raise LLMSessionTimeout(self.error or f"{self.cli} timed out")
        raise RuntimeError(self.error or f"{self.cli} exit={self.returncode}")


__all__ = [
    "LLMSessionTimeout",
    "LLMTransientError",
    "SessionResult",
]
