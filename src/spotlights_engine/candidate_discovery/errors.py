"""Stage 1 candidate discovery exception hierarchy.

All three classes carry a structured `context: dict` so callers (Stage 3,
tests) can branch on cause without parsing the message string. `__str__`
renders that context alongside the message, because the manager persists
failures as `f"{type(e).__name__}: {e}"` — without it a rate-limited run is
indistinguishable from a genuinely malformed payload in `status.json`.
"""

from __future__ import annotations

from typing import Any

# Rendered last, after the cheap scalars, so the human-readable explanation of
# *why* the iteration failed is the final thing on the line.
_CONTEXT_TAIL_KEYS = ("cause",)


class _DiscoveryError(Exception):
    """Base for the three Stage-1 discovery exceptions."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self._message = message
        self.context: dict[str, Any] = dict(context)

    def __str__(self) -> str:
        parts = [
            f"{k}={v}"
            for k, v in self.context.items()
            if v is not None and k not in _CONTEXT_TAIL_KEYS
        ]
        parts += [
            f"{k}={self.context[k]}"
            for k in _CONTEXT_TAIL_KEYS
            if self.context.get(k) is not None
        ]
        if not parts:
            return self._message
        return f"{self._message} ({', '.join(parts)})"


class DiscoverySetupError(_DiscoveryError):
    """Raised before any iteration starts (bad config, missing CLI, pre-existing run dir)."""


class DiscoveryValidationError(_DiscoveryError):
    """Raised by §6.2 second-failure, §6.3, §6.7, and §6.8-empty checks."""


class DiscoveryAgentFailureError(DiscoveryValidationError):
    """The agent produced no parseable output on either attempt (§6.2).

    A subclass of `DiscoveryValidationError` so existing handlers keep working,
    but distinct from it because the cause is environmental (rate limit, timeout,
    context exhaustion) rather than a contract violation. Iterations that already
    succeeded stay valid, so the orchestrator salvages them instead of failing
    the module.
    """


class DiscoveryMutationError(_DiscoveryError):
    """Raised by §6.1 when the target repo is mutated during a subprocess call."""


__all__ = [
    "DiscoveryAgentFailureError",
    "DiscoveryMutationError",
    "DiscoverySetupError",
    "DiscoveryValidationError",
]
