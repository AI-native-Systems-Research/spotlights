"""Step 5 (`agent_proposals`) exception hierarchy."""

from __future__ import annotations

from typing import Any


class AgentProposalsSetupError(RuntimeError):
    """Bad config (missing paths / CLI / etc.) raised before any candidate runs."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


class AgentProposalsValidationError(ValueError):
    """Input violates the per-step contract."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


__all__ = [
    "AgentProposalsSetupError",
    "AgentProposalsValidationError",
]
