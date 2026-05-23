"""Step 4 (`proposal_from_finding_creator`) exception hierarchy."""

from __future__ import annotations

from typing import Any


class ProposalFromFindingSetupError(RuntimeError):
    """Bad config (missing paths / CLI / etc.) raised before any pair runs."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


class ProposalFromFindingValidationError(ValueError):
    """Input violates the per-step contract."""

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = dict(context)


__all__ = [
    "ProposalFromFindingSetupError",
    "ProposalFromFindingValidationError",
]
