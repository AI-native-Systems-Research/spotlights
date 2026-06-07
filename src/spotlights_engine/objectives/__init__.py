"""Bundle G: objective setting. Captures PM intent for a discovery session."""

from spotlights_engine.objectives.core import (
    assemble_proposal,
    build_intent,
    finalize_objective,
)
from spotlights_engine.objectives.schemas import (
    Objective,
    ObjectiveIntent,
    ObjectiveProposal,
)

__all__ = [
    "Objective",
    "ObjectiveIntent",
    "ObjectiveProposal",
    "assemble_proposal",
    "build_intent",
    "finalize_objective",
]
