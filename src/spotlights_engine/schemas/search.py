"""Search-query log entities produced by the per-module deep-research step.

These types capture, for observability, the web/literature search queries each
research agent issued and the results those queries returned. They are
**advisory** — unlike `Finding` (load-bearing, deliberately strict), search
logs must never be able to fail validation, or a stray extra field / empty
query in the debug payload would discard a runner's real findings. Hence both
types are lenient: `extra="ignore"` and **no `min_length`** on any field.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    """One result a search query returned (advisory, lenient)."""

    model_config = ConfigDict(extra="ignore")

    title: str = ""
    url: str = ""
    snippet: str = ""


class SearchQueryLog(BaseModel):
    """One search query an agent issued, tagged with the runner that issued it.

    `agent` is set from `RunnerOutcome.agent_name` during the merge, so it is
    non-empty in practice — but the schema must not depend on it (see the
    strictness rule in the debug design doc)."""

    model_config = ConfigDict(extra="ignore")

    agent: str = ""  # "codex" | "claude"; NO min_length
    query: str = ""  # NO min_length — empty must not fail the payload
    tool: str = ""
    results: list[SearchResult] = Field(default_factory=list)
    # Set only in candidate deep-research mode, where one module's log
    # concatenates several per-candidate surveys. `None` in module mode.
    # `extra="ignore"` drops undeclared keys, so the field must be declared
    # here to survive a round-trip.
    candidate_id: str | None = None


__all__ = [
    "SearchQueryLog",
    "SearchResult",
]
