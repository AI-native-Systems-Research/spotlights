"""Shared types for stage modules.

Kept in a leaf module so individual stages don't need to import from
`runner` (which depends on the stage registry, creating a cycle).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from spotlights_engine.signal_pipeline.layout import RunDirLayout, StageId, StageShape

if TYPE_CHECKING:
    from spotlights_engine.signal_pipeline.schemas import SignalPipelineInput


@dataclass
class StageContext:
    """Runtime context handed to a stage's `run` / `run_one` callable.

    Stages read their inputs from `upstream` (already parsed by the runner
    via each upstream stage's `parse_artifact` / `parse_item`), and return
    their output. The runner persists outputs atomically — stages should
    not write to `layout` directly.

    `on_event`, when set, is the per-stage formatter the runner installed
    for live progress lines. Stages forward it to their `claude -p`
    helpers so subprocess events get a `[NN]` prefix and burst-dedupe;
    pass `None` to stay silent.
    """

    stage_id: StageId
    layout: RunDirLayout
    signal_input: "SignalPipelineInput"
    upstream: dict[StageId, Any]
    log_dir: Path
    on_event: Callable[[str], None] | None = None


# Callable types for stage entry points.
SingleRun = Callable[[StageContext], Any]
"""For single-artifact stages — returns the payload to be persisted."""

FanoutRunOne = Callable[[StageContext, str], Any]
"""For fan-out stages — called once per id; returns the per-id payload."""

IdsFromUpstream = Callable[[Any], list[str]]
"""Maps the upstream stage's parsed payload to the ordered id list this
fan-out stage should cover. Called by the runner; not by the stage."""

ParseArtifact = Callable[[Any], Any]
"""Map raw JSON (from `json.load`) to the typed payload. Each stage owns
its own deserializer to keep schema drift local."""


@dataclass(frozen=True)
class StageSpec:
    """Static description of a pipeline stage."""

    stage_id: StageId
    name: str
    shape: StageShape
    upstream: tuple[StageId, ...]
    # Deserializer for the stage's own artifact (single → file content;
    # fanout → unused, use `parse_item` per entry instead).
    parse_artifact: ParseArtifact

    # Single-artifact stages
    run: SingleRun | None = None

    # Fan-out stages
    parse_item: ParseArtifact | None = None
    upstream_for_hash: StageId | None = None
    upstream_hash_field: str | None = None  # e.g. "upstream_candidates_hash"
    ids_from_upstream: IdsFromUpstream | None = None
    run_one: FanoutRunOne | None = None


__all__ = [
    "FanoutRunOne",
    "IdsFromUpstream",
    "ParseArtifact",
    "SingleRun",
    "StageContext",
    "StageSpec",
]
