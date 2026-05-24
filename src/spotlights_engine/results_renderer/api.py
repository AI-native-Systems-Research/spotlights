"""Public types and entrypoint for the results renderer."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RendererInput(BaseModel):
    """Where to read from and where to write to.

    `artifacts_dir` is the same value passed to `SpotlightsManagerConfig`. The
    renderer reads `<artifacts_dir>/spotlights_manager/`. `output_folder` is
    where `index.md` and `modules/*.md` are written.
    """

    model_config = ConfigDict(extra="forbid")

    artifacts_dir: Path
    output_folder: Path


class RendererConfig(BaseModel):
    """Renderer-side knobs."""

    model_config = ConfigDict(extra="forbid")

    overwrite: bool = True
    include_failed_modules: bool = True
    include_skipped_modules: bool = True
    relative_repo_links: bool = True


class RendererResult(BaseModel):
    """Summary of what the renderer wrote."""

    model_config = ConfigDict(extra="forbid")

    index_path: Path
    module_pages: dict[str, Path] = Field(default_factory=dict)
    skipped_modules: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def render(
    input: RendererInput, *, config: RendererConfig | None = None
) -> RendererResult:
    """Render a `spotlights_manager` checkpoint tree into Markdown.

    Reads `<artifacts_dir>/spotlights_manager/` and writes `index.md` plus
    `modules/<slug>.md` under `output_folder`.
    """
    # Late import keeps `api` import cheap and avoids any chance of an import
    # cycle via `spotlights_manager.api` (which also imports `RendererResult`).
    from spotlights_engine.results_renderer.loader import load_run
    from spotlights_engine.results_renderer.writer import emit

    cfg = config or RendererConfig()
    loaded = load_run(input.artifacts_dir)
    result = emit(input=input, loaded=loaded, config=cfg)
    return result


__all__ = [
    "RendererConfig",
    "RendererInput",
    "RendererResult",
    "render",
]
