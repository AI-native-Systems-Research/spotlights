"""Results renderer — renders the SpotlightsManager checkpoint tree as Markdown.

Public surface:

- `render(input, *, config)` — read `<artifacts_dir>/spotlights_manager/`
  and write `index.md` plus `modules/<slug>.md` under `output_folder`.
- `RendererInput`, `RendererConfig`, `RendererResult` — public types.
- `RendererSetupError`, `RendererLoadError` — error types.
"""

from __future__ import annotations

from spotlights_engine.results_renderer.api import (
    RendererConfig,
    RendererInput,
    RendererResult,
    render,
)
from spotlights_engine.results_renderer.errors import (
    RendererLoadError,
    RendererSetupError,
)

__all__ = [
    "RendererConfig",
    "RendererInput",
    "RendererLoadError",
    "RendererResult",
    "RendererSetupError",
    "render",
]
