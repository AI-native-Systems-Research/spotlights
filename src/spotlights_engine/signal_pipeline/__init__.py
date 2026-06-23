"""Signal-based discovery pipeline — MVP runner + stages.

See `docs/signal-based/` for the design (flow doc, mvp module APIs, human
overview).

Entry points:
- `run_pipeline(...)` — programmatic
- `spotlights-engine signal` — top-level CLI (recommended; goes through
  the unified runner, emits a `SpotlightReport` with `pipeline="unified"`)
- `signal-pipeline` — standalone CLI (lower-level controls:
  `--from-stage` / `--to-stage` / `--inject`, plus opt-in stage 05)

Public surface kept small. Stage implementations live in
`spotlights_engine.signal_pipeline.stages.*` and are *not* re-exported here
so callers don't accidentally bypass the runner's resume/inject logic.
"""

from __future__ import annotations

from spotlights_engine.signal_pipeline.runner import (
    InjectSpec,
    SignalPipelineInput,
    SignalPipelineResult,
    StageId,
    StageSelection,
    run_pipeline,
)

#: Project-wide default model for stages whose `SPEC.model` is None and
#: whose run didn't pass `--model` / `SignalPipelineInput.model`. None
#: means "let `claude -p` pick its own default" (currently Opus).
#:
#: Precedence (highest first):
#:   1. `StageSpec.model` — pinned per-stage in `stages/sNN.py`
#:   2. `SignalPipelineInput.model` — set by `--model <id>` on the CLI
#:   3. `DEFAULT_MODEL` — this constant
#:   4. `claude -p` default
#:
#: To set the project default to e.g. Sonnet 4.6, change this to
#: `"claude-sonnet-4-6"`. The runner reads the value via attribute
#: lookup at stage-execution time, so monkeypatching this in tests
#: behaves as expected.
DEFAULT_MODEL: str | None = None

__all__ = [
    "DEFAULT_MODEL",
    "InjectSpec",
    "SignalPipelineInput",
    "SignalPipelineResult",
    "StageId",
    "StageSelection",
    "run_pipeline",
]
