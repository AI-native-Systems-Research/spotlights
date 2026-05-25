"""Stage registry for the signal pipeline.

Each stage module exposes a `SPEC: StageSpec`. This module assembles them
into the ordered registry the runner consumes.
"""

from __future__ import annotations

from spotlights_engine.signal_pipeline.layout import ALL_STAGES, StageId
from spotlights_engine.signal_pipeline.stages._types import (
    StageContext,
    StageSpec,
)
from spotlights_engine.signal_pipeline.stages.s01_signal_extraction import SPEC as _S01
from spotlights_engine.signal_pipeline.stages.s02_projecttree import SPEC as _S02
from spotlights_engine.signal_pipeline.stages.s03_candidate_generation import SPEC as _S03
from spotlights_engine.signal_pipeline.stages.s04_change_generation import SPEC as _S04
from spotlights_engine.signal_pipeline.stages.s05_execution import SPEC as _S05


STAGES: dict[StageId, StageSpec] = {
    "01": _S01,
    "02": _S02,
    "03": _S03,
    "04": _S04,
    "05": _S05,
}

# Sanity: keep registry alignment with the canonical stage order in
# `layout.ALL_STAGES`. A drift here would silently change resume gating.
assert tuple(STAGES.keys()) == ALL_STAGES, "STAGES out of sync with ALL_STAGES"


__all__ = ["STAGES", "StageContext", "StageSpec"]
