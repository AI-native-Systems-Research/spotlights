"""Stage 03 — candidate generation (Bundle C).

**Phase 1 stub.** Emits two placeholder `Candidate`s so the fan-out stages
downstream have something to iterate. Step 5 of the implementation order
replaces this with a `claude -p` session that exposes:

- `projecttree_by_path` (real, in-memory over cached `02_projecttree.json`)
- `read_module_source` (real, disk reads under `subject_root`)
- `get_raw_trace` (stubbed non-throwing — returns the canonical "unavailable"
  JSON shape; see `humble-plotting-cook.md` divergences §2)

Output is `list[Candidate]`. The artifact stores the *list* directly (not
wrapped in a `Candidates` envelope) — the envelope from main's
`schemas.candidate.Candidates` carries `module_qualified_name`, which
isn't a meaningful field in the signal-based pipeline (candidates may span
modules per their `locations[]`).
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.schemas.candidate import Candidate
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> list[Candidate]:
    if not isinstance(raw, list):
        raise TypeError(
            f"03_candidates.json must be a JSON list of Candidates; got {type(raw).__name__}"
        )
    return [Candidate.model_validate(item) for item in raw]


def run(ctx: StageContext) -> list[Candidate]:
    # Phase 1 stub. The two candidates exercise the fan-out path in stages 04/05.
    return [
        Candidate(
            id="cand-0001",
            file="stub_module/example.py",
            line_start=1,
            line_end=10,
            symbol="stub_function_a",
            kind="function",
            description="Phase 1 stub candidate.",
            current_approach="Stubbed.",
            evolve_rationale="Stubbed.",
            estimated_impact="medium",
            estimated_impact_explanation="Stubbed.",
        ),
        Candidate(
            id="cand-0002",
            file="stub_module/example.py",
            line_start=20,
            line_end=30,
            symbol="stub_function_b",
            kind="function",
            description="Phase 1 stub candidate.",
            current_approach="Stubbed.",
            evolve_rationale="Stubbed.",
            estimated_impact="low",
            estimated_impact_explanation="Stubbed.",
        ),
    ]


SPEC = StageSpec(
    stage_id="03",
    name="candidates",
    shape="single",
    upstream=("01", "02"),
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC"]
