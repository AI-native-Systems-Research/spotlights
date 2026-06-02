"""Stage 02 — ProjectTree extraction.

Delegates to `spotlights_engine.modules_extractor.extract` from origin/main
(LLM-backed, runs `claude -p` on the subject repo). Each invocation places
its modules-extractor working tree under a fresh timestamped subdir of the
stage's `log_dir`, sidestepping main's "pre-existing run dir is unsupported"
guard so re-runs don't collide.

⚠ Doc divergence flagged in the plan
(`/Users/idanfr/.claude/plans/humble-plotting-cook.md` "Known doc/code
divergences §1"): the flow doc labels this stage "deterministic, no LLM"
but the schema's `description` / `role` fields require an LLM. Resolution
belongs to the flow-doc owner.

The `modules_extractor` import is local to `_extract_project_tree` (not
at module top) so a missing origin/main layout fails fast in the runner's
`_check_layout()` rather than as an opaque ImportError at package import.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> ProjectTree:
    return ProjectTree.model_validate(raw)


def _ts_subdir() -> str:
    """Filesystem-safe UTC ISO timestamp for a per-invocation subdir name."""
    iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return re.sub(r"[:+]", "-", iso)


def _extract_project_tree(
    subject_root: Path, log_dir: Path, on_event=None
) -> ProjectTree:
    """Real extraction path. Factored for ease of monkeypatching in tests.

    The conftest in `tests/unit/signal_pipeline/` swaps this out with a
    placeholder so runner state-machine tests don't fire Claude.
    """
    # Lazy import per the safety order in the approved plan.
    from spotlights_engine.modules_extractor import ExtractorConfig, extract
    from spotlights_engine.schemas.pipeline import ModulesExtractorInput

    artifacts_dir = log_dir / _ts_subdir()
    return extract(
        ModulesExtractorInput(repo_path=subject_root),
        config=ExtractorConfig(artifacts_dir=artifacts_dir),
        on_event=on_event,
    )


def run(ctx: StageContext) -> ProjectTree:
    return _extract_project_tree(
        ctx.signal_input.subject_root, ctx.log_dir, ctx.on_event
    )


SPEC = StageSpec(
    stage_id="02",
    name="projecttree",
    shape="single",
    upstream=(),  # ProjectTree extraction takes subject_root only, no signals dep
    parse_artifact=parse_artifact,
    run=run,
)


__all__ = ["SPEC"]
