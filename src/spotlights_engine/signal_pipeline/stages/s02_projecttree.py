"""Stage 02 — ProjectTree extraction.

**Phase 1 stub.** Emits a minimal `ProjectTree` so downstream stages can
run end-to-end. Step 3 of the implementation order replaces this with a
delegation to `spotlights_engine.modules_extractor.extract`.

Doc divergence flagged in
`/Users/idanfr/.claude/plans/humble-plotting-cook.md` "Known doc/code
divergences §1": the flow doc labels this stage "deterministic, no LLM"
but main's `modules_extractor` is LLM-backed (the schema's `description` /
`role` fields require it). Resolution belongs to the flow-doc owner.

The `modules_extractor.extract` import lives inside `run` (not at module
top) so a missing origin/main layout fails fast via the runner's
`_check_layout()` precondition rather than as an opaque ImportError at
package import time.
"""

from __future__ import annotations

from typing import Any

from spotlights_engine.schemas.project import Module, ProjectTree, Repository
from spotlights_engine.signal_pipeline.stages._types import StageContext, StageSpec


def parse_artifact(raw: Any) -> ProjectTree:
    return ProjectTree.model_validate(raw)


def run(ctx: StageContext) -> ProjectTree:
    # Phase 1 stub. Step 3 replaces with:
    #   from spotlights_engine.modules_extractor import extract, ExtractorConfig
    #   from spotlights_engine.schemas.pipeline import ModulesExtractorInput
    #   return extract(ModulesExtractorInput(repo_path=ctx.signal_input.subject_root))
    return ProjectTree(
        repository=Repository(
            name="stub-repo",
            summary="Phase 1 stub — replace via modules_extractor.extract in step 3.",
        ),
        modules=[
            Module(
                name="stub_module",
                path="stub_module",
                description="Phase 1 stub module.",
            )
        ],
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
