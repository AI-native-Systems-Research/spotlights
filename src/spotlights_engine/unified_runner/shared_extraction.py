"""Extract `ProjectTree` once, then pre-populate each sub-pipeline's resume state.

Strategy:
1. Run `modules_extractor.extract_with_telemetry(...)` once into the unified
   `_extractor/` sub-dir.
2. For the signal sub-run-dir: write `02_projecttree.json` (matching
   `signal_pipeline.layout.RunDirLayout.stage_artifact("02", shape="single")`)
   and update `status.json` so signal's resume logic sees stage 02 as `done`.
3. For the DR sub-run-dir: call `spotlights_manager.persistence.write_extractor_outputs`
   to drop `project_tree.json` + `extractor_invocation.json`, then write a
   manager manifest with `extractor.completed=True` and matching fingerprints.

When DR's `_run_async` boots with `resume=True`, `_ensure_resume_compatible`
finds the prepopulated manifest, fingerprints match, and
`_run_extractor_if_needed` short-circuits on the cached state. Same for signal:
`_is_complete` returns True for stage 02 because the artifact + status are
already on disk.

Notes / gotchas:

- The unified runner does NOT touch the cross-run signal cache at
  `~/.cache/spotlights-engine/projecttree/`. That cache's key/version logic is
  owned by `signal_pipeline.stages.s02_projecttree`; reaching into it from
  outside risks divergence. A future improvement could expose a public helper
  there for the unified runner to call.
- DR fingerprints are computed via the same helpers DR's own orchestrator
  uses, so the resume check is byte-identical to a normal DR run that just
  crashed-and-resumed mid-extractor.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from spotlights_engine.modules_extractor import (
    ExtractorConfig,
    ExtractorResult,
    extract_with_telemetry,
)
from spotlights_engine.modules_extractor.agent import ExtractionInvocation
from spotlights_engine.schemas.pipeline import (
    ModulesExtractorInput,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import ProjectTree
from spotlights_engine.signal_pipeline.layout import (
    RunDirLayout,
    atomic_write_json,
    atomic_write_text,
)
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.api import SpotlightsManagerConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def extract_once(
    *,
    repo_path: Path,
    cache_dir: Path,
    on_event: Callable[[str], None] | None = None,
) -> ExtractorResult:
    """Run the structural extractor exactly once into `cache_dir`.

    `cache_dir` becomes the extractor's `artifacts_dir`, so its prompt /
    `raw_stdout.log` / `schema.json` land under `<cache_dir>/<timestamp>/`.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg = ExtractorConfig(artifacts_dir=cache_dir)
    return extract_with_telemetry(
        ModulesExtractorInput(repo_path=repo_path),
        config=cfg,
        on_event=on_event,
    )


def populate_signal_run_dir(
    *,
    signal_run_dir: Path,
    project_tree: ProjectTree,
) -> None:
    """Pre-write signal's stage 02 artifact + a `status.json` marking it done.

    Signal's runner (`run_pipeline`) under `resume=True` calls `_is_complete`
    for each selected stage; for a single-shape stage that requires (a) the
    canonical artifact file exists and parses, AND (b) `status.stages[NN].state
    == "done"`. We satisfy both.
    """
    signal_run_dir.mkdir(parents=True, exist_ok=True)
    layout = RunDirLayout(signal_run_dir.resolve())

    # 1) Stage 02 artifact — match `_run_single`'s serialization exactly.
    artifact_path = layout.stage_artifact("02", shape="single")
    payload = project_tree.model_dump(
        mode="json", by_alias=True, exclude_none=False
    )
    atomic_write_json(artifact_path, payload)

    # 2) Status: mark stage 02 done.  Read the existing status if any so we
    # don't clobber other stages' completion state on reruns.
    status_doc: dict = {"stages": {}}
    if layout.status_path.exists():
        try:
            status_doc = json.loads(layout.status_path.read_text(encoding="utf-8"))
            if not isinstance(status_doc.get("stages"), dict):
                status_doc["stages"] = {}
        except (json.JSONDecodeError, OSError):
            status_doc = {"stages": {}}
    now = _now_iso()
    status_doc["stages"]["02"] = {
        "state": "done",
        "started_at": now,
        "ended_at": now,
        "error": None,
        "issues": [],
        "model": None,
        "cost_usd": None,
        "duration_s": None,
    }
    atomic_write_text(
        layout.status_path,
        json.dumps(status_doc, indent=2, sort_keys=True) + "\n",
    )


def populate_dr_run_dir(
    *,
    dr_artifacts_dir: Path,
    dr_input: SpotlightsManagerInput,
    dr_config: SpotlightsManagerConfig,
    project_tree: ProjectTree,
    invocation: ExtractionInvocation,
    extractor_duration_s: float,
) -> None:
    """Pre-write DR's extractor outputs + a manifest with `extractor.completed=True`.

    Fingerprints are computed via the same helpers DR uses internally so
    `_ensure_resume_compatible` accepts the prepopulated state on resume.
    """
    paths = P.ManagerPaths(artifacts_dir=dr_artifacts_dir)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.modules_root.mkdir(parents=True, exist_ok=True)

    P.write_extractor_outputs(paths, project_tree, invocation)

    input_fp = P.build_input_fingerprint(
        repo_path=dr_input.repo_path,
        context=dr_input.context,
        max_findings_per_module=dr_input.max_findings_per_module,
        continue_on_module_failure=dr_input.continue_on_module_failure,
    )
    config_fp = P.build_config_fingerprint(
        module_filter=dr_config.module_filter,
        extractor_cfg=dr_config.extractor,
        discovery_cfg=dr_config.discovery,
        deep_research_cfg=dr_config.deep_research,
        proposal_from_finding_cfg=dr_config.proposal_from_finding,
        agent_proposals_cfg=dr_config.agent_proposals,
    )

    now = _now_iso()
    manifest: dict = {
        "schema_version": P.SCHEMA_VERSION,
        "created_at": now,
        "updated_at": now,
        "status": "RUNNING",
        "input_fingerprint": input_fp,
        "config_fingerprint": config_fp,
        "context": dr_input.context.model_dump(mode="json"),
        "extractor": {
            "completed": True,
            "duration_s": extractor_duration_s,
        },
        "modules": {},
    }
    P.write_manifest(paths, manifest)


__all__ = [
    "extract_once",
    "populate_dr_run_dir",
    "populate_signal_run_dir",
]
