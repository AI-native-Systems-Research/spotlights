"""Unified runner — orchestrates the selected pipelines over one extraction.

`run_unified(input, config) -> UnifiedResult` is the single entry. The CLI
surface is the top-level `--pipelines` flag (multi) and the
`spotlights-engine telemetry` prefix (telemetry-only).

High-level flow:

1. Setup paths.  Compute the unified-layer fingerprint and run id from the
   `UnifiedInput`.  If a previous unified run dir exists with a different
   fingerprint and `resume=False`, clear it; with `resume=True`, raise
   `UnifiedResumeMismatchError`.
2. Run the structural extractor once into `<run_dir>/_extractor/`.
3. Pre-populate the selected sub-pipelines' run dirs so each one's own
   resume logic sees stage 02 / step 1 as already complete.
4. Run the selected pipelines concurrently via `asyncio.gather`.  Telemetry
   is locked at `to_stage=04` (s05 mutates the subject; DR's `repo_guard`
   would catch it); enable s05 explicitly via the standalone `signal-pipeline`
   CLI.
5. Build per-pipeline `SpotlightReport`s (DR returns one in-memory; telemetry
   produced one to `<telemetry_run_dir>/spotlight_report.json`), persist DR's
   to `<run_dir>/deep_research/spotlight_report.json` for symmetry, then
   merge into the top-level `<run_dir>/spotlight_report.json` whose
   `RunInfo.pipelines` lists every contributor.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spotlights_engine.agent_proposals import AgentProposalsConfig
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
)
from spotlights_engine.schemas.pipeline import (
    SpotlightReport,
    SpotlightsManagerInput,
)
from spotlights_engine.signal_pipeline.runner import (
    SignalPipelineInput,
    StageSelection,
    run_pipeline as run_signal_pipeline,
)
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.api import (
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
)
from spotlights_engine.unified_runner.errors import (
    UnifiedResumeMismatchError,
    UnifiedSetupError,
)
from spotlights_engine.unified_runner.merge import merge_reports
from spotlights_engine.unified_runner.schemas import (
    UnifiedConfig,
    UnifiedInput,
    UnifiedResult,
    UnifiedRunSummary,
)
from spotlights_engine.unified_runner.shared_extraction import (
    extract_once,
    populate_dr_run_dir,
    populate_signal_run_dir,
)

_log = logging.getLogger(__name__)

_MANIFEST_NAME = "manifest.json"
_EXTRACTOR_DIR = "_extractor"
_TELEMETRY_DIR = "telemetry"
_DR_DIR = "deep_research"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unified_fingerprint(
    input: UnifiedInput, config: UnifiedConfig
) -> dict[str, Any]:
    """Content-addressed identity of one unified run.

    Mirrors DR's input/config fingerprint pattern so `--no-resume` semantics
    feel consistent: change the meaning of the run → fingerprint diverges →
    resume fails.  Path-only fields (`artifacts_dir`, `output_folder`) and
    surface knobs that don't affect output content (`max_parallel_*`) are
    excluded so re-running the same logical work in a different layout still
    resumes cleanly.
    """
    return {
        "pipelines": sorted(input.pipelines),
        "repo_path": str(input.repo_path),
        "context_hash": _stable_hash(input.context.model_dump(mode="json")),
        "telemetry_from": str(input.telemetry_from)
        if input.telemetry_from is not None
        else None,
        "backend_id": input.backend_id,
        "max_candidates": input.max_candidates,
        "model": input.model,
        "max_findings_per_module": input.max_findings_per_module,
        "continue_on_module_failure": input.continue_on_module_failure,
        "module_filter": (
            config.module_filter.model_dump(mode="json")
            if config.module_filter is not None
            else None
        ),
        "debug_first_n_pairs": config.debug_first_n_pairs,
        "debug_first_n_candidates": config.debug_first_n_candidates,
    }


def _run_id_from_fingerprint(fp: dict[str, Any]) -> str:
    payload = json.dumps(fp, sort_keys=True, separators=(",", ":"))
    return "run-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _ensure_compatible_run_dir(
    *,
    run_dir: Path,
    fingerprint: dict[str, Any],
    resume: bool,
) -> None:
    """Apply the unified-layer resume policy.

    - No manifest → fresh run, nothing to do.
    - Matching manifest → resume both sub-pipelines as-is.
    - Mismatching manifest with `resume=False` → wipe `run_dir` and start fresh.
    - Mismatching manifest with `resume=True` → raise `UnifiedResumeMismatchError`.
    """
    manifest_path = run_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return

    try:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        if not resume:
            shutil.rmtree(run_dir)
            return
        raise UnifiedSetupError(
            f"unified manifest at {manifest_path} is unreadable: {e}; "
            "pass --no-resume to clear the dir or fix it manually"
        ) from e

    if existing.get("fingerprint") == fingerprint:
        return

    if resume:
        raise UnifiedResumeMismatchError(
            f"unified run dir {run_dir} fingerprint differs from current "
            "invocation; pass --no-resume to clear and start fresh"
        )

    shutil.rmtree(run_dir)


def _write_unified_manifest(
    *,
    run_dir: Path,
    fingerprint: dict[str, Any],
    run_id: str,
    started_at: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "fingerprint": fingerprint,
        "created_at": started_at,
        "updated_at": _now_iso(),
        "status": "RUNNING",
    }
    (run_dir / _MANIFEST_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _finalize_unified_manifest(
    *, run_dir: Path, status: str, finished_at: str, cost_usd: float | None
) -> None:
    manifest_path = run_dir / _MANIFEST_NAME
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["finished_at"] = finished_at
    payload["cost_usd"] = cost_usd
    payload["updated_at"] = _now_iso()
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _build_telemetry_input(
    input: UnifiedInput, telemetry_run_dir: Path
) -> SignalPipelineInput:
    return SignalPipelineInput(
        subject_root=input.repo_path,
        telemetry_from=input.telemetry_from,
        backend_id=input.backend_id,
        # The unified runner pre-populates stage 02 artifacts; the telemetry
        # pipeline's built-in cross-run cache becomes a no-op for this run.
        # Set False so any cache miss surfaces clearly rather than silently
        # re-running the agent.
        projecttree_cache=False,
        max_candidates=input.max_candidates,
        model=input.model,
        context=input.context,
    )


def _build_dr_input(input: UnifiedInput) -> SpotlightsManagerInput:
    kwargs: dict[str, Any] = {
        "repo_path": input.repo_path,
        "context": input.context,
        "continue_on_module_failure": input.continue_on_module_failure,
    }
    if input.max_findings_per_module is not None:
        kwargs["max_findings_per_module"] = input.max_findings_per_module
    return SpotlightsManagerInput(**kwargs)


def _build_dr_config(
    *, input: UnifiedInput, config: UnifiedConfig, dr_artifacts_dir: Path
) -> SpotlightsManagerConfig:
    proposal_cfg: ProposalFromFindingConfig | None = None
    if (
        config.max_parallel_pairs is not None
        or config.debug_first_n_pairs is not None
    ):
        kwargs: dict[str, Any] = {}
        if config.max_parallel_pairs is not None:
            kwargs["max_parallel_pairs"] = config.max_parallel_pairs
        if config.debug_first_n_pairs is not None:
            kwargs["debug_first_n_pairs"] = config.debug_first_n_pairs
        proposal_cfg = ProposalFromFindingConfig(**kwargs)

    agent_cfg: AgentProposalsConfig | None = None
    if (
        config.max_parallel_candidates is not None
        or config.debug_first_n_candidates is not None
    ):
        kwargs = {}
        if config.max_parallel_candidates is not None:
            kwargs["max_parallel_candidates"] = config.max_parallel_candidates
        if config.debug_first_n_candidates is not None:
            kwargs["debug_first_n_candidates"] = config.debug_first_n_candidates
        agent_cfg = AgentProposalsConfig(**kwargs)

    return SpotlightsManagerConfig(
        artifacts_dir=dr_artifacts_dir,
        output_folder=dr_artifacts_dir / "output",
        max_parallel_sessions=config.max_parallel_sessions,
        module_filter=config.module_filter,
        proposal_from_finding=proposal_cfg,
        agent_proposals=agent_cfg,
        # Sub-pipeline always resumes: the unified runner pre-populates
        # state and a unified-level --no-resume is handled by clearing the
        # whole run dir before this point.
        resume=True,
    )


def _load_telemetry_report(telemetry_run_dir: Path) -> SpotlightReport | None:
    """Read the telemetry pipeline's auto-emitted `spotlight_report.json`."""
    path = telemetry_run_dir / "spotlight_report.json"
    if not path.exists():
        return None
    try:
        return SpotlightReport.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("failed to load telemetry report at %s: %s", path, exc)
        return None


def _persist_dr_report(report: SpotlightReport, dr_dir: Path) -> Path:
    """Mirror the telemetry pipeline's behavior by writing DR's report to its
    sub-run-dir (DR doesn't persist its own SpotlightReport today)."""
    dr_dir.mkdir(parents=True, exist_ok=True)
    target = dr_dir / "spotlight_report.json"
    target.write_text(
        report.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8"
    )
    return target


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


def run_unified(input: UnifiedInput, *, config: UnifiedConfig) -> UnifiedResult:
    """Run the unified pipeline (sync wrapper around the async core)."""
    return asyncio.run(_run_async(input, config=config))


async def _run_async(input: UnifiedInput, *, config: UnifiedConfig) -> UnifiedResult:
    started_at = _now_iso()
    fingerprint = _unified_fingerprint(input, config)
    run_id = _run_id_from_fingerprint(fingerprint)

    # Resolve the run dir so the unified manifest, _extractor/, signal/, and
    # deep_research/ all share one root.  We park each unified run under a
    # `<artifacts_dir>/<run_id>/` subtree so concurrent runs against the
    # same artifacts dir don't collide.
    run_dir = (config.artifacts_dir / run_id).resolve()
    _ensure_compatible_run_dir(
        run_dir=run_dir, fingerprint=fingerprint, resume=config.resume
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_unified_manifest(
        run_dir=run_dir,
        fingerprint=fingerprint,
        run_id=run_id,
        started_at=started_at,
    )

    extractor_dir = run_dir / _EXTRACTOR_DIR
    telemetry_run_dir = run_dir / _TELEMETRY_DIR
    dr_artifacts_dir = run_dir / _DR_DIR
    extractor_done_marker = run_dir / _EXTRACTOR_DIR / ".extracted"

    issues: list[str] = []

    # ── Step 1 — extract once (skipped on resume if the marker exists) ───
    from spotlights_engine.modules_extractor.agent import ExtractionInvocation
    from spotlights_engine.schemas.project import ProjectTree

    if extractor_done_marker.exists() and (extractor_dir / "project_tree.json").exists():
        _log.info("unified runner: extractor cached")
        project_tree = ProjectTree.model_validate_json(
            (extractor_dir / "project_tree.json").read_text(encoding="utf-8")
        )
        invocation_payload = json.loads(
            (extractor_dir / "extractor_invocation.json").read_text(encoding="utf-8")
        )
        invocation = ExtractionInvocation(**invocation_payload)
        extraction_duration = 0.0
    else:
        _log.info("unified runner: running extractor against %s", input.repo_path)
        extract_start = time.monotonic()
        extracted = extract_once(repo_path=input.repo_path, cache_dir=extractor_dir)
        extraction_duration = time.monotonic() - extract_start
        project_tree = extracted.project_tree
        invocation = extracted.invocation
        import dataclasses

        (extractor_dir / "project_tree.json").write_text(
            project_tree.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        (extractor_dir / "extractor_invocation.json").write_text(
            json.dumps(dataclasses.asdict(invocation), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        extractor_done_marker.write_text(_now_iso() + "\n", encoding="utf-8")

    # ── Step 2 — pre-populate sub-pipeline run dirs ──────────────────────
    will_run_telemetry = "telemetry" in input.pipelines
    will_run_dr = "deep_research" in input.pipelines

    if will_run_telemetry:
        populate_signal_run_dir(
            signal_run_dir=telemetry_run_dir, project_tree=project_tree
        )

    dr_input: SpotlightsManagerInput | None = None
    dr_config: SpotlightsManagerConfig | None = None
    if will_run_dr:
        dr_input = _build_dr_input(input)
        dr_config = _build_dr_config(
            input=input, config=config, dr_artifacts_dir=dr_artifacts_dir
        )
        populate_dr_run_dir(
            dr_artifacts_dir=dr_artifacts_dir,
            dr_input=dr_input,
            dr_config=dr_config,
            project_tree=project_tree,
            invocation=invocation,
            extractor_duration_s=extraction_duration,
        )

    # ── Step 3 — run sub-pipelines concurrently ──────────────────────────
    telemetry_input = (
        _build_telemetry_input(input, telemetry_run_dir)
        if will_run_telemetry
        else None
    )

    async def _telemetry_task() -> tuple[SpotlightReport | None, list[str]]:
        if telemetry_input is None:
            return None, []
        # Telemetry pipeline is locked to stage 04: stage 05 mutates the
        # subject and would collide with DR's repo_guard.  Stage 05 stays
        # opt-in via the standalone `signal-pipeline` CLI for now.
        sel = StageSelection(from_stage="01", to_stage="04")
        result = await asyncio.to_thread(
            run_signal_pipeline,
            telemetry_input,
            run_dir=telemetry_run_dir,
            stages=sel,
            resume=True,
        )
        report = _load_telemetry_report(telemetry_run_dir)
        return report, list(result.issues)

    async def _dr_task() -> tuple[SpotlightReport | None, list[str]]:
        if dr_input is None or dr_config is None:
            return None, []
        # `_run_async` is the orchestrator's async core; call it directly so
        # we don't pay for the extra `asyncio.run` that `run` / `run_with_telemetry`
        # would impose.
        from spotlights_engine.spotlights_manager.orchestrator import (
            _run_async as dr_run_async,
        )

        result: SpotlightsManagerResult = await dr_run_async(dr_input, config=dr_config)
        # Persist DR's per-pipeline report for symmetry with the telemetry
        # pipeline's auto-emit.
        _persist_dr_report(result.report, dr_artifacts_dir)
        dr_issues = [
            f"{issue.step}: {issue.message}" for issue in result.manager_issues
        ]
        return result.report, dr_issues

    telemetry_pair, dr_pair = await asyncio.gather(_telemetry_task(), _dr_task())
    telemetry_report, telemetry_issues = telemetry_pair
    dr_report, dr_issues = dr_pair
    issues.extend(f"telemetry: {i}" for i in telemetry_issues)
    issues.extend(f"dr: {i}" for i in dr_issues)

    if telemetry_report is None and dr_report is None:
        finished_at = _now_iso()
        _finalize_unified_manifest(
            run_dir=run_dir,
            status="FAILED",
            finished_at=finished_at,
            cost_usd=None,
        )
        raise UnifiedSetupError(
            "unified runner: no selected pipeline produced a SpotlightReport; "
            f"see issues: {issues}"
        )

    # ── Step 4 — merge ───────────────────────────────────────────────────
    merged = merge_reports(
        telemetry=telemetry_report, dr=dr_report, run_id=run_id
    )

    target = run_dir / "spotlight_report.json"
    target.write_text(
        merged.model_dump_json(indent=2, by_alias=True) + "\n", encoding="utf-8"
    )

    finished_at = _now_iso()
    summary = UnifiedRunSummary(
        pipelines=sorted(input.pipelines),
        started_at=started_at,
        finished_at=finished_at,
        cost_usd=merged.run.cost_usd,
        telemetry_run_dir=telemetry_run_dir if will_run_telemetry else None,
        dr_artifacts_dir=dr_artifacts_dir if will_run_dr else None,
        issues=issues,
    )
    (run_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    _finalize_unified_manifest(
        run_dir=run_dir,
        status="COMPLETE",
        finished_at=finished_at,
        cost_usd=merged.run.cost_usd,
    )

    return UnifiedResult(report=merged, run_dir=run_dir, summary=summary)


__all__ = ["run_unified"]
