"""Async per-run orchestration.

Runs step 1 (extractor) once, then schedules per-module pipelines under a
bounded `asyncio.Semaphore`. Each per-module task wraps the existing **sync**
step entrypoints with `asyncio.to_thread` so the steps do not need to be
refactored to be async. The on-disk checkpoint tree is the single source of
truth; in-memory state is only there to drive the scheduling.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spotlights_engine.candidate_discovery import (
    DiscoveryConfig,
    DiscoveryMutationError,
    DiscoverySetupError,
    DiscoveryValidationError,
    discover,
)
from spotlights_engine.module_deep_research import research_module
from spotlights_engine.module_deep_research.codex_exec import CodexExecOptions
from spotlights_engine.modules_extractor import (
    ExtractorConfig,
    extract_with_telemetry,
)
from spotlights_engine.schemas.candidate import Candidates
from spotlights_engine.schemas.common import (
    ModuleRunStatus,
    PipelineStep,
    StepIssue,
)
from spotlights_engine.schemas.pipeline import (
    CandidateDiscoveryInput,
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
    ModuleRun,
    ModulesExtractorInput,
    SpotlightsManagerInput,
)
from spotlights_engine.schemas.project import Module, ProjectTree
from spotlights_engine.spotlights_manager.api import (
    ModuleTelemetry,
    SpotlightsManagerConfig,
    SpotlightsManagerResult,
)
from spotlights_engine.spotlights_manager.errors import (
    ManagerSetupError,
    ResumeMismatchError,
)
from spotlights_engine.spotlights_manager.filters import apply_filter
from spotlights_engine.spotlights_manager import persistence as P
from spotlights_engine.spotlights_manager.persistence import (
    LoadedModuleState,
    ManagerPaths,
    ModuleCheckpoint,
    ModulePaths,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slash_to_dot(qn: str) -> str:
    return qn.replace("/", ".")


def _issue(
    step: PipelineStep, message: str, *, recoverable: bool, severity: str = "error"
) -> StepIssue:
    return StepIssue(
        step=step,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        recoverable=recoverable,
    )


def _final_status_from_research(output: ModuleDeepResearchOutput) -> ModuleRunStatus:
    """Translate step-3 issues into the architectural `ModuleRunStatus`.

    - any `recoverable=False` issue -> `FAILED`
    - any issue at all (all recoverable) -> `DEGRADED`
    - no issues -> `SUCCEEDED` (empty `findings` is allowed)
    """
    if any(not iss.recoverable for iss in output.issues):
        return "FAILED"
    if output.issues:
        return "DEGRADED"
    return "SUCCEEDED"


def _build_discovery_config(
    cfg: SpotlightsManagerConfig,
    repo_path: Path,
    artifacts_dir: Path,
) -> DiscoveryConfig:
    base = cfg.discovery if cfg.discovery is not None else DiscoveryConfig()
    return base.model_copy(
        update={"repo_path": repo_path, "artifacts_dir": artifacts_dir}
    )


def _build_deep_research_options(
    cfg: SpotlightsManagerConfig,
    repo_path: Path,
    last_message_path: Path,
) -> CodexExecOptions:
    """Per-module options: copy any caller config and override `cwd` and
    `output_last_message`. Never mutates the shared object."""
    base = cfg.deep_research
    if base is None:
        return CodexExecOptions(cwd=repo_path, output_last_message=last_message_path)
    return base.model_copy(
        update={"cwd": repo_path, "output_last_message": last_message_path}
    )


def _now_checkpoint(
    *,
    qn: str,
    status: P.CheckpointStatus,
    last_step: PipelineStep | None,
    failed_step: PipelineStep | None = None,
    error: str | None = None,
    retryable: bool = False,
    issues: list[StepIssue] | None = None,
    started_at: str | None = None,
) -> ModuleCheckpoint:
    return ModuleCheckpoint(
        module_qualified_name=qn,
        status=status,
        last_step=last_step,
        failed_step=failed_step,
        error=error,
        retryable=retryable,
        issues=list(issues or []),
        started_at=started_at or _now_iso(),
        updated_at=_now_iso(),
    )


# ---------------------------------------------------------------------------
# Setup / resume
# ---------------------------------------------------------------------------


def _validate_setup(input: SpotlightsManagerInput, paths: ManagerPaths) -> None:
    repo = input.repo_path
    if not repo.exists() or not repo.is_dir():
        raise ManagerSetupError(
            f"repo_path does not exist or is not a directory: {repo}",
            repo_path=str(repo),
        )
    resolved_repo = repo.resolve(strict=False)
    resolved_artifacts = paths.artifacts_dir.resolve(strict=False)
    try:
        resolved_artifacts.relative_to(resolved_repo)
    except ValueError:
        pass
    else:
        raise ManagerSetupError(
            f"artifacts_dir resolves inside repo_path: {paths.artifacts_dir}",
            artifacts_dir=str(paths.artifacts_dir),
            repo_path=str(repo),
        )


def _ensure_resume_compatible(
    paths: ManagerPaths,
    input_fp: dict[str, Any],
    config_fp: dict[str, Any],
    *,
    resume: bool,
) -> dict[str, Any]:
    existing = P.read_manifest(paths)
    if existing is None:
        return P.init_manifest(
            paths,
            input_fingerprint=input_fp,
            config_fingerprint=config_fp,
        )

    if not resume:
        raise ManagerSetupError(
            "spotlights_manager run dir exists and resume=False; "
            "remove the directory or pass a fresh artifacts_dir",
            run_dir=str(paths.root),
        )

    if existing.get("input_fingerprint") != input_fp:
        raise ResumeMismatchError(
            "input fingerprint changed since the manager run dir was created",
            existing=existing.get("input_fingerprint"),
            current=input_fp,
        )
    if existing.get("config_fingerprint") != config_fp:
        raise ResumeMismatchError(
            "config fingerprint changed since the manager run dir was created",
            existing=existing.get("config_fingerprint"),
            current=config_fp,
        )

    # Resuming: clear terminal status so a previously-FAILED run can re-enter.
    existing["status"] = "RUNNING"
    P.write_manifest(paths, existing)
    return existing


# ---------------------------------------------------------------------------
# Step 1 — extractor
# ---------------------------------------------------------------------------


def _run_extractor_if_needed(
    input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    paths: ManagerPaths,
    manifest: dict[str, Any],
) -> tuple[ProjectTree, Any]:
    extractor_state = manifest.get("extractor", {})
    completed = bool(extractor_state.get("completed"))

    tree, invocation = P.read_extractor_outputs(paths)

    if completed and tree is not None and invocation is not None:
        return tree, invocation

    # Anything other than "fully complete with both sidecars on disk" forces a
    # re-run. Per plan §7.5, mid-step-1 crashes leave artifacts behind; clear
    # them so the per-step "no resume" guard is satisfied.
    P.clear_extractor_artifacts(paths)

    extractor_cfg: ExtractorConfig = cfg.extractor.model_copy(
        update={"artifacts_dir": paths.root}
    )

    start = time.monotonic()
    try:
        result = extract_with_telemetry(
            ModulesExtractorInput(repo_path=input.repo_path),
            config=extractor_cfg,
        )
    except Exception as exc:
        manifest["status"] = "FAILED"
        manifest["extractor"] = {
            "completed": False,
            "duration_s": time.monotonic() - start,
            "error": f"{type(exc).__name__}: {exc}",
        }
        P.write_manifest(paths, manifest)
        raise
    duration = time.monotonic() - start

    P.write_extractor_outputs(paths, result.project_tree, result.invocation)
    manifest["extractor"] = {"completed": True, "duration_s": duration}
    P.write_manifest(paths, manifest)
    return result.project_tree, result.invocation


# ---------------------------------------------------------------------------
# Per-module decision + execution
# ---------------------------------------------------------------------------


@dataclass
class _ModulePlan:
    """What `_run_module` is going to do, derived from on-disk state."""

    skip_module: bool
    redo_step2: bool
    redo_step3: bool
    started_at: str


_FAIL_FAST_CANCELLED_MESSAGE = (
    "module not started because another module failed and "
    "continue_on_module_failure=False"
)


def _plan_module(state: LoadedModuleState) -> _ModulePlan:
    """Resolve plan §7.4 onto a pair of booleans. The orchestrator always runs
    step 3 if `redo_step3` is True OR step 2 just produced fresh candidates."""
    cp = state.checkpoint
    started_at = cp.started_at if cp is not None else _now_iso()

    if cp is None:
        return _ModulePlan(False, redo_step2=True, redo_step3=False, started_at=started_at)

    if cp.status in {"SUCCEEDED", "DEGRADED", "SKIPPED"}:
        return _ModulePlan(True, False, False, started_at)
    if cp.status == "FAILED" and not cp.retryable:
        return _ModulePlan(True, False, False, started_at)

    if cp.status == "FAILED" and cp.retryable:
        if cp.failed_step == "module_deep_research":
            return _ModulePlan(False, False, True, started_at)
        # Anything else (incl. unknown/None) -> redo step 2 from scratch.
        return _ModulePlan(False, True, False, started_at)

    if cp.status == "PENDING":
        return _ModulePlan(False, True, False, started_at)

    if cp.status == "DISCOVERED":
        if state.candidates is None:
            return _ModulePlan(False, True, False, started_at)
        # Need step 3.
        return _ModulePlan(False, False, True, started_at)

    if cp.status == "DEEP_RESEARCHED":
        if state.deep_research is None:
            return _ModulePlan(False, False, True, started_at)
        return _ModulePlan(False, False, False, started_at)  # finalize-only

    return _ModulePlan(False, True, False, started_at)


def _plan_requires_step_execution(plan: _ModulePlan) -> bool:
    return plan.redo_step2 or plan.redo_step3


async def _do_step2(
    *,
    qn: str,
    tree: ProjectTree,
    mgr_input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    module_paths: ModulePaths,
) -> tuple[Candidates, list, float, float | None]:
    """Returns (candidates, iterations, total_duration_s, total_cost_usd)."""
    discovery_input = CandidateDiscoveryInput(
        project_tree=tree,
        module_qualified_name=qn,
        context=mgr_input.context,
    )
    discovery_cfg = _build_discovery_config(
        cfg, mgr_input.repo_path, module_paths.dir
    )
    result = await asyncio.to_thread(discover, discovery_input, config=discovery_cfg)
    return (
        result.candidates,
        list(result.iterations),
        result.total_duration_s,
        result.total_cost_usd,
    )


async def _do_step3(
    *,
    qn: str,
    tree: ProjectTree,
    mgr_input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    module_paths: ModulePaths,
) -> tuple[ModuleDeepResearchOutput, float]:
    research_input = ModuleDeepResearchInput(
        project_tree=tree,
        module_qualified_name=qn,
        context=mgr_input.context,
        repo_path=mgr_input.repo_path,
        max_findings_per_module=mgr_input.max_findings_per_module,
    )
    options = _build_deep_research_options(
        cfg, mgr_input.repo_path, module_paths.deep_research_last_message_path
    )
    start = time.monotonic()
    output = await asyncio.to_thread(research_module, research_input, options)
    duration = time.monotonic() - start
    return output, duration


async def _write_cancelled_checkpoint(
    *,
    qn: str,
    state: LoadedModuleState,
    plan: _ModulePlan,
    paths: ManagerPaths,
    module_paths: ModulePaths,
    manifest_lock: asyncio.Lock,
    manifest: dict[str, Any],
) -> ModuleCheckpoint:
    failed_step: PipelineStep = (
        "module_deep_research"
        if plan.redo_step3 and not plan.redo_step2
        else "candidate_discovery"
    )
    cp = _now_checkpoint(
        qn=qn,
        status="FAILED",
        last_step=state.checkpoint.last_step if state.checkpoint else None,
        failed_step=failed_step,
        error=_FAIL_FAST_CANCELLED_MESSAGE,
        retryable=True,
        issues=[
            _issue(
                failed_step,
                _FAIL_FAST_CANCELLED_MESSAGE,
                recoverable=True,
            )
        ],
        started_at=plan.started_at,
    )
    P.write_checkpoint(module_paths, cp)
    await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
    return cp


async def _run_module(
    *,
    qn: str,
    sem: asyncio.Semaphore,
    cancel_event: asyncio.Event | None,
    tree: ProjectTree,
    mgr_input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    paths: ManagerPaths,
    manifest_lock: asyncio.Lock,
    manifest: dict[str, Any],
) -> ModuleCheckpoint:
    module_paths = paths.for_module(qn)
    module_paths.dir.mkdir(parents=True, exist_ok=True)
    state = P.read_module_state(module_paths)
    plan = _plan_module(state)

    if plan.skip_module:
        assert state.checkpoint is not None
        return state.checkpoint

    if (
        cancel_event is not None
        and cancel_event.is_set()
        and _plan_requires_step_execution(plan)
    ):
        return await _write_cancelled_checkpoint(
            qn=qn,
            state=state,
            plan=plan,
            paths=paths,
            module_paths=module_paths,
            manifest_lock=manifest_lock,
            manifest=manifest,
        )

    async with sem:
        if (
            cancel_event is not None
            and cancel_event.is_set()
            and _plan_requires_step_execution(plan)
        ):
            return await _write_cancelled_checkpoint(
                qn=qn,
                state=state,
                plan=plan,
                paths=paths,
                module_paths=module_paths,
                manifest_lock=manifest_lock,
                manifest=manifest,
            )

        # ------------------------- step 2 -----------------------------------
        candidates: Candidates | None = state.candidates
        if plan.redo_step2:
            P.clear_discovery_artifacts(module_paths)
            P.clear_deep_research_artifacts(module_paths)
            cp = _now_checkpoint(
                qn=qn,
                status="PENDING",
                last_step=None,
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)

            try:
                candidates, iters, dur, cost = await _do_step2(
                    qn=qn,
                    tree=tree,
                    mgr_input=mgr_input,
                    cfg=cfg,
                    module_paths=module_paths,
                )
            except Exception as e:  # noqa: BLE001
                if isinstance(
                    e, (DiscoverySetupError, DiscoveryValidationError, ValueError)
                ):
                    retryable = False
                elif isinstance(e, DiscoveryMutationError):
                    retryable = True
                else:
                    retryable = True
                cp = _now_checkpoint(
                    qn=qn,
                    status="FAILED",
                    last_step=None,
                    failed_step="candidate_discovery",
                    error=f"{type(e).__name__}: {e}",
                    retryable=retryable,
                    issues=[
                        _issue(
                            "candidate_discovery",
                            f"{type(e).__name__}: {e}",
                            recoverable=retryable,
                        )
                    ],
                    started_at=plan.started_at,
                )
                P.write_checkpoint(module_paths, cp)
                await _update_module_in_manifest(
                    paths, manifest, manifest_lock, qn, cp
                )
                return cp

            P.write_candidates(module_paths, candidates)
            P.write_discovery_telemetry(module_paths, iters, dur, cost)
            cp = _now_checkpoint(
                qn=qn,
                status="DISCOVERED",
                last_step="candidate_discovery",
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)

        # SKIPPED short-circuit: zero candidates -> no step 3.
        if candidates is None:
            # Should be impossible past _plan_module if redo_step2 ran.
            cp = _now_checkpoint(
                qn=qn,
                status="FAILED",
                last_step=None,
                failed_step="candidate_discovery",
                error="internal: candidates missing after step 2",
                retryable=True,
                issues=[
                    _issue(
                        "candidate_discovery",
                        "internal: candidates missing after step 2",
                        recoverable=True,
                    )
                ],
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
            return cp

        if not candidates.candidates:
            cp = _now_checkpoint(
                qn=qn,
                status="SKIPPED",
                last_step="candidate_discovery",
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
            return cp

        # ------------------------- step 3 -----------------------------------
        run_step3 = plan.redo_step2 or plan.redo_step3 or state.deep_research is None
        if plan.redo_step3:
            P.clear_deep_research_artifacts(module_paths)

        if run_step3:
            try:
                research_output, dr_duration = await _do_step3(
                    qn=qn,
                    tree=tree,
                    mgr_input=mgr_input,
                    cfg=cfg,
                    module_paths=module_paths,
                )
            except Exception as e:  # noqa: BLE001
                cp = _now_checkpoint(
                    qn=qn,
                    status="FAILED",
                    last_step="candidate_discovery",
                    failed_step="module_deep_research",
                    error=f"{type(e).__name__}: {e}",
                    retryable=True,
                    issues=[
                        _issue(
                            "module_deep_research",
                            f"{type(e).__name__}: {e}",
                            recoverable=True,
                        )
                    ],
                    started_at=plan.started_at,
                )
                P.write_checkpoint(module_paths, cp)
                await _update_module_in_manifest(
                    paths, manifest, manifest_lock, qn, cp
                )
                return cp

            P.write_deep_research(module_paths, research_output, dr_duration)
            cp = _now_checkpoint(
                qn=qn,
                status="DEEP_RESEARCHED",
                last_step="module_deep_research",
                issues=list(research_output.issues),
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
        else:
            assert state.deep_research is not None
            research_output = state.deep_research

        final_status = _final_status_from_research(research_output)
        cp = _now_checkpoint(
            qn=qn,
            status=final_status,
            last_step=(
                "module_deep_research"
                if final_status != "FAILED"
                else "candidate_discovery"
            ),
            failed_step=(
                "module_deep_research" if final_status == "FAILED" else None
            ),
            error=(
                None
                if final_status != "FAILED"
                else "; ".join(
                    iss.message for iss in research_output.issues if not iss.recoverable
                )
            ),
            issues=list(research_output.issues),
            started_at=plan.started_at,
        )
        P.write_checkpoint(module_paths, cp)
        await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
        return cp


async def _update_module_in_manifest(
    paths: ManagerPaths,
    manifest: dict[str, Any],
    lock: asyncio.Lock,
    qn: str,
    cp: ModuleCheckpoint,
) -> None:
    async with lock:
        manifest.setdefault("modules", {})[qn] = {
            "status": cp.status,
            "last_step": cp.last_step,
        }
        P.write_manifest(paths, manifest)


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------


def run(
    input: SpotlightsManagerInput, *, config: SpotlightsManagerConfig
) -> SpotlightsManagerResult:
    return asyncio.run(_run_async(input, config=config))


async def _run_async(
    input: SpotlightsManagerInput, *, config: SpotlightsManagerConfig
) -> SpotlightsManagerResult:
    paths = ManagerPaths(config.artifacts_dir)
    _validate_setup(input, paths)

    input_fp = P.build_input_fingerprint(
        repo_path=input.repo_path,
        context=input.context,
        max_findings_per_module=input.max_findings_per_module,
        continue_on_module_failure=input.continue_on_module_failure,
    )
    config_fp = P.build_config_fingerprint(
        module_filter=config.module_filter,
        extractor_cfg=config.extractor,
        discovery_cfg=config.discovery,
        deep_research_cfg=config.deep_research,
    )
    manifest = _ensure_resume_compatible(
        paths, input_fp, config_fp, resume=config.resume
    )

    # Step 1.
    tree, invocation = _run_extractor_if_needed(input, config, paths, manifest)

    # Compute target list (dot-form keys).
    leaves: list[tuple[str, Module]] = [
        (_slash_to_dot(qn), m) for qn, m in tree.leaves()
    ]
    leaf_qns = [qn for qn, _ in leaves]
    selected = apply_filter(leaf_qns, config.module_filter)

    # Slug collision check (theoretical — dot form is unique by construction).
    seen_slugs: dict[str, str] = {}
    for qn in selected:
        slug = P.slug_for(qn)
        if slug in seen_slugs:
            raise ManagerSetupError(
                f"module slug collision: {seen_slugs[slug]!r} and {qn!r} -> {slug!r}",
                slug=slug,
            )
        seen_slugs[slug] = qn

    selected_set = set(selected)
    selected_modules: dict[str, Module] = {
        qn: m for qn, m in leaves if qn in selected_set
    }
    ordered_qns = (
        list(selected)
        if (config.module_filter and config.module_filter.include)
        else [qn for qn in leaf_qns if qn in selected_set]
    )

    sem = asyncio.Semaphore(config.max_parallel_sessions)
    manifest_lock = asyncio.Lock()
    cancel_event = (
        None if input.continue_on_module_failure else asyncio.Event()
    )

    async def _wrapped(qn: str) -> ModuleCheckpoint:
        try:
            cp = await _run_module(
                qn=qn,
                sem=sem,
                cancel_event=cancel_event,
                tree=tree,
                mgr_input=input,
                cfg=config,
                paths=paths,
                manifest_lock=manifest_lock,
                manifest=manifest,
            )
        except Exception:
            if cancel_event is not None:
                cancel_event.set()
            raise
        if (
            cancel_event is not None
            and cp.status == "FAILED"
            and not cp.retryable
        ):
            cancel_event.set()
        return cp

    tasks = [asyncio.create_task(_wrapped(qn)) for qn in ordered_qns]
    results = (
        await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
    )

    # Build the module_runs and per-module telemetry from disk so a crash
    # mid-write doesn't show as success.
    module_runs: dict[str, ModuleRun] = {}
    per_module_telemetry: dict[str, ModuleTelemetry] = {}
    any_unrecoverable = False

    for idx, qn in enumerate(ordered_qns):
        module_paths = paths.for_module(qn)
        state = P.read_module_state(module_paths)
        run_record = _assemble_module_run(
            qn,
            state,
            task_exception=(
                results[idx] if isinstance(results[idx], BaseException) else None
            ),
        )
        module_runs[qn] = run_record
        per_module_telemetry[qn] = ModuleTelemetry(
            discovery_iterations=list(state.discovery_telemetry),
            discovery_total_duration_s=state.discovery_total_duration_s,
            discovery_total_cost_usd=state.discovery_total_cost_usd,
            deep_research_duration_s=state.deep_research_duration_s,
            issues=list(run_record.issues),
        )
        if run_record.status == "FAILED" and any(
            not iss.recoverable for iss in run_record.issues
        ):
            any_unrecoverable = True

    manifest["status"] = (
        "FAILED"
        if (cancel_event is not None and cancel_event.is_set() and any_unrecoverable)
        else "COMPLETE"
    )
    P.write_manifest(paths, manifest)

    return SpotlightsManagerResult(
        project_tree=tree,
        context=input.context,
        module_runs=module_runs,
        extractor_invocation=invocation,
        per_module_telemetry=per_module_telemetry,
    )


def _assemble_module_run(
    qn: str,
    state: LoadedModuleState,
    *,
    task_exception: BaseException | None,
) -> ModuleRun:
    cp = state.checkpoint
    exception_issues: list[StepIssue] = []
    if task_exception is not None:
        step = cp.failed_step if cp and cp.failed_step else None
        if step is None:
            step = cp.last_step if cp and cp.last_step else "candidate_discovery"
        msg = f"{type(task_exception).__name__}: {task_exception}"
        exception_issues.append(_issue(step, msg, recoverable=False))

    if cp is None:
        # A task raised before any checkpoint was written. Surface a synthetic
        # FAILED record so the result is well-formed.
        issues = exception_issues or [
            _issue("candidate_discovery", "no checkpoint on disk", recoverable=False)
        ]
        return ModuleRun(
            module_qualified_name=qn,
            status="FAILED",
            candidates=None,
            findings=[],
            issues=issues,
        )

    arch_status: ModuleRunStatus
    if task_exception is not None:
        arch_status = "FAILED"
    elif cp.status in {"SUCCEEDED", "DEGRADED", "SKIPPED", "FAILED"}:
        arch_status = cp.status  # type: ignore[assignment]
    else:
        arch_status = "FAILED"

    findings = list(state.deep_research.findings) if state.deep_research else []
    issues = list(cp.issues)
    issues.extend(exception_issues)
    return ModuleRun(
        module_qualified_name=qn,
        status=arch_status,
        candidates=state.candidates,
        findings=findings,
        issues=issues,
    )


__all__ = ["run"]
