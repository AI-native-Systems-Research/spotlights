"""Async per-run orchestration.

Runs step 1 (extractor) once, then schedules per-module pipelines under a
bounded `asyncio.Semaphore`. Each per-module task wraps the existing **sync**
step entrypoints with `asyncio.to_thread` so the steps do not need to be
refactored to be async. The on-disk checkpoint tree is the single source of
truth; in-memory state is only there to drive the scheduling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from spotlights_engine.agent_proposals import (
    AgentProposalsConfig,
    AgentProposalsSetupError,
    AgentProposalsValidationError,
    create_agent_proposals_with_telemetry,
)
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
from spotlights_engine.proposal_from_finding_creator import (
    ProposalFromFindingConfig,
    ProposalFromFindingSetupError,
    ProposalFromFindingValidationError,
    create_proposals_with_telemetry,
)
from spotlights_engine.schemas.candidate import Candidate, Candidates
from spotlights_engine.schemas.common import (
    ModuleRunStatus,
    PipelineStep,
    SpotlightContext,
    StepIssue,
)
from spotlights_engine.schemas.pipeline import (
    AgentProposalsInput,
    AgentProposalsOutput,
    CandidateDiscoveryInput,
    ModuleDeepResearchInput,
    ModuleDeepResearchOutput,
    ModuleRun,
    ModulesExtractorInput,
    ProposalFromFindingCreatorInput,
    ProposalFromFindingCreatorOutput,
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


_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _issue(
    step: PipelineStep, message: str, *, recoverable: bool, severity: str = "error"
) -> StepIssue:
    return StepIssue(
        step=step,
        severity=severity,  # type: ignore[arg-type]
        message=message,
        recoverable=recoverable,
    )


def _final_status(
    research: ModuleDeepResearchOutput,
    proposals: ProposalFromFindingCreatorOutput | None,
    agent_output: AgentProposalsOutput | None = None,
) -> ModuleRunStatus:
    """Translate combined step-3 + step-4 + step-5 issues into the module status.

    - any `recoverable=False` issue -> `FAILED`
    - any issue at all (all recoverable) -> `DEGRADED`
    - no issues -> `SUCCEEDED`
    """
    issues: list[StepIssue] = list(research.issues)
    if proposals is not None:
        issues.extend(proposals.issues)
    if agent_output is not None:
        issues.extend(agent_output.issues)
    if any(not iss.recoverable for iss in issues):
        return "FAILED"
    if issues:
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


def _build_proposal_from_finding_config(
    cfg: SpotlightsManagerConfig,
    repo_path: Path,
    artifacts_dir: Path,
) -> ProposalFromFindingConfig:
    base = (
        cfg.proposal_from_finding
        if cfg.proposal_from_finding is not None
        else ProposalFromFindingConfig()
    )
    return base.model_copy(
        update={"repo_path": repo_path, "artifacts_dir": artifacts_dir}
    )


def _build_agent_proposals_config(
    cfg: SpotlightsManagerConfig,
    repo_path: Path,
    artifacts_dir: Path,
) -> AgentProposalsConfig:
    base = (
        cfg.agent_proposals
        if cfg.agent_proposals is not None
        else AgentProposalsConfig()
    )
    return base.model_copy(
        update={"repo_path": repo_path, "artifacts_dir": artifacts_dir}
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
    context: SpotlightContext,
) -> dict[str, Any]:
    existing = P.read_manifest(paths)
    if existing is None:
        return P.init_manifest(
            paths,
            input_fingerprint=input_fp,
            config_fingerprint=config_fp,
            context=context,
        )

    if not resume:
        raise ManagerSetupError(
            "spotlights_manager run dir exists and resume=False; "
            "remove the directory or pass a fresh artifacts_dir",
            run_dir=str(paths.root),
        )

    existing_schema_version = existing.get("schema_version", 1)
    if existing_schema_version != P.SCHEMA_VERSION:
        raise ResumeMismatchError(
            "manager run dir schema_version "
            f"{existing_schema_version!r} is incompatible with the current "
            f"schema_version {P.SCHEMA_VERSION!r} (qualified-name format "
            "changed); start a fresh artifacts_dir",
            existing=existing_schema_version,
            current=P.SCHEMA_VERSION,
        )

    if existing.get("input_fingerprint") != input_fp:
        raise ResumeMismatchError(
            "input fingerprint changed since the manager run dir was created",
            existing=existing.get("input_fingerprint"),
            current=input_fp,
        )

    # One-shot forward migration for pre-step-5 manifests: if the existing
    # fingerprint lacks `agent_proposals_hash`, otherwise matches the current
    # fingerprint with that key removed, AND the current `agent_proposals_hash`
    # matches the default `AgentProposalsConfig`, accept by upgrading the
    # on-disk fingerprint. A non-default step-5 config falls through to the
    # standard mismatch error since old manifests didn't record any step-5
    # knobs.
    existing_fp = existing.get("config_fingerprint") or {}
    if (
        isinstance(existing_fp, dict)
        and "agent_proposals_hash" not in existing_fp
        and "agent_proposals_hash" in config_fp
        and config_fp["agent_proposals_hash"] == P.default_agent_proposals_hash()
    ):
        current_without_step5 = {
            k: v for k, v in config_fp.items() if k != "agent_proposals_hash"
        }
        if existing_fp == current_without_step5:
            existing["config_fingerprint"] = dict(config_fp)
            P.write_manifest(paths, existing)
            existing_fp = existing["config_fingerprint"]

    if existing_fp != config_fp:
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
        prev_dur = extractor_state.get("duration_s")
        if isinstance(prev_dur, (int, float)):
            _log.info("extractor: cached (%.1fs on previous run)", prev_dur)
        else:
            _log.info("extractor: cached")
        return tree, invocation

    # Anything other than "fully complete with both sidecars on disk" forces a
    # re-run. Per plan §7.5, mid-step-1 crashes leave artifacts behind; clear
    # them so the per-step "no resume" guard is satisfied.
    P.clear_extractor_artifacts(paths)

    extractor_cfg: ExtractorConfig = cfg.extractor.model_copy(
        update={"artifacts_dir": paths.root}
    )

    _log.info("extractor: start")
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
        _log.error("extractor: failed: %s: %s", type(exc).__name__, exc)
        raise
    duration = time.monotonic() - start

    P.write_extractor_outputs(paths, result.project_tree, result.invocation)
    manifest["extractor"] = {"completed": True, "duration_s": duration}
    P.write_manifest(paths, manifest)
    n_leaves = sum(1 for _ in result.project_tree.leaves())
    _log.info(
        "extractor: complete in %.1fs — kept %d leaf modules", duration, n_leaves
    )
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
    redo_step4: bool
    redo_step5: bool
    started_at: str


_FAIL_FAST_CANCELLED_MESSAGE = (
    "module not started because another module failed and "
    "continue_on_module_failure=False"
)


def _plan_module(state: LoadedModuleState) -> _ModulePlan:
    """Resolve plan §5.4 onto a quad of booleans. The orchestrator runs
    step 3 if `redo_step3` OR step 2 just produced fresh candidates, runs
    step 4 if `redo_step4` OR step 3 just produced fresh research, and runs
    step 5 if `redo_step5` OR any earlier step just produced fresh output."""
    cp = state.checkpoint
    started_at = cp.started_at if cp is not None else _now_iso()

    def _plan(
        *, s2=False, s3=False, s4=False, s5=False, skip=False
    ) -> _ModulePlan:
        return _ModulePlan(
            skip_module=skip,
            redo_step2=s2,
            redo_step3=s3,
            redo_step4=s4,
            redo_step5=s5,
            started_at=started_at,
        )

    def _ladder_to_first_missing() -> _ModulePlan:
        if state.candidates is None:
            return _plan(s2=True)
        if state.deep_research is None:
            return _plan(s3=True)
        if state.proposal_from_finding is None:
            return _plan(s4=True)
        return _plan(s5=True)

    if cp is None:
        return _plan(s2=True)

    # Legacy pre-step-5 terminal checkpoints: SUCCEEDED/DEGRADED whose
    # `last_step` is step 4 and which never wrote agent_proposals.json. Treat
    # as "step 5 owed" so old runs acquire step 5 without redoing 1-4. Must
    # run before the generic terminal-status skip.
    if (
        cp.status in {"SUCCEEDED", "DEGRADED"}
        and cp.last_step == "proposal_from_finding_creator"
        and state.agent_proposals is None
    ):
        return _ladder_to_first_missing()

    if cp.status in {"SUCCEEDED", "DEGRADED", "SKIPPED"}:
        return _plan(skip=True)
    if cp.status == "FAILED" and not cp.retryable:
        return _plan(skip=True)

    if cp.status == "FAILED" and cp.retryable:
        if cp.failed_step == "agent_proposals":
            return _ladder_to_first_missing()
        if cp.failed_step == "proposal_from_finding_creator":
            if state.candidates is None:
                return _plan(s2=True)
            if state.deep_research is None:
                return _plan(s3=True)
            return _plan(s4=True)
        if cp.failed_step == "module_deep_research":
            if state.candidates is None:
                return _plan(s2=True)
            return _plan(s3=True)
        # Anything else (incl. unknown/None) -> redo step 2 from scratch.
        return _plan(s2=True)

    if cp.status == "PENDING":
        return _plan(s2=True)

    if cp.status == "DISCOVERED":
        if state.candidates is None:
            return _plan(s2=True)
        return _plan(s3=True)

    if cp.status == "DEEP_RESEARCHED":
        if state.candidates is None:
            return _plan(s2=True)
        if state.deep_research is None:
            return _plan(s3=True)
        return _plan(s4=True)

    if cp.status == "FINDING_PROPOSALS_CREATED":
        if state.candidates is None:
            return _plan(s2=True)
        if state.deep_research is None:
            return _plan(s3=True)
        if state.proposal_from_finding is None:
            return _plan(s4=True)
        return _plan(s5=True)

    if cp.status == "AGENT_PROPOSALS_CREATED":
        if state.candidates is None:
            return _plan(s2=True)
        if state.deep_research is None:
            return _plan(s3=True)
        if state.proposal_from_finding is None:
            return _plan(s4=True)
        if state.agent_proposals is None:
            return _plan(s5=True)
        return _plan()  # finalize-only

    return _plan(s2=True)


def _plan_requires_step_execution(plan: _ModulePlan) -> bool:
    return (
        plan.redo_step2 or plan.redo_step3 or plan.redo_step4 or plan.redo_step5
    )


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


def _candidate_at_state_finding_proposals_created(
    candidate: Candidate,
) -> Candidate:
    """Round-trip an input candidate forward to `FINDING_PROPOSALS_CREATED`
    with no proposals — the synthetic zero-findings path."""
    return candidate.model_copy(
        update={
            "state": "FINDING_PROPOSALS_CREATED",
            "deep_research_proposals": [],
            "agent_proposals": list(candidate.agent_proposals),
        }
    )


def _synthetic_step4_output_for_zero_findings(
    candidates: Candidates,
) -> ProposalFromFindingCreatorOutput:
    """Build the step-4 output the architecture mandates when step 3 found
    zero findings: every candidate advances to `FINDING_PROPOSALS_CREATED`
    with `deep_research_proposals=[]` and no issues."""
    return ProposalFromFindingCreatorOutput(
        candidates=Candidates(
            module_qualified_name=candidates.module_qualified_name,
            candidates=[
                _candidate_at_state_finding_proposals_created(c)
                for c in candidates.candidates
            ],
        ),
        issues=[],
    )


async def _do_step4(
    *,
    candidates: Candidates,
    findings,
    mgr_input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    module_paths: ModulePaths,
) -> tuple[ProposalFromFindingCreatorOutput, float, dict[str, float]]:
    """Returns (output, total_duration_s, per_pair_durations_s)."""
    pf_input = ProposalFromFindingCreatorInput(
        candidates=candidates,
        findings=list(findings),
        context=mgr_input.context,
    )
    pf_cfg = _build_proposal_from_finding_config(
        cfg, mgr_input.repo_path, module_paths.dir
    )
    result = await asyncio.to_thread(
        create_proposals_with_telemetry, pf_input, config=pf_cfg
    )
    return result.output, result.total_duration_s, dict(result.per_pair_durations_s)


async def _do_step5(
    *,
    candidates: Candidates,
    tree: ProjectTree,
    mgr_input: SpotlightsManagerInput,
    cfg: SpotlightsManagerConfig,
    module_paths: ModulePaths,
) -> tuple[AgentProposalsOutput, float, dict[str, dict[str, float]]]:
    """Returns (output, total_duration_s, per_candidate_durations_s)."""
    ap_input = AgentProposalsInput(
        project_tree=tree,
        candidates=candidates,
        context=mgr_input.context,
    )
    ap_cfg = _build_agent_proposals_config(
        cfg, mgr_input.repo_path, module_paths.dir
    )
    result = await asyncio.to_thread(
        create_agent_proposals_with_telemetry, ap_input, config=ap_cfg
    )
    return (
        result.output,
        result.total_duration_s,
        {
            cand_id: dict(durations)
            for cand_id, durations in result.per_candidate_durations_s.items()
        },
    )


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
    if (
        plan.redo_step5
        and not plan.redo_step2
        and not plan.redo_step3
        and not plan.redo_step4
    ):
        failed_step: PipelineStep = "agent_proposals"
    elif plan.redo_step4 and not plan.redo_step2 and not plan.redo_step3:
        failed_step = "proposal_from_finding_creator"
    elif plan.redo_step3 and not plan.redo_step2:
        failed_step = "module_deep_research"
    else:
        failed_step = "candidate_discovery"
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
        _log.info("[%s] skipping (already %s)", qn, state.checkpoint.status)
        return state.checkpoint

    if state.checkpoint is not None and state.checkpoint.last_step is not None:
        _log.info("[%s] resuming from %s", qn, state.checkpoint.last_step)

    module_start = time.monotonic()

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
            P.clear_proposal_from_finding_artifacts(module_paths)
            P.clear_agent_proposals_artifacts(module_paths)
            cp = _now_checkpoint(
                qn=qn,
                status="PENDING",
                last_step=None,
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)

            _log.info("[%s] discovery: start", qn)
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
                _log.error(
                    "[%s] discovery: failed: %s: %s", qn, type(e).__name__, e
                )
                _log.debug("[%s] discovery: traceback", qn, exc_info=True)
                _log.info(
                    "[%s] module FAILED in %.1fs",
                    qn,
                    time.monotonic() - module_start,
                )
                return cp

            P.write_candidates(module_paths, candidates)
            P.write_discovery_telemetry(module_paths, iters, dur, cost)
            cost_str = f" ${cost:.2f}" if cost is not None else ""
            _log.info(
                "[%s] discovery: complete in %.1fs — %d candidates%s",
                qn,
                dur,
                len(candidates.candidates),
                cost_str,
            )
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
            _log.info("[%s] discovery: 0 candidates → SKIPPED", qn)
            _log.info(
                "[%s] module SKIPPED in %.1fs",
                qn,
                time.monotonic() - module_start,
            )
            return cp

        # ------------------------- step 3 -----------------------------------
        run_step3 = plan.redo_step2 or plan.redo_step3 or state.deep_research is None
        if plan.redo_step3:
            P.clear_deep_research_artifacts(module_paths)
            P.clear_proposal_from_finding_artifacts(module_paths)
            P.clear_agent_proposals_artifacts(module_paths)

        if run_step3:
            _log.info("[%s] deep_research: start (Codex session)", qn)
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
                _log.error(
                    "[%s] deep_research: failed: %s: %s",
                    qn,
                    type(e).__name__,
                    e,
                )
                _log.debug("[%s] deep_research: traceback", qn, exc_info=True)
                _log.info(
                    "[%s] module FAILED in %.1fs",
                    qn,
                    time.monotonic() - module_start,
                )
                return cp

            _log.info(
                "[%s] deep_research: complete in %.1fs — %d findings",
                qn,
                dr_duration,
                len(research_output.findings),
            )
            for iss in research_output.issues:
                _log.warning(
                    "[%s] deep_research: %s: %s", qn, iss.severity, iss.message
                )
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

        # ------------------------- step 4 -----------------------------------
        run_step4 = (
            plan.redo_step2
            or plan.redo_step3
            or plan.redo_step4
            or state.proposal_from_finding is None
        )
        if plan.redo_step4 and not (plan.redo_step2 or plan.redo_step3):
            P.clear_proposal_from_finding_artifacts(module_paths)
            P.clear_agent_proposals_artifacts(module_paths)

        proposal_output: ProposalFromFindingCreatorOutput | None = None

        if run_step4:
            if not research_output.findings:
                # Architecture-mandated short-circuit: every candidate
                # advances to FINDING_PROPOSALS_CREATED with no proposals,
                # no Claude session is scheduled.
                _log.info(
                    "[%s] proposal_from_finding: skipped (no findings) — "
                    "synthetic empty output",
                    qn,
                )
                proposal_output = _synthetic_step4_output_for_zero_findings(
                    candidates
                )
                P.write_proposal_from_finding(
                    module_paths,
                    proposal_output,
                    duration_s=0.0,
                    per_pair_durations_s={},
                )
            else:
                n_pairs = len(candidates.candidates) * len(research_output.findings)
                _log.info(
                    "[%s] proposal_from_finding: start — %d (candidate, finding) pairs",
                    qn,
                    n_pairs,
                )
                try:
                    proposal_output, pf_duration, per_pair = await _do_step4(
                        candidates=candidates,
                        findings=research_output.findings,
                        mgr_input=mgr_input,
                        cfg=cfg,
                        module_paths=module_paths,
                    )
                except Exception as e:  # noqa: BLE001
                    if isinstance(
                        e,
                        (
                            ProposalFromFindingSetupError,
                            ProposalFromFindingValidationError,
                            ValueError,
                        ),
                    ):
                        retryable = False
                    else:
                        retryable = True
                    cp = _now_checkpoint(
                        qn=qn,
                        status="FAILED",
                        last_step="module_deep_research",
                        failed_step="proposal_from_finding_creator",
                        error=f"{type(e).__name__}: {e}",
                        retryable=retryable,
                        issues=[
                            _issue(
                                "proposal_from_finding_creator",
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
                    _log.error(
                        "[%s] proposal_from_finding: failed: %s: %s",
                        qn,
                        type(e).__name__,
                        e,
                    )
                    _log.debug(
                        "[%s] proposal_from_finding: traceback",
                        qn,
                        exc_info=True,
                    )
                    _log.info(
                        "[%s] module FAILED in %.1fs",
                        qn,
                        time.monotonic() - module_start,
                    )
                    return cp

                n_proposals = sum(
                    len(c.deep_research_proposals)
                    for c in proposal_output.candidates.candidates
                )
                _log.info(
                    "[%s] proposal_from_finding: complete in %.1fs — "
                    "%d proposals attached",
                    qn,
                    pf_duration,
                    n_proposals,
                )
                for iss in proposal_output.issues:
                    _log.warning(
                        "[%s] proposal_from_finding: %s: %s",
                        qn,
                        iss.severity,
                        iss.message,
                    )
                P.write_proposal_from_finding(
                    module_paths,
                    proposal_output,
                    duration_s=pf_duration,
                    per_pair_durations_s=per_pair,
                )

            cp = _now_checkpoint(
                qn=qn,
                status="FINDING_PROPOSALS_CREATED",
                last_step="proposal_from_finding_creator",
                issues=list(research_output.issues) + list(proposal_output.issues),
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
        else:
            assert state.proposal_from_finding is not None
            proposal_output = state.proposal_from_finding

        # ------------------------- step 5 -----------------------------------
        run_step5 = (
            plan.redo_step2
            or plan.redo_step3
            or plan.redo_step4
            or plan.redo_step5
            or state.agent_proposals is None
        )
        if plan.redo_step5 and not (
            plan.redo_step2 or plan.redo_step3 or plan.redo_step4
        ):
            P.clear_agent_proposals_artifacts(module_paths)

        agent_output: AgentProposalsOutput | None = None
        if run_step5:
            n_candidates = len(proposal_output.candidates.candidates)
            _log.info(
                "[%s] agent_proposals: start — %d candidates", qn, n_candidates
            )
            try:
                agent_output, ap_duration, per_cand = await _do_step5(
                    candidates=proposal_output.candidates,
                    tree=tree,
                    mgr_input=mgr_input,
                    cfg=cfg,
                    module_paths=module_paths,
                )
            except Exception as e:  # noqa: BLE001
                if isinstance(
                    e,
                    (
                        AgentProposalsSetupError,
                        AgentProposalsValidationError,
                        ValueError,
                    ),
                ):
                    retryable = False
                else:
                    retryable = True
                cp = _now_checkpoint(
                    qn=qn,
                    status="FAILED",
                    last_step="proposal_from_finding_creator",
                    failed_step="agent_proposals",
                    error=f"{type(e).__name__}: {e}",
                    retryable=retryable,
                    issues=[
                        _issue(
                            "agent_proposals",
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
                _log.error(
                    "[%s] agent_proposals: failed: %s: %s",
                    qn,
                    type(e).__name__,
                    e,
                )
                _log.debug(
                    "[%s] agent_proposals: traceback", qn, exc_info=True
                )
                _log.info(
                    "[%s] module FAILED in %.1fs",
                    qn,
                    time.monotonic() - module_start,
                )
                return cp

            n_agent_proposals = sum(
                len(c.agent_proposals)
                for c in agent_output.candidates.candidates
            )
            _log.info(
                "[%s] agent_proposals: complete in %.1fs — "
                "%d agent proposals attached",
                qn,
                ap_duration,
                n_agent_proposals,
            )
            for iss in agent_output.issues:
                _log.warning(
                    "[%s] agent_proposals: %s: %s",
                    qn,
                    iss.severity,
                    iss.message,
                )
            P.write_agent_proposals(
                module_paths,
                agent_output,
                duration_s=ap_duration,
                per_candidate_durations_s=per_cand,
            )
            cp = _now_checkpoint(
                qn=qn,
                status="AGENT_PROPOSALS_CREATED",
                last_step="agent_proposals",
                issues=(
                    list(research_output.issues)
                    + list(proposal_output.issues)
                    + list(agent_output.issues)
                ),
                started_at=plan.started_at,
            )
            P.write_checkpoint(module_paths, cp)
            await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
        else:
            assert state.agent_proposals is not None
            agent_output = state.agent_proposals

        # ------------------------- finalize ---------------------------------
        final_status = _final_status(research_output, proposal_output, agent_output)
        combined_issues: list[StepIssue] = list(research_output.issues)
        if proposal_output is not None:
            combined_issues.extend(proposal_output.issues)
        if agent_output is not None:
            combined_issues.extend(agent_output.issues)

        # On FAILED, prefer the latest step that actually produced an
        # unrecoverable issue. On success, last_step is the furthest step
        # whose sidecar was written.
        if final_status == "FAILED":
            if agent_output is not None and any(
                not iss.recoverable for iss in agent_output.issues
            ):
                failed_step: PipelineStep | None = "agent_proposals"
            elif proposal_output is not None and any(
                not iss.recoverable for iss in proposal_output.issues
            ):
                failed_step = "proposal_from_finding_creator"
            else:
                failed_step = "module_deep_research"
            last_step_for_cp: PipelineStep | None = (
                "agent_proposals"
                if agent_output is not None
                else "proposal_from_finding_creator"
                if proposal_output is not None
                else "module_deep_research"
            )
        else:
            failed_step = None
            last_step_for_cp = "agent_proposals"

        cp = _now_checkpoint(
            qn=qn,
            status=final_status,
            last_step=last_step_for_cp,
            failed_step=failed_step,
            error=(
                None
                if final_status != "FAILED"
                else "; ".join(
                    iss.message for iss in combined_issues if not iss.recoverable
                )
            ),
            issues=combined_issues,
            started_at=plan.started_at,
        )
        P.write_checkpoint(module_paths, cp)
        await _update_module_in_manifest(paths, manifest, manifest_lock, qn, cp)
        _log.info(
            "[%s] module %s in %.1fs",
            qn,
            final_status,
            time.monotonic() - module_start,
        )
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
    run_start = time.monotonic()
    paths = ManagerPaths(config.artifacts_dir)
    _validate_setup(input, paths)

    filter_label = (
        "include"
        if config.module_filter is not None and config.module_filter.include
        else "all"
    )
    _log.info(
        "run start: repo=%s objective=%r modules_filter=%s max_parallel=%d",
        input.repo_path,
        input.context.objective,
        filter_label,
        config.max_parallel_sessions,
    )

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
        proposal_from_finding_cfg=config.proposal_from_finding,
        agent_proposals_cfg=config.agent_proposals,
    )
    manifest = _ensure_resume_compatible(
        paths, input_fp, config_fp, resume=config.resume, context=input.context
    )

    # Step 1.
    tree, invocation = _run_extractor_if_needed(input, config, paths, manifest)

    # Compute target list. Qualified names are slash-form (source-root-relative
    # paths) and are the single canonical key throughout — `module_runs`, slugs,
    # the CLI filter, and `EvolveSpec` all key on this string.
    leaves: list[tuple[str, Module]] = list(tree.leaves())
    leaf_qns = [qn for qn, _ in leaves]
    selected = apply_filter(leaf_qns, config.module_filter)

    # Slug collision check (theoretical — slash form is unique by construction).
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
            proposal_from_finding_duration_s=state.proposal_from_finding_duration_s,
            proposal_from_finding_per_pair_durations_s=dict(
                state.proposal_from_finding_per_pair_durations_s
            ),
            agent_proposals_duration_s=state.agent_proposals_duration_s,
            agent_proposals_per_candidate_durations_s={
                cand_id: dict(durations)
                for cand_id, durations in state.agent_proposals_per_candidate_durations_s.items()
            },
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

    renderer_result, manager_issues = _render_results(
        artifacts_dir=paths.artifacts_dir,
        output_folder=config.output_folder,
    )

    counts = {"SUCCEEDED": 0, "DEGRADED": 0, "FAILED": 0, "SKIPPED": 0}
    for run_record in module_runs.values():
        counts[run_record.status] = counts.get(run_record.status, 0) + 1
    total_cost = sum(
        (
            t.discovery_total_cost_usd or 0.0
            for t in per_module_telemetry.values()
        )
    )
    cost_str = f", discovery cost ${total_cost:.2f}" if total_cost else ""
    _log.info(
        "run complete: %d succeeded / %d degraded / %d failed / %d skipped "
        "in %.1fs%s",
        counts["SUCCEEDED"],
        counts["DEGRADED"],
        counts["FAILED"],
        counts["SKIPPED"],
        time.monotonic() - run_start,
        cost_str,
    )

    return SpotlightsManagerResult(
        project_tree=tree,
        context=input.context,
        module_runs=module_runs,
        extractor_invocation=invocation,
        per_module_telemetry=per_module_telemetry,
        manager_issues=manager_issues,
        renderer_result=renderer_result,
    )


def _render_results(
    *, artifacts_dir: Path, output_folder: Path
) -> tuple[Any, list[StepIssue]]:
    """Step 6 — invoke the results renderer and translate failures into a
    manager-level `StepIssue`. Returns `(RendererResult | None, issues)`."""
    # Late import to avoid an import cycle (results_renderer imports
    # spotlights_manager.persistence).
    from spotlights_engine.results_renderer import (
        RendererInput,
        RendererSetupError,
        render,
    )

    issues: list[StepIssue] = []
    _log.info("renderer: start")
    try:
        result = render(
            RendererInput(
                artifacts_dir=artifacts_dir,
                output_folder=output_folder,
            )
        )
        _log.info("renderer: complete — %s", result.index_path)
        return result, issues
    except RendererSetupError as exc:
        issues.append(
            _issue(
                "results_renderer",
                f"results_renderer skipped: {type(exc).__name__}: {exc}",
                recoverable=True,
                severity="warning",
            )
        )
        _log.warning(
            "renderer: skipped: %s: %s", type(exc).__name__, exc
        )
        return None, issues
    except Exception as exc:  # noqa: BLE001
        issues.append(
            _issue(
                "results_renderer",
                f"results_renderer failed: {type(exc).__name__}: {exc}",
                recoverable=True,
                severity="error",
            )
        )
        _log.error("renderer: failed: %s: %s", type(exc).__name__, exc)
        _log.debug("renderer: traceback", exc_info=True)
        return None, issues


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
    # Prefer the post-step-5 candidates (state advanced through agent_proposals).
    # Fall back to the post-step-4 view, then the step-2 view, when later
    # steps never produced output.
    if state.agent_proposals is not None:
        candidates = state.agent_proposals.candidates
    elif state.proposal_from_finding is not None:
        candidates = state.proposal_from_finding.candidates
    else:
        candidates = state.candidates
    return ModuleRun(
        module_qualified_name=qn,
        status=arch_status,
        candidates=candidates,
        findings=findings,
        issues=issues,
    )


__all__ = ["run"]
