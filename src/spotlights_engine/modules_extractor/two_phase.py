"""Two-phase module extraction (Idea 4): deterministic skeleton, LLM enrichment.

Orchestrates the four sequential stages described in
`design/module_extraction_fix_impl_plan.md` (the pipeline historically had a
Stage 4 semantic review, since removed; assembly keeps its Stage-5 numbering):

1. Claude + Python — repository metadata and source root.
2. Python — deterministic directory inventory / skeleton.
3. Claude + Python — Stage 3A labels every skeleton path `MODULE`/`PART`
   (single call or sharded, with V1–V3 + path-only V5 validation and bounded
   repairs), code resolves owners, then Stage 3B fetches descriptions and main
   files for final module territories in deterministic batches (V4 + final V5).
5. Python — deterministic final `ProjectTree` assembly and re-validation.

The public return type (`ExtractionRunResult`) and the serialized `ProjectTree`
schema are unchanged. When `artifacts_dir` is set, every stage persists its
inputs and outputs *as it runs*, so a finished (or failed) run is fully
reconstructable from disk. An artifact-write failure on an otherwise successful
path raises `ExtractorArtifactError`; a write failure while handling an existing
stage error is logged and never masks the original error.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from spotlights_engine.modules_extractor.agent import (
    ExtractionInvocation,
    ExtractionRunResult,
)
from spotlights_engine.modules_extractor.assignments import (
    AssignmentIssue,
    CollisionPrecedence,
    compute_assignment_coverage,
    compute_collision_precedence,
    compute_metadata_coverage,
    format_assignment_issues,
    validate_assignment_labels,
    validate_assignment_paths,
    validate_metadata_entries,
)
from spotlights_engine.modules_extractor.claude_stage import (
    ClaudeStageResult,
    StageTelemetry,
    add_optional_float,
    add_optional_int,
    atomic_write_json,
    atomic_write_text,
    notify,
    run_structured_claude_stage,
)
from spotlights_engine.modules_extractor.coverage import (
    CrossArtifactError,
    forced_repository_level_files,
    validate_source_root_decision,
)
from spotlights_engine.modules_extractor.derive import (
    compute_assignment_lints,
    derive_metadata_batches,
    derive_module_forest,
    derive_project_tree,
    resolve_assignments,
    verify_resolved_assignments,
)
from spotlights_engine.modules_extractor.errors import (
    ExtractorAgentError,
    ExtractorArtifactError,
    ExtractorCoverageError,
    ExtractorValidationError,
)
from spotlights_engine.modules_extractor.prompts import (
    render_assignment_prompt,
    render_assignment_shard_prompt,
    render_identify_source_root_prompt,
    render_metadata_prompt,
)
from spotlights_engine.modules_extractor.sharding import (
    AssignmentShard,
    AssignmentShardPlan,
    assignment_branch_coverage,
    assignment_owning_shard,
    assignment_shard_weight,
    covers_entire_skeleton,
    derive_assignment_shards,
    has_several_source_roots,
    merge_assignment_fragments,
    render_assignment_shard_plan_markdown,
    validate_assignment_shard_scope,
)
from spotlights_engine.modules_extractor.skeleton import (
    ALGORITHM_PROFILE_VERSION,
    build_skeleton,
    compute_fingerprint,
    scan_source_files,
)
from spotlights_engine.modules_extractor.stage_schemas import (
    AssignmentTree,
    ExcludedSourcePath,
    ModuleInfo,
    ModuleMetadataBatch,
    ModuleMetadataMap,
    ResolvedAssignmentTree,
    Skeleton,
    SourceRootDecision,
)
from spotlights_engine.modules_extractor.tree_report import (
    build_tree_decision_report_v2,
    render_markdown_v2,
)
from spotlights_engine.schemas.project import (
    ProjectTree,
)

if TYPE_CHECKING:
    from spotlights_engine.modules_extractor.derive import (
        AssignmentLint,
        MetadataBatch,
    )
    from spotlights_engine.modules_extractor.extractor import ExtractorConfig

_STAGE_REPAIR_PAYLOAD_MAX_CHARS = 60_000

# Stage-3 attempt budget: the initial call plus at most one repair per
# *distinct* failure class (parse / validation / coverage), capped here. The
# cap exists because a repair can trade one failure class for another — batch
# evidence showed repairs that fixed validation while regressing coverage —
# and a shared single repair then hard-fails a tree that one more targeted
# repair would have completed. A class that fails twice still fails fast: the
# second occurrence raises rather than repairing again.
_MAX_STAGE3_ATTEMPTS = 3


def _spend_repair(kind: str, repaired_kinds: set[str], attempt: int) -> bool:
    """True when a repair for `kind` may run: the class hasn't been repaired
    yet and the attempt budget allows one more call. Mutates `repaired_kinds`."""
    if kind in repaired_kinds or attempt >= _MAX_STAGE3_ATTEMPTS:
        return False
    repaired_kinds.add(kind)
    return True

_T = TypeVar("_T")


def _run_event_loop(coro: Coroutine[Any, Any, _T]) -> _T:
    """Drive `coro` to completion from synchronous code.

    `run_two_phase_extraction` is a synchronous API, but not every caller
    reaches it from a bare thread: `spotlights_manager.orchestrator` runs step 1
    from *inside* `asyncio.run(_run_async(...))`. `asyncio.run` refuses to nest
    (`RuntimeError: asyncio.run() cannot be called from a running event loop`),
    so when this thread already owns a running loop the coroutine is handed to a
    dedicated worker thread with a loop of its own. The calling thread blocks
    either way, which is exactly what a synchronous entrypoint promises.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="extractor-shards"
    ) as pool:
        return pool.submit(asyncio.run, coro).result()


# ── Telemetry accumulation ────────────────────────────────────────────────


@dataclass
class _StageSession:
    stage: str
    provider: str
    session_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read: int | None
    cache_create: int | None
    cost_usd: float | None
    duration_s: float


@dataclass
class _Telemetry:
    sessions: list[_StageSession] = field(default_factory=list)
    total_duration_s: float = 0.0
    total_cost_usd: float | None = None
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    accepted_session_id: str | None = None

    def add_stage(self, stage: str, t: StageTelemetry) -> None:
        usage = t.usage
        self.sessions.append(
            _StageSession(
                stage=stage,
                provider="claude",
                session_id=t.session_id,
                input_tokens=t.input_tokens,
                output_tokens=t.output_tokens,
                cache_read=usage.cache_read if usage is not None else None,
                cache_create=usage.cache_create if usage is not None else None,
                cost_usd=t.cost_usd,
                duration_s=t.duration_s,
            )
        )
        self.total_duration_s += t.duration_s
        self.total_cost_usd = add_optional_float(self.total_cost_usd, t.cost_usd)
        self.total_input_tokens = add_optional_int(
            self.total_input_tokens, t.input_tokens
        )
        self.total_output_tokens = add_optional_int(
            self.total_output_tokens, t.output_tokens
        )

    def add_claude(
        self, stage: str, result: ClaudeStageResult[Any]
    ) -> None:
        self.add_stage(stage, result.telemetry)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted_session_id": self.accepted_session_id,
            "total_duration_s": self.total_duration_s,
            "total_cost_usd": self.total_cost_usd,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "sessions": [
                {
                    "stage": s.stage,
                    "provider": s.provider,
                    "session_id": s.session_id,
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                    "cache_read": s.cache_read,
                    "cache_create": s.cache_create,
                    "cost_usd": s.cost_usd,
                    "duration_s": s.duration_s,
                }
                for s in self.sessions
            ],
        }


# ── Artifact persistence helpers ──────────────────────────────────────────


def _required_write_json(base: Path | None, rel: str, obj: object) -> None:
    """Write `obj` on an otherwise-successful path. A failure here is itself an
    extraction failure (`ExtractorArtifactError`)."""
    if base is None:
        return
    try:
        atomic_write_json(base / rel, obj)
    except OSError as exc:
        raise ExtractorArtifactError(
            f"failed to write required artifact {rel}: {exc}", artifact=rel
        ) from exc


def _required_write_text(base: Path | None, rel: str, text: str) -> None:
    if base is None:
        return
    try:
        atomic_write_text(base / rel, text)
    except OSError as exc:
        raise ExtractorArtifactError(
            f"failed to write required artifact {rel}: {exc}", artifact=rel
        ) from exc


def _best_effort_write_json(
    base: Path | None, rel: str, obj: object, on_event: Callable | None
) -> None:
    """Write during error handling: never mask the in-flight stage error."""
    if base is None:
        return
    try:
        atomic_write_json(base / rel, obj)
    except OSError as exc:  # noqa: BLE001 - must not mask the original error
        notify(on_event, f"extractor: failed to persist {rel} during error handling: {exc}")


def _best_effort_write_text(
    base: Path | None, rel: str, text: str, on_event: Callable | None
) -> None:
    """Write during error handling: never mask the in-flight stage error."""
    if base is None:
        return
    try:
        atomic_write_text(base / rel, text)
    except OSError as exc:  # noqa: BLE001 - must not mask the original error
        notify(on_event, f"extractor: failed to persist {rel} during error handling: {exc}")


def _persist_sessions(base: Path | None, telemetry: _Telemetry) -> None:
    """sessions.json is rewritten atomically after every invocation."""
    _required_write_json(base, "sessions.json", telemetry.as_dict())


# ── Public entrypoint ─────────────────────────────────────────────────────


def run_two_phase_extraction(
    repo_path: Path,
    *,
    config: ExtractorConfig,
    on_event: Callable[[str], None] | None,
    artifacts_dir: Path | None,
) -> ExtractionRunResult:
    """Run the full four-stage two-phase extraction over `repo_path`.

    `artifacts_dir` is the `modules_extractor/` run directory (already created
    by the caller) or `None` to disable persistence.
    """
    base = artifacts_dir
    telemetry = _Telemetry()
    _required_write_json(
        base,
        "extractor_config.json",
        config.model_dump(mode="json", exclude={"artifacts_dir"}),
    )

    # Repo-wide safe source scan + fingerprint, shared by Stage 1 and Stage 5.
    repo_scan = scan_source_files(repo_path, "")
    repo_fingerprint = compute_fingerprint(
        repo_path, repo_scan.files, extra_paths=repo_scan.skipped_symlinks
    )

    # ── Stage 1 — source-root decision ────────────────────────────────────
    decision = _stage1_source_root(
        repo_path,
        base=base,
        config=config,
        telemetry=telemetry,
        detected_source_files=repo_scan.files,
        repo_fingerprint=repo_fingerprint,
        on_event=on_event,
    )
    repository = decision.repository
    _persist_sessions(base, telemetry)

    # ── Stage 2 — deterministic skeleton ──────────────────────────────────
    skeleton = _stage2_skeleton(
        repo_path, decision, base=base, on_event=on_event
    )

    # ── Stage 3A/3B + deterministic Stage 5 assembly ─────────────────────
    tree = _run_assignment_extraction(
        repo_path,
        repository,
        decision,
        skeleton,
        base=base,
        config=config,
        telemetry=telemetry,
        repo_fingerprint=repo_fingerprint,
        on_event=on_event,
    )
    _persist_sessions(base, telemetry)
    invocation = ExtractionInvocation(
        session_id=telemetry.accepted_session_id,
        duration_s=telemetry.total_duration_s,
        cost_usd=telemetry.total_cost_usd,
        input_tokens=telemetry.total_input_tokens,
        output_tokens=telemetry.total_output_tokens,
    )
    return ExtractionRunResult(
        project_tree=tree,
        invocation=invocation,
        raw_payload=tree.model_dump_json(indent=2),
    )

# ── Stage 1 ───────────────────────────────────────────────────────────────


def _inject_forced_repository_level_files(
    decision: SourceRootDecision, detected_source_files: list[str]
) -> SourceRootDecision:
    """Deterministically add the mechanically-forced `repository_level_file`
    exclusions the Stage-1 model may have missed.

    A source file sitting directly at the selected root can only ever be
    excluded (the directory-only schema cannot represent it), so requiring the
    model to enumerate every one is fragile — the real vLLM run failed because a
    single root-level script (`build_vllm_ppc64le.sh`) was left unclassified.
    We compute that forced set and merge in only the files not already covered
    by an existing exclusion (the model's own classifications, and ancestor
    directory exclusions, are preserved). Semantic exclusions stay the model's
    job. The result is idempotent: files already excluded are never re-added.
    """
    source_root = decision.repository.source_root
    forced = forced_repository_level_files(source_root, detected_source_files)
    if not forced:
        return decision

    existing = {e.path for e in decision.excluded_source_paths}
    covered_by_dir = [e.path for e in decision.excluded_source_paths]

    def _already_covered(path: str) -> bool:
        if path in existing:
            return True
        return any(path.startswith(ex + "/") for ex in covered_by_dir)

    additions = [
        ExcludedSourcePath(
            path=path,
            reason="repository_level_file",
            explanation=(
                "source file directly at the source root; the directory-only "
                "module schema cannot represent it (auto-classified)"
            ),
        )
        for path in forced
        if not _already_covered(path)
    ]
    if not additions:
        return decision

    return decision.model_copy(
        update={
            "excluded_source_paths": [
                *decision.excluded_source_paths,
                *additions,
            ]
        }
    )


def _stage1_source_root(
    repo_path: Path,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    detected_source_files: list[str],
    repo_fingerprint: str,
    on_event: Callable[[str], None] | None,
) -> SourceRootDecision:
    stage_dir = base / "01_source_root" if base is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = render_identify_source_root_prompt()
    if stage_dir is not None:
        _required_write_text(base, "01_source_root/prompt.md", base_prompt)

    prompt = base_prompt
    last_errors: str | None = None
    for attempt in range(1, 3):  # initial + one repair
        attempt_dir = (
            stage_dir / f"attempt_{attempt:02d}" if stage_dir is not None else None
        )
        if attempt == 2 and last_errors is not None:
            prompt = _repair_prompt(base_prompt, last_errors, previous_payload=None)

        result = run_structured_claude_stage(
            output_type=SourceRootDecision,
            repo_path=repo_path,
            prompt=prompt,
            stage_name="source_root",
            attempt_dir=attempt_dir,
            claude_bin=config.claude_bin,
            claude_model=config.claude_model,
            max_turns=config.source_root_max_turns,
            timeout_s=_effective_timeout(config.source_root_timeout_s, config.timeout_s),
            on_event=on_event,
        )
        telemetry.add_claude("01_source_root", result)
        _persist_sessions(base, telemetry)

        if result.parsed is None:
            last_errors = str(result.validation_error)
            _write_attempt_validation(
                attempt_dir, ok=False, kind="pydantic", detail=last_errors
            )
            if attempt == 1:
                notify(on_event, "extractor: stage-1 parse failed; one repair")
                continue
            raise ExtractorValidationError(
                f"source-root decision failed to parse after 2 attempts: "
                f"{result.validation_error}",
                stage="source_root",
            )

        decision = _inject_forced_repository_level_files(
            result.parsed, detected_source_files
        )
        try:
            validate_source_root_decision(decision, repo_path, detected_source_files)
        except CrossArtifactError as exc:
            last_errors = str(exc)
            _write_attempt_validation(
                attempt_dir, ok=False, kind="filesystem", detail=last_errors
            )
            if attempt == 1:
                notify(on_event, "extractor: stage-1 filesystem validation failed; one repair")
                continue
            raise ExtractorValidationError(
                f"source-root decision failed filesystem validation: {exc}",
                stage="source_root",
            ) from exc

        _write_attempt_validation(attempt_dir, ok=True, kind=None, detail=None)
        _required_write_json(
            base,
            "01_source_root/validation.json",
            {
                "ok": True,
                "repo_fingerprint": repo_fingerprint,
                "algorithm_profile": ALGORITHM_PROFILE_VERSION,
                "detected_source_files": sorted(detected_source_files),
            },
        )
        _required_write_json(
            base, "01_source_root/decision.json", decision.model_dump(mode="json")
        )
        _required_write_json(
            base, "source_root_decision.json", decision.model_dump(mode="json")
        )
        return decision

    raise AssertionError("unreachable: stage-1 attempt loop exhausted")


def _write_attempt_validation(
    attempt_dir: Path | None, *, ok: bool, kind: str | None, detail: str | None
) -> None:
    if attempt_dir is None:
        return
    try:
        atomic_write_json(
            attempt_dir / "validation.json",
            {"ok": ok, "kind": kind, "detail": detail},
        )
    except OSError:
        pass


# ── Stage 2 ───────────────────────────────────────────────────────────────


def _split_exclusions(
    repo_path: Path, decision: SourceRootDecision
) -> tuple[frozenset[str], frozenset[str]]:
    dirs: set[str] = set()
    files: set[str] = set()
    for ex in decision.excluded_source_paths:
        if (repo_path / ex.path).is_dir():
            dirs.add(ex.path)
        else:
            files.add(ex.path)
    return frozenset(dirs), frozenset(files)


def _stage2_skeleton(
    repo_path: Path,
    decision: SourceRootDecision,
    *,
    base: Path | None,
    on_event: Callable[[str], None] | None,
) -> Skeleton:
    excluded_dirs, excluded_files = _split_exclusions(repo_path, decision)
    source_root = decision.repository.source_root
    skeleton = build_skeleton(
        repo_path,
        source_root,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
    )
    _required_write_json(
        base,
        "02_skeleton/inputs.json",
        {
            "source_root": source_root,
            "excluded_dirs": sorted(excluded_dirs),
            "excluded_files": sorted(excluded_files),
            "algorithm_profile": ALGORITHM_PROFILE_VERSION,
        },
    )
    _required_write_json(
        base, "02_skeleton/skeleton.json", skeleton.model_dump(mode="json")
    )
    _required_write_json(
        base,
        "02_skeleton/walk_audit.json",
        {
            "ignored": skeleton.ignored,
            "excluded": skeleton.excluded,
            "organizational_only": skeleton.organizational_only,
            "skipped_symlinks": skeleton.skipped_symlinks,
        },
    )
    _required_write_json(base, "skeleton.json", skeleton.model_dump(mode="json"))
    notify(
        on_event,
        f"extractor: skeleton — {len(skeleton.required_paths())} required, "
        f"{len(skeleton.all_paths())} inventoried",
    )
    return skeleton


# ── Stage 3 ───────────────────────────────────────────────────────────────


def _effective_timeout(specific: int | None, fallback: int) -> int:
    """`None` on a stage-specific deadline means "inherit `timeout_s`"."""
    return specific if specific is not None else fallback


class _CancelFlag:
    """Stop queued work after the first concurrent shard/batch failure."""

    def __init__(self) -> None:
        self.cancelled = False


def _verify_repo_unchanged(
    repo_path: Path,
    decision: SourceRootDecision,
    skeleton: Skeleton,
    *,
    repo_fingerprint: str,
) -> None:
    """Stage-5 concurrent-mutation defenses:
    re-run Stage-1 exclusion validation on a fresh scan, compare the repo-wide
    content fingerprint, rebuild the skeleton, and compare its fingerprint."""
    fresh_scan = scan_source_files(repo_path, "")
    try:
        validate_source_root_decision(decision, repo_path, fresh_scan.files)
    except CrossArtifactError as exc:
        raise ExtractorValidationError(
            "source-root decision no longer validates at assembly "
            f"(repository changed during extraction?): {exc}",
            stage="assemble",
        ) from exc
    fresh_repo_fingerprint = compute_fingerprint(
        repo_path, fresh_scan.files, extra_paths=fresh_scan.skipped_symlinks
    )
    if fresh_repo_fingerprint != repo_fingerprint:
        raise ExtractorValidationError(
            "repository source content changed during extraction "
            "(repo-wide fingerprint mismatch)",
            stage="assemble",
        )

    excluded_dirs, excluded_files = _split_exclusions(repo_path, decision)
    fresh_skeleton = build_skeleton(
        repo_path,
        decision.repository.source_root,
        excluded_dirs=excluded_dirs,
        excluded_files=excluded_files,
    )
    if fresh_skeleton.inventory_fingerprint != skeleton.inventory_fingerprint:
        raise ExtractorValidationError(
            "repository inventory changed during extraction "
            "(skeleton fingerprint mismatch)",
            stage="assemble",
        )


def _bounded(text: str) -> str:
    if len(text) > _STAGE_REPAIR_PAYLOAD_MAX_CHARS:
        return text[:_STAGE_REPAIR_PAYLOAD_MAX_CHARS] + "\n...<truncated>"
    return text


def _repair_prompt(
    base_prompt: str, errors: str, *, previous_payload: str | None
) -> str:
    parts = [
        base_prompt,
        "\n\n## Repair after local validation failure\n",
        "Your previous output was rejected. Fix the exact problems below and "
        "return ONE complete valid JSON object in the same schema.\n\n",
        "Errors:\n",
        _bounded(errors),
    ]
    if previous_payload is not None:
        parts.append("\n\nPrevious output:\n")
        parts.append(_bounded(previous_payload))
    return "".join(parts) + "\n"


# ══════════════════════════════════════════════════════════════════════════
# Stage 3 — assignments and metadata
# ══════════════════════════════════════════════════════════════════════════


def _assignment_coverage_repair_errors(missing: list[str]) -> str:
    """Coverage-repair message containing the missing assignment labels."""
    return (
        "Missing assignment labels for these skeleton paths — `assignments` "
        "must contain every SKELETON path exactly once: " + ", ".join(missing)
    )


def _write_tree_decisions_v2(
    base: Path | None,
    *,
    required: bool,
    skeleton: Skeleton,
    assignments: AssignmentTree | None,
    resolved: ResolvedAssignmentTree | None,
    metadata: dict[str, ModuleInfo] | None,
    coverage: Any,
    lints: list[AssignmentLint] | None,
    issues: list[AssignmentIssue] | None,
    decision: SourceRootDecision | None,
    merge_threshold: int,
    on_event: Callable[[str], None] | None,
) -> None:
    """The v2 decision report, on the success (`required=True`) or failure
    (best-effort) path. The best-effort form must never mask the in-flight
    stage error, so build + serialize stay inside the guard."""
    if base is None:
        return
    try:
        report = build_tree_decision_report_v2(
            skeleton,
            assignments=assignments,
            resolved=resolved,
            metadata=metadata,
            coverage=coverage,
            lints=lints,
            issues=issues,
            decision=decision,
            merge_threshold=merge_threshold,
        )
        payload = report.model_dump(mode="json")
        markdown = render_markdown_v2(report)
    except Exception as exc:  # noqa: BLE001 - must not mask the original error
        if required:
            raise
        notify(
            on_event,
            f"extractor: failed to build tree_decisions during error handling: {exc}",
        )
        return
    if required:
        _required_write_json(base, "tree_decisions.json", payload)
        _required_write_text(base, "tree_decisions.md", markdown)
    else:
        _best_effort_write_json(base, "tree_decisions.json", payload, on_event)
        _best_effort_write_text(base, "tree_decisions.md", markdown, on_event)


# ── Stage 3A — single call ────────────────────────────────────────────────


def _stage3a_single(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    precedence: CollisionPrecedence,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    decision: SourceRootDecision | None,
    on_event: Callable[[str], None] | None,
) -> AssignmentTree:
    """One whole-repository assignment call with the bounded repair budget.

    V2 (totality) is a preflight with precedence over V1/V3/V5: a missing
    label plus its now-orphan decision must not consume both the coverage and
    validation repairs. Once totality holds, V1, V3, and the path-only V5
    (`derive_module_forest`) run together.
    """
    repo_data = repository.model_dump(mode="json")
    skel_data = skeleton.model_dump(mode="json")
    scope = {
        "whole_repository": True,
        "root_path": skeleton.source_root,
        "forced_part_paths": sorted(precedence.forced_part),
    }
    base_prompt = render_assignment_prompt(
        repository=repo_data,
        skeleton=skel_data,
        scope=scope,
        merge_threshold=config.merge_threshold,
    )
    _required_write_text(base, "03_enrich/prompt.md", base_prompt)
    _required_write_json(
        base,
        "03_enrich/data_blocks.json",
        {"repository": repo_data, "skeleton": skel_data, "scope": scope},
    )

    anchor_paths = {n.path for n in skeleton.nodes}
    prompt = base_prompt
    repaired_kinds: set[str] = set()
    last_parsed: AssignmentTree | None = None
    last_issues: list[AssignmentIssue] | None = None
    for attempt in range(1, _MAX_STAGE3_ATTEMPTS + 1):
        attempt_rel = f"03_enrich/attempt_{attempt:02d}"
        attempt_dir = base / attempt_rel if base is not None else None
        result: ClaudeStageResult[AssignmentTree] = run_structured_claude_stage(
            output_type=AssignmentTree,
            repo_path=repo_path,
            prompt=prompt,
            stage_name="assign",
            attempt_dir=attempt_dir,
            claude_bin=config.claude_bin,
            claude_model=config.claude_model,
            max_turns=config.max_turns,
            timeout_s=config.timeout_s,
            on_event=on_event,
        )
        telemetry.add_claude("03_enrich", result)
        _persist_sessions(base, telemetry)

        if result.parsed is None:
            detail = str(result.validation_error)
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {"ok": False, "kind": "parse", "detail": detail},
            )
            if _spend_repair("parse", repaired_kinds, attempt):
                notify(on_event, "extractor: stage-3A parse failed; repairing")
                prompt = _repair_prompt(
                    base_prompt, detail, previous_payload=result.raw_payload
                )
                continue
            if last_parsed is not None:
                _write_tree_decisions_v2(
                    base,
                    required=False,
                    skeleton=skeleton,
                    assignments=last_parsed,
                    resolved=None,
                    metadata=None,
                    coverage=compute_assignment_coverage(
                        last_parsed, skeleton, invalid=last_issues
                    ),
                    lints=None,
                    issues=last_issues,
                    decision=decision,
                    merge_threshold=config.merge_threshold,
                    on_event=on_event,
                )
            raise ExtractorValidationError(
                f"assignment tree failed to parse after {attempt} attempts: "
                f"{result.validation_error}",
                stage="enrich",
            )

        parsed = result.parsed
        last_parsed = parsed
        last_issues = None
        _required_write_json(
            base,
            f"{attempt_rel}/assignment_tree.json",
            parsed.model_dump(mode="json"),
        )

        # V2 preflight — totality before any label semantics.
        coverage = compute_assignment_coverage(parsed, skeleton)
        _required_write_json(
            base, f"{attempt_rel}/coverage.json", coverage.model_dump(mode="json")
        )
        if coverage.missing:
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {
                    "ok": False,
                    "kind": "coverage",
                    "detail": f"{len(coverage.missing)} unlabeled inventory path(s)",
                },
            )
            if _spend_repair("coverage", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: stage-3A missing {len(coverage.missing)} "
                    "labels; repairing",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    _assignment_coverage_repair_errors(coverage.missing),
                    previous_payload=result.raw_payload,
                )
                continue
            _write_tree_decisions_v2(
                base,
                required=False,
                skeleton=skeleton,
                assignments=parsed,
                resolved=None,
                metadata=None,
                coverage=coverage,
                lints=None,
                issues=None,
                decision=decision,
                merge_threshold=config.merge_threshold,
                on_event=on_event,
            )
            raise ExtractorCoverageError(
                f"{len(coverage.missing)} inventory paths unlabeled",
                missing=coverage.missing,
                stage="enrich",
            )

        issues = validate_assignment_paths(parsed, skeleton, repo_path)
        issues += validate_assignment_labels(
            parsed,
            skeleton,
            merge_threshold=config.merge_threshold,
            precedence=precedence,
            anchor_paths=anchor_paths,
            require_owner_in_scope=True,
        )
        if not issues:
            try:
                derive_module_forest(repository, parsed)
            except ValueError as exc:
                issues.append(
                    AssignmentIssue(
                        code="public_tree_invalid",
                        detail=f"provisional public tree failed validation: {exc}",
                    )
                )
        last_issues = issues or None
        _required_write_json(
            base,
            f"{attempt_rel}/validation.json",
            {
                "ok": not issues,
                "kind": "validation" if issues else None,
                "issues": [i.model_dump(mode="json") for i in issues],
            },
        )
        if issues:
            if _spend_repair("validation", repaired_kinds, attempt):
                notify(on_event, "extractor: stage-3A validation failed; repairing")
                prompt = _repair_prompt(
                    base_prompt,
                    format_assignment_issues(issues),
                    previous_payload=result.raw_payload,
                )
                continue
            _write_tree_decisions_v2(
                base,
                required=False,
                skeleton=skeleton,
                assignments=parsed,
                resolved=None,
                metadata=None,
                coverage=compute_assignment_coverage(
                    parsed, skeleton, invalid=issues
                ),
                lints=None,
                issues=issues,
                decision=decision,
                merge_threshold=config.merge_threshold,
                on_event=on_event,
            )
            raise ExtractorValidationError(
                "assignment tree failed validation: "
                f"{format_assignment_issues(issues)}",
                stage="enrich",
            )

        telemetry.accepted_session_id = result.telemetry.session_id
        return parsed

    raise AssertionError("unreachable: stage-3A attempt loop exhausted")


# ── Stage 3A — sharded ────────────────────────────────────────────────────


@dataclass
class _AssignmentShardOutcome:
    shard: AssignmentShard
    fragment: AssignmentTree
    session_id: str | None
    attempts: int


@dataclass
class _AssignmentFailureSnapshot:
    fragment: AssignmentTree
    issues: list[AssignmentIssue] | None


def _assignment_scope_dict(
    shard: AssignmentShard, precedence: CollisionPrecedence
) -> dict[str, Any]:
    in_scope = shard.subtree.all_paths()
    return {
        "key": shard.key,
        "root_path": shard.root_path,
        "is_branch_root": shard.is_branch_root,
        "is_subshard": shard.is_subshard,
        "parent_key": shard.parent_key,
        "depth": shard.depth,
        "promoted_children": list(shard.promoted_children),
        "promotion_parent": shard.promotion_parent,
        # Full-skeleton-derived collision policy, restricted to this scope so
        # precedence is identical under every partition.
        "forced_part_paths": sorted(
            p for p in precedence.forced_part if p in in_scope
        ),
    }


def _assignment_shard_plan_dict(plan: AssignmentShardPlan) -> dict[str, Any]:
    return {
        "branch_roots": dict(plan.branch_roots),
        "not_split_reasons": dict(plan.not_split_reasons),
        "shards": [
            {
                "key": s.key,
                "root_path": s.root_path,
                "is_branch_root": s.is_branch_root,
                "is_subshard": s.is_subshard,
                "parent_key": s.parent_key,
                "depth": s.depth,
                "promoted_children": list(s.promoted_children),
                "promotion_parent": s.promotion_parent,
                "inventory_nodes": len(s.subtree.all_paths()),
            }
            for s in plan.shards
        ],
    }


def _validate_assignment_fragment(
    shard: AssignmentShard,
    fragment: AssignmentTree,
    *,
    repo_path: Path,
    repository: Any,
    precedence: CollisionPrecedence,
    merge_threshold: int,
) -> list[AssignmentIssue]:
    """Locally decidable validation for one fragment: scope keys, V1, V3 (a
    nested root may be `PART`; owners resolve after union), and path-only V5."""
    issues: list[AssignmentIssue] = []
    try:
        validate_assignment_shard_scope(shard, fragment)
    except CrossArtifactError as exc:
        issues.append(AssignmentIssue(code="out_of_scope", detail=str(exc)))
    issues += validate_assignment_paths(fragment, shard.subtree, repo_path)
    issues += validate_assignment_labels(
        fragment,
        shard.subtree,
        merge_threshold=merge_threshold,
        precedence=precedence,
        anchor_paths={shard.root_path} if shard.is_branch_root else set(),
        require_owner_in_scope=False,
    )
    if not issues:
        try:
            derive_module_forest(repository, fragment)
        except ValueError as exc:
            issues.append(
                AssignmentIssue(
                    code="public_tree_invalid",
                    detail=f"provisional public tree failed validation: {exc}",
                )
            )
    return issues


_TStage = TypeVar("_TStage", bound=BaseModel)


async def _run_stage_with_api_retry(
    *,
    output_type: type[_TStage],
    repo_path: Path,
    prompt: str,
    stage_name: str,
    label: str,
    base: Path | None,
    attempt_rel: str,
    timeout_s: int,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    stage_tag: str,
    on_event: Callable[[str], None] | None,
) -> tuple[ClaudeStageResult[_TStage], str]:
    """One structured stage call, retrying API-level failures with backoff.

    `run_structured_claude_stage` tags its `ExtractorAgentError` with
    `context["api_failure"]` when the CLI died on the API transport (rate
    limit, request timeout, server overload) rather than on anything repo- or
    prompt-specific. By then the CLI has already burned its internal retry
    budget, so this wrapper waits out the rate-limit window
    (`enrich_api_backoff_s`, doubling per retry) and starts a fresh session,
    up to `enrich_api_retries` extra attempts. Any other failure — and
    exhaustion — re-raises unchanged, preserving fail-fast.

    Each try keeps its own artifact directory (`<attempt_rel>`,
    `<attempt_rel>_api2`, ...): a failed try's stream stays on disk as
    evidence. Returns the result plus the rel of the directory that produced
    it, so the caller's follow-up artifacts land next to the winning stream.
    The `await asyncio.sleep` keeps a backoff-parked shard promptly
    cancellable when a sibling hard-fails.
    """
    delay = config.enrich_api_backoff_s
    total_tries = config.enrich_api_retries + 1
    for api_try in range(1, total_tries + 1):
        rel = attempt_rel if api_try == 1 else f"{attempt_rel}_api{api_try}"
        try:
            result: ClaudeStageResult[_TStage] = await asyncio.to_thread(
                run_structured_claude_stage,
                output_type=output_type,
                repo_path=repo_path,
                prompt=prompt,
                stage_name=stage_name,
                attempt_dir=base / rel if base is not None else None,
                claude_bin=config.claude_bin,
                claude_model=config.claude_model,
                max_turns=config.max_turns,
                timeout_s=timeout_s,
                on_event=on_event,
            )
        except ExtractorAgentError as exc:
            reason = exc.context.get("api_failure")
            failed_telemetry = exc.context.get("telemetry")
            if isinstance(failed_telemetry, StageTelemetry):
                suffix = f"#api{api_try}" if reason is not None else "#failed"
                telemetry.add_stage(f"{stage_tag}{suffix}", failed_telemetry)
                _persist_sessions(base, telemetry)
            if reason is None or api_try >= total_tries:
                raise
            notify(
                on_event,
                f"extractor: {label} hit an API failure ({reason}); "
                f"retrying in {delay:.0f}s "
                f"({api_try}/{config.enrich_api_retries})",
            )
            await asyncio.sleep(delay)
            delay *= 2
            continue
        return result, rel
    raise AssertionError("unreachable: API retry loop exhausted")


async def _assignment_shard_attempts(
    shard: AssignmentShard,
    *,
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    precedence: CollisionPrecedence,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    failure_snapshots: dict[str, _AssignmentFailureSnapshot],
    on_event: Callable[[str], None] | None,
) -> _AssignmentShardOutcome:
    shard_rel = f"03_enrich/{shard.key}"
    repo_data = repository.model_dump(mode="json")
    subtree_data = shard.subtree.model_dump(mode="json")
    scope = _assignment_scope_dict(shard, precedence)
    base_prompt = render_assignment_shard_prompt(
        repository=repo_data,
        subtree=subtree_data,
        scope=scope,
        merge_threshold=config.merge_threshold,
    )
    _required_write_json(base, f"{shard_rel}/scope.json", scope)
    _required_write_json(base, f"{shard_rel}/subtree_skeleton.json", subtree_data)
    _required_write_text(base, f"{shard_rel}/prompt.md", base_prompt)

    # A shard derivation that could not cut down to size keeps the full budget.
    cut_down_to_size = (
        assignment_shard_weight(shard) <= config.enrich_subshard_threshold
    )
    timeout_s = (
        _effective_timeout(config.enrich_timeout_s, config.timeout_s)
        if cut_down_to_size and not covers_entire_skeleton(shard, skeleton)
        else config.timeout_s
    )

    prompt = base_prompt
    repaired_kinds: set[str] = set()
    last_parsed: AssignmentTree | None = None
    last_issues: list[AssignmentIssue] | None = None

    def _record_failure() -> None:
        if last_parsed is None:
            return
        failure_snapshots[shard.key] = _AssignmentFailureSnapshot(
            fragment=last_parsed,
            issues=last_issues,
        )

    for attempt in range(1, _MAX_STAGE3_ATTEMPTS + 1):
        repair_tag = "" if attempt == 1 else "#repair" + (
            str(attempt - 1) if attempt > 2 else ""
        )
        stage_tag = f"03_enrich[{shard.key}]{repair_tag}"
        result: ClaudeStageResult[AssignmentTree]
        result, attempt_rel = await _run_stage_with_api_retry(
            output_type=AssignmentTree,
            repo_path=repo_path,
            prompt=prompt,
            stage_name=f"assign:{shard.key}",
            label=f"shard {shard.key}",
            base=base,
            attempt_rel=f"{shard_rel}/attempt_{attempt:02d}",
            timeout_s=timeout_s,
            config=config,
            telemetry=telemetry,
            stage_tag=stage_tag,
            on_event=on_event,
        )
        telemetry.add_claude(stage_tag, result)

        if result.parsed is None:
            detail = str(result.validation_error)
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {"ok": False, "kind": "parse", "detail": detail},
            )
            _persist_sessions(base, telemetry)
            if _spend_repair("parse", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: shard {shard.key} parse failed; repairing",
                )
                prompt = _repair_prompt(
                    base_prompt, detail, previous_payload=result.raw_payload
                )
                continue
            _record_failure()
            raise ExtractorValidationError(
                f"shard {shard.key!r} failed to parse after {attempt} attempts: "
                f"{result.validation_error}",
                stage="enrich",
                shard=shard.key,
            )

        fragment = result.parsed
        last_parsed = fragment
        last_issues = None
        _required_write_json(
            base,
            f"{attempt_rel}/assignment_tree.json",
            fragment.model_dump(mode="json"),
        )

        # V2 preflight against this shard's slice inventory.
        coverage = compute_assignment_coverage(fragment, shard.subtree)
        _required_write_json(
            base, f"{attempt_rel}/coverage.json", coverage.model_dump(mode="json")
        )
        _persist_sessions(base, telemetry)
        if coverage.missing:
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {
                    "ok": False,
                    "kind": "coverage",
                    "detail": f"{len(coverage.missing)} unlabeled path(s)",
                },
            )
            if _spend_repair("coverage", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: shard {shard.key} missing "
                    f"{len(coverage.missing)} labels; repairing",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    _assignment_coverage_repair_errors(coverage.missing),
                    previous_payload=result.raw_payload,
                )
                continue
            _record_failure()
            raise ExtractorCoverageError(
                f"shard {shard.key!r}: {len(coverage.missing)} paths unlabeled",
                missing=coverage.missing,
                stage="enrich",
                shard=shard.key,
            )

        issues = _validate_assignment_fragment(
            shard,
            fragment,
            repo_path=repo_path,
            repository=repository,
            precedence=precedence,
            merge_threshold=config.merge_threshold,
        )
        last_issues = issues or None
        _required_write_json(
            base,
            f"{attempt_rel}/validation.json",
            {
                "ok": not issues,
                "kind": "validation" if issues else None,
                "issues": [i.model_dump(mode="json") for i in issues],
            },
        )
        if issues:
            if _spend_repair("validation", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: shard {shard.key} validation failed; repairing",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    format_assignment_issues(issues),
                    previous_payload=result.raw_payload,
                )
                continue
            _record_failure()
            raise ExtractorValidationError(
                f"shard {shard.key!r} failed validation: "
                f"{format_assignment_issues(issues)}",
                stage="enrich",
                shard=shard.key,
            )

        _required_write_json(
            base,
            f"{shard_rel}/assignment_fragment.json",
            fragment.model_dump(mode="json"),
        )
        return _AssignmentShardOutcome(
            shard=shard,
            fragment=fragment,
            session_id=result.telemetry.session_id,
            attempts=attempt,
        )

    raise AssertionError("unreachable: assignment shard attempt loop exhausted")


async def _run_one_assignment_shard(
    shard: AssignmentShard,
    *,
    sem: asyncio.Semaphore,
    cancel: _CancelFlag,
    **kwargs: Any,
) -> _AssignmentShardOutcome:
    async with sem:
        if cancel.cancelled:
            raise asyncio.CancelledError(
                f"shard {shard.key!r} cancelled after a sibling failed"
            )
        try:
            return await _assignment_shard_attempts(shard, **kwargs)
        except BaseException:
            cancel.cancelled = True
            raise


async def _run_assignment_shards_async(
    shards: list[AssignmentShard], *, sem: asyncio.Semaphore, **kwargs: Any
) -> list[_AssignmentShardOutcome]:
    cancel = _CancelFlag()
    tasks = [
        asyncio.create_task(
            _run_one_assignment_shard(s, sem=sem, cancel=cancel, **kwargs)
        )
        for s in shards
    ]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        cancel.cancelled = True
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _stage3a_sharded(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    precedence: CollisionPrecedence,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    decision: SourceRootDecision | None,
    on_event: Callable[[str], None] | None,
) -> AssignmentTree:
    plan = derive_assignment_shards(skeleton, config)
    _required_write_json(
        base, "03_enrich/shards.json", _assignment_shard_plan_dict(plan)
    )
    _required_write_text(
        base, "03_enrich/sharding.md", render_assignment_shard_plan_markdown(plan)
    )
    notify(
        on_event,
        f"extractor: stage-3A sharded into {len(plan.shards)} shard(s) "
        f"across {len(plan.branch_roots)} top-level branch(es)",
    )
    failure_snapshots: dict[str, _AssignmentFailureSnapshot] = {}

    async def _run() -> list[_AssignmentShardOutcome]:
        return await _run_assignment_shards_async(
            plan.shards,
            sem=asyncio.Semaphore(config.max_parallel_enrich_shards),
            repo_path=repo_path,
            repository=repository,
            skeleton=skeleton,
            precedence=precedence,
            base=base,
            config=config,
            telemetry=telemetry,
            failure_snapshots=failure_snapshots,
            on_event=on_event,
        )

    try:
        outcomes = _run_event_loop(_run())
    except BaseException:
        _best_effort_write_json(base, "sessions.json", telemetry.as_dict(), on_event)
        if failure_snapshots:
            failed_key = min(failure_snapshots)
            snapshot = failure_snapshots[failed_key]
            _write_tree_decisions_v2(
                base,
                required=False,
                skeleton=skeleton,
                assignments=snapshot.fragment,
                resolved=None,
                metadata=None,
                coverage=compute_assignment_coverage(
                    snapshot.fragment, skeleton, invalid=snapshot.issues
                ),
                lints=None,
                issues=snapshot.issues,
                decision=decision,
                merge_threshold=config.merge_threshold,
                on_event=on_event,
            )
        raise
    _persist_sessions(base, telemetry)

    pairs = [(o.shard, o.fragment) for o in outcomes]
    accepted = {o.shard.key: o.session_id for o in outcomes}

    # Per-branch totality precheck: cheap losslessness + blame localizer.
    for branch_key in plan.branch_keys():
        branch_pairs = [
            (s, f) for s, f in pairs if plan.branch_key_of(s) == branch_key
        ]
        report = assignment_branch_coverage(plan, branch_key, branch_pairs)
        _required_write_json(
            base,
            f"03_enrich/branches/{branch_key}/coverage.json",
            report.model_dump(mode="json"),
        )
        if report.missing:
            owners = {
                p: (
                    assignment_owning_shard(p, [s for s, _ in branch_pairs])
                    or branch_pairs[0][0]
                ).key
                for p in report.missing
            }
            raise ExtractorCoverageError(
                f"branch {branch_key!r} is missing {len(report.missing)} "
                f"label(s) after union: {owners}",
                missing=report.missing,
                stage="enrich",
                branch=branch_key,
            )

    merged = merge_assignment_fragments(pairs)
    _required_write_json(
        base,
        "03_enrich/merged/assignment_tree.json",
        merged.model_dump(mode="json"),
    )

    primary = plan.primary_shard()
    telemetry.accepted_session_id = (
        accepted.get(primary.key) if primary is not None else None
    )
    notify(
        on_event,
        f"extractor: united {len(pairs)} assignment fragment(s) covering "
        f"{len(merged.assignments)} path(s)",
    )
    return merged


# ── Stage 3B — metadata batches ───────────────────────────────────────────


@dataclass
class _MetadataFailureSnapshot:
    modules: dict[str, ModuleInfo] | None
    issues: list[AssignmentIssue] | None


async def _metadata_batch_attempts(
    batch: MetadataBatch,
    *,
    repo_path: Path,
    repository: Any,
    resolved: ResolvedAssignmentTree,
    failure_snapshots: dict[str, _MetadataFailureSnapshot],
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    on_event: Callable[[str], None] | None,
) -> tuple[MetadataBatch, dict[str, ModuleInfo], str | None]:
    batch_rel = f"03_enrich/metadata/{batch.key}"
    repo_data = repository.model_dump(mode="json")
    scope_payload = {
        "key": batch.key,
        "requested_modules": list(batch.module_paths),
        "modules": [s.model_dump(mode="json") for s in batch.scopes],
    }
    base_prompt = render_metadata_prompt(
        repository=repo_data, scopes=scope_payload
    )
    _required_write_json(base, f"{batch_rel}/scope.json", scope_payload)
    _required_write_text(base, f"{batch_rel}/prompt.md", base_prompt)

    requested = set(batch.module_paths)
    module_paths = resolved.module_paths()
    timeout_s = (
        config.timeout_s
        if batch.use_full_timeout
        else _effective_timeout(config.enrich_timeout_s, config.timeout_s)
    )

    prompt = base_prompt
    repaired_kinds: set[str] = set()
    last_modules: dict[str, ModuleInfo] | None = None
    last_issues: list[AssignmentIssue] | None = None

    def _record_failure() -> None:
        failure_snapshots[batch.key] = _MetadataFailureSnapshot(
            modules=last_modules,
            issues=last_issues,
        )

    for attempt in range(1, _MAX_STAGE3_ATTEMPTS + 1):
        repair_tag = "" if attempt == 1 else "#repair" + (
            str(attempt - 1) if attempt > 2 else ""
        )
        stage_tag = f"03_metadata[{batch.key}]{repair_tag}"
        result: ClaudeStageResult[ModuleMetadataBatch]
        result, attempt_rel = await _run_stage_with_api_retry(
            output_type=ModuleMetadataBatch,
            repo_path=repo_path,
            prompt=prompt,
            stage_name=f"metadata:{batch.key}",
            label=f"metadata batch {batch.key}",
            base=base,
            attempt_rel=f"{batch_rel}/attempt_{attempt:02d}",
            timeout_s=timeout_s,
            config=config,
            telemetry=telemetry,
            stage_tag=stage_tag,
            on_event=on_event,
        )
        telemetry.add_claude(stage_tag, result)

        if result.parsed is None:
            detail = str(result.validation_error)
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {"ok": False, "kind": "parse", "detail": detail},
            )
            _persist_sessions(base, telemetry)
            if _spend_repair("parse", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: metadata batch {batch.key} parse failed; repairing",
                )
                prompt = _repair_prompt(
                    base_prompt, detail, previous_payload=result.raw_payload
                )
                continue
            _record_failure()
            raise ExtractorValidationError(
                f"metadata batch {batch.key!r} failed to parse after {attempt} "
                f"attempts: {result.validation_error}",
                stage="metadata",
                batch=batch.key,
            )

        parsed = result.parsed
        last_modules = dict(parsed.modules)
        last_issues = None
        _required_write_json(
            base,
            f"{attempt_rel}/module_metadata.json",
            parsed.model_dump(mode="json"),
        )
        coverage = compute_metadata_coverage(parsed.modules, requested)
        _required_write_json(
            base,
            f"{attempt_rel}/metadata_coverage.json",
            coverage.model_dump(mode="json"),
        )
        _persist_sessions(base, telemetry)

        if coverage.missing:
            last_issues = [
                AssignmentIssue(
                    code="metadata_missing",
                    path=path,
                    detail=f"requested module {path!r} has no metadata entry",
                )
                for path in coverage.missing
            ]
            _required_write_json(
                base,
                f"{attempt_rel}/validation.json",
                {
                    "ok": False,
                    "kind": "coverage",
                    "detail": (
                        f"missing metadata for {len(coverage.missing)} "
                        "requested module(s)"
                    ),
                },
            )
            if _spend_repair("coverage", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: metadata batch {batch.key} missing "
                    f"{len(coverage.missing)} module(s); repairing",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    "Missing metadata for these requested modules — `modules` "
                    "must contain exactly the requested module set: "
                    + ", ".join(coverage.missing),
                    previous_payload=result.raw_payload,
                )
                continue
            _record_failure()
            raise ExtractorCoverageError(
                f"metadata batch {batch.key!r}: {len(coverage.missing)} "
                "requested module(s) missing",
                missing=coverage.missing,
                stage="metadata",
                batch=batch.key,
            )

        issues = validate_metadata_entries(
            parsed.modules, requested, module_paths, repo_path
        )
        last_issues = issues or None
        _required_write_json(
            base,
            f"{attempt_rel}/validation.json",
            {
                "ok": not issues,
                "kind": "validation" if issues else None,
                "issues": [i.model_dump(mode="json") for i in issues],
            },
        )
        if issues:
            if _spend_repair("validation", repaired_kinds, attempt):
                notify(
                    on_event,
                    f"extractor: metadata batch {batch.key} validation failed; "
                    "repairing",
                )
                prompt = _repair_prompt(
                    base_prompt,
                    format_assignment_issues(issues),
                    previous_payload=result.raw_payload,
                )
                continue
            _record_failure()
            raise ExtractorValidationError(
                f"metadata batch {batch.key!r} failed validation: "
                f"{format_assignment_issues(issues)}",
                stage="metadata",
                batch=batch.key,
            )

        _required_write_json(
            base,
            f"{batch_rel}/module_metadata.json",
            parsed.model_dump(mode="json"),
        )
        return batch, dict(parsed.modules), result.telemetry.session_id

    raise AssertionError("unreachable: metadata batch attempt loop exhausted")


async def _run_one_metadata_batch(
    batch: MetadataBatch,
    *,
    sem: asyncio.Semaphore,
    cancel: _CancelFlag,
    **kwargs: Any,
) -> tuple[MetadataBatch, dict[str, ModuleInfo], str | None]:
    async with sem:
        if cancel.cancelled:
            raise asyncio.CancelledError(
                f"metadata batch {batch.key!r} cancelled after a sibling failed"
            )
        try:
            return await _metadata_batch_attempts(batch, **kwargs)
        except BaseException:
            cancel.cancelled = True
            raise


def _stage3b_metadata(
    repo_path: Path,
    repository: Any,
    skeleton: Skeleton,
    resolved: ResolvedAssignmentTree,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    decision: SourceRootDecision | None,
    assignment_tree: AssignmentTree,
    coverage: Any,
    lints: list[AssignmentLint],
    on_event: Callable[[str], None] | None,
) -> dict[str, ModuleInfo]:
    """Run the deterministic metadata batches and gate their exact union.

    Batching is derived from final territories (never the Stage-3A shard
    plan), so a nested assignment shard that resolved `PART` contributes its
    files to the external owner's scope automatically.
    """
    plan = derive_metadata_batches(resolved, skeleton, config)
    _required_write_json(
        base, "03_enrich/metadata/plan.json", plan.model_dump(mode="json")
    )
    module_paths = resolved.module_paths()

    outcomes: list[tuple[MetadataBatch, dict[str, ModuleInfo], str | None]] = []
    failure_snapshots: dict[str, _MetadataFailureSnapshot] = {}
    if plan.batches:
        notify(
            on_event,
            f"extractor: stage-3B batched {len(module_paths)} module(s) into "
            f"{len(plan.batches)} metadata call(s)",
        )

        async def _run() -> list[
            tuple[MetadataBatch, dict[str, ModuleInfo], str | None]
        ]:
            cancel = _CancelFlag()
            sem = asyncio.Semaphore(config.max_parallel_enrich_shards)
            tasks = [
                asyncio.create_task(
                    _run_one_metadata_batch(
                        b,
                        sem=sem,
                        cancel=cancel,
                        repo_path=repo_path,
                        repository=repository,
                        resolved=resolved,
                        failure_snapshots=failure_snapshots,
                        base=base,
                        config=config,
                        telemetry=telemetry,
                        on_event=on_event,
                    )
                )
                for b in plan.batches
            ]
            try:
                return list(await asyncio.gather(*tasks))
            except BaseException:
                cancel.cancelled = True
                for t in tasks:
                    t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        try:
            outcomes = _run_event_loop(_run())
        except BaseException:
            _best_effort_write_json(
                base, "sessions.json", telemetry.as_dict(), on_event
            )
            if failure_snapshots:
                failed_key = min(failure_snapshots)
                snapshot = failure_snapshots[failed_key]
                _write_tree_decisions_v2(
                    base,
                    required=False,
                    skeleton=skeleton,
                    assignments=assignment_tree,
                    resolved=resolved,
                    metadata=snapshot.modules,
                    coverage=coverage,
                    lints=lints,
                    issues=snapshot.issues,
                    decision=decision,
                    merge_threshold=config.merge_threshold,
                    on_event=on_event,
                )
            else:
                _best_effort_tree_decisions_v2_metadata_failure(
                    base,
                    skeleton=skeleton,
                    assignment_tree=assignment_tree,
                    resolved=resolved,
                    coverage=coverage,
                    lints=lints,
                    decision=decision,
                    merge_threshold=config.merge_threshold,
                    on_event=on_event,
                )
            raise
        _persist_sessions(base, telemetry)

    # Exact disjoint union in stable batch/path order.
    union: dict[str, ModuleInfo] = {}
    for _batch, modules, _session in sorted(outcomes, key=lambda o: o[0].key):
        for path in sorted(modules):
            if path in union:
                raise ExtractorValidationError(
                    f"metadata batches both returned {path!r}; the batch "
                    "partition is broken",
                    stage="metadata",
                )
            union[path] = modules[path]

    metadata_cov = compute_metadata_coverage(union, module_paths)
    _required_write_json(
        base,
        "03_enrich/metadata/metadata_coverage.json",
        metadata_cov.model_dump(mode="json"),
    )
    _required_write_json(
        base, "metadata_coverage.json", metadata_cov.model_dump(mode="json")
    )

    claimed: dict[str, str] = {}
    issues = validate_metadata_entries(
        union, module_paths, module_paths, repo_path, claimed=claimed
    )
    v5_error: str | None = None
    if not issues and not metadata_cov.missing and not metadata_cov.extra:
        try:
            derive_project_tree(repository, resolved, union)
        except (CrossArtifactError, ValueError) as exc:
            v5_error = str(exc)
    _required_write_json(
        base,
        "03_enrich/metadata/validation.json",
        {
            "ok": not issues and metadata_cov.ok and not metadata_cov.extra
            and v5_error is None,
            "issues": [i.model_dump(mode="json") for i in issues],
            "v5_error": v5_error,
        },
    )

    if metadata_cov.missing or metadata_cov.extra or issues or v5_error:
        _write_tree_decisions_v2(
            base,
            required=False,
            skeleton=skeleton,
            assignments=assignment_tree,
            resolved=resolved,
            metadata=union,
            coverage=coverage,
            lints=lints,
            issues=issues or None,
            decision=decision,
            merge_threshold=config.merge_threshold,
            on_event=on_event,
        )
        problems: list[str] = []
        if metadata_cov.missing:
            problems.append(f"missing metadata for {metadata_cov.missing}")
        if metadata_cov.extra:
            problems.append(f"metadata for unrequested {metadata_cov.extra}")
        if issues:
            problems.append(format_assignment_issues(issues))
        if v5_error:
            problems.append(v5_error)
        raise ExtractorValidationError(
            f"metadata union failed the final gate: {'; '.join(problems)}",
            stage="metadata",
        )

    payload = ModuleMetadataMap(modules=union).model_dump(mode="json")
    _required_write_json(base, "03_enrich/module_metadata.json", payload)
    _required_write_json(base, "module_metadata.json", payload)
    return union


def _best_effort_tree_decisions_v2_metadata_failure(
    base: Path | None,
    *,
    skeleton: Skeleton,
    assignment_tree: AssignmentTree,
    resolved: ResolvedAssignmentTree,
    coverage: Any,
    lints: list[AssignmentLint],
    decision: SourceRootDecision | None,
    merge_threshold: int,
    on_event: Callable[[str], None] | None,
) -> None:
    """An exhausted metadata batch still gets the assignment-side report."""
    _write_tree_decisions_v2(
        base,
        required=False,
        skeleton=skeleton,
        assignments=assignment_tree,
        resolved=resolved,
        metadata=None,
        coverage=coverage,
        lints=lints,
        issues=None,
        decision=decision,
        merge_threshold=merge_threshold,
        on_event=on_event,
    )


# ── Stage 5 — deterministic assembly ──────────────────────────────────────


def _stage5_assignments_assemble(
    repo_path: Path,
    repository: Any,
    decision: SourceRootDecision,
    skeleton: Skeleton,
    resolved: ResolvedAssignmentTree,
    metadata: dict[str, ModuleInfo],
    *,
    base: Path | None,
    repo_fingerprint: str,
    on_event: Callable[[str], None] | None,
) -> ProjectTree:
    """Deterministic final assembly with the full defense set: recompute and
    compare the resolved maps, revalidate metadata and the derived tree, then
    the shared repo-mutation defenses, then — and only then — the write."""
    try:
        verify_resolved_assignments(resolved, skeleton)
    except CrossArtifactError as exc:
        raise ExtractorValidationError(
            f"resolved assignments failed recomputation at assembly: {exc}",
            stage="assemble",
        ) from exc

    module_paths = resolved.module_paths()
    metadata_cov = compute_metadata_coverage(metadata, module_paths)
    issues = validate_metadata_entries(
        metadata, module_paths, module_paths, repo_path
    )
    if metadata_cov.missing or metadata_cov.extra or issues:
        raise ExtractorValidationError(
            "metadata failed revalidation at assembly: "
            f"missing={metadata_cov.missing} extra={metadata_cov.extra} "
            f"{format_assignment_issues(issues)}",
            stage="assemble",
        )

    try:
        tree = derive_project_tree(repository, resolved, metadata)
    except (CrossArtifactError, ValueError) as exc:
        raise ExtractorValidationError(
            f"final tree failed validation at assembly: {exc}",
            stage="assemble",
        ) from exc

    _verify_repo_unchanged(
        repo_path, decision, skeleton, repo_fingerprint=repo_fingerprint
    )

    if base is not None:
        try:
            tree.to_json(base / "project_tree.json")
        except OSError as exc:
            raise ExtractorArtifactError(
                f"failed to write project_tree.json: {exc}",
                artifact="project_tree.json",
            ) from exc
    notify(on_event, f"extractor: assembled {len(tree.modules)} top-level modules")
    return tree


# ── Stage 3 pipeline ───────────────────────────────────────────────────────


def _run_assignment_extraction(
    repo_path: Path,
    repository: Any,
    decision: SourceRootDecision,
    skeleton: Skeleton,
    *,
    base: Path | None,
    config: ExtractorConfig,
    telemetry: _Telemetry,
    repo_fingerprint: str,
    on_event: Callable[[str], None] | None,
) -> ProjectTree:
    """Stage 3A (assignment), resolution, Stage 3B (metadata), and Stage 5."""
    stage_dir = base / "03_enrich" if base is not None else None
    if stage_dir is not None:
        stage_dir.mkdir(parents=True, exist_ok=True)
    # Deterministic collision preflight — before any Stage-3 Claude call.
    precedence = compute_collision_precedence(skeleton)
    if precedence.fatal:
        classes = "; ".join(
            ", ".join(repr(p) for p in group)
            for group in precedence.top_level_collisions
        )
        detail = (
            "top-level skeleton paths collide on their normalized public "
            f"qualified name and cannot both be modules: {classes}. This is a "
            "limitation of the existing public naming contract, not a "
            "repairable model choice."
        )
        _write_tree_decisions_v2(
            base,
            required=False,
            skeleton=skeleton,
            assignments=None,
            resolved=None,
            metadata=None,
            coverage=None,
            lints=None,
            issues=[AssignmentIssue(code="top_level_qn_collision", detail=detail)],
            decision=decision,
            merge_threshold=config.merge_threshold,
            on_event=on_event,
        )
        raise ExtractorValidationError(detail, stage="enrich")

    # ── Stage 3A ──────────────────────────────────────────────────────────
    if config.enrich_sharding == "single" and not has_several_source_roots(skeleton):
        assignment_tree = _stage3a_single(
            repo_path,
            repository,
            skeleton,
            precedence,
            base=base,
            config=config,
            telemetry=telemetry,
            decision=decision,
            on_event=on_event,
        )
    else:
        assignment_tree = _stage3a_sharded(
            repo_path,
            repository,
            skeleton,
            precedence,
            base=base,
            config=config,
            telemetry=telemetry,
            decision=decision,
            on_event=on_event,
        )
    _persist_sessions(base, telemetry)

    # Full V1/V2/V3/path-only-V5 gate on the accepted/merged assignment. For
    # the single path this re-checks what its repair loop already accepted;
    # for the sharded path it is the authoritative global gate.
    coverage = compute_assignment_coverage(assignment_tree, skeleton)
    issues = validate_assignment_paths(assignment_tree, skeleton, repo_path)
    issues += validate_assignment_labels(
        assignment_tree,
        skeleton,
        merge_threshold=config.merge_threshold,
        precedence=precedence,
        anchor_paths={n.path for n in skeleton.nodes},
        require_owner_in_scope=True,
    )
    v5_error: str | None = None
    if not issues and not coverage.missing:
        try:
            derive_module_forest(repository, assignment_tree)
        except ValueError as exc:
            v5_error = str(exc)
    _required_write_json(
        base, "03_enrich/coverage.json", coverage.model_dump(mode="json")
    )
    _required_write_json(
        base,
        "03_enrich/validation.json",
        {
            "ok": not issues and not coverage.missing and v5_error is None,
            "issues": [i.model_dump(mode="json") for i in issues],
            "v5_error": v5_error,
        },
    )
    if coverage.missing or issues or v5_error:
        _write_tree_decisions_v2(
            base,
            required=False,
            skeleton=skeleton,
            assignments=assignment_tree,
            resolved=None,
            metadata=None,
            coverage=compute_assignment_coverage(
                assignment_tree, skeleton, invalid=issues
            ),
            lints=None,
            issues=issues or None,
            decision=decision,
            merge_threshold=config.merge_threshold,
            on_event=on_event,
        )
        if coverage.missing:
            raise ExtractorCoverageError(
                f"{len(coverage.missing)} inventory paths unlabeled after "
                "Stage 3A",
                missing=coverage.missing,
                stage="enrich",
            )
        raise ExtractorValidationError(
            "merged assignment failed validation: "
            + (format_assignment_issues(issues) if issues else str(v5_error)),
            stage="enrich",
        )

    try:
        resolved = resolve_assignments(
            assignment_tree, skeleton, config.merge_threshold
        )
    except CrossArtifactError as exc:
        _write_tree_decisions_v2(
            base,
            required=False,
            skeleton=skeleton,
            assignments=assignment_tree,
            resolved=None,
            metadata=None,
            coverage=coverage,
            lints=None,
            issues=None,
            decision=decision,
            merge_threshold=config.merge_threshold,
            on_event=on_event,
        )
        raise ExtractorValidationError(
            f"assignment resolution failed: {exc}", stage="enrich"
        ) from exc

    lints = compute_assignment_lints(resolved, skeleton)
    _required_write_json(
        base, "03_enrich/assignment_tree.json", assignment_tree.model_dump(mode="json")
    )
    _required_write_json(
        base,
        "03_enrich/resolved_assignments.json",
        resolved.model_dump(mode="json"),
    )
    _required_write_json(
        base,
        "03_enrich/assignment_lints.json",
        [lint.model_dump(mode="json") for lint in lints],
    )
    _required_write_json(
        base, "assignment_tree.json", assignment_tree.model_dump(mode="json")
    )
    _required_write_json(
        base, "resolved_assignments.json", resolved.model_dump(mode="json")
    )
    _required_write_json(base, "coverage.json", coverage.model_dump(mode="json"))
    _required_write_json(
        base,
        "assignment_lints.json",
        [lint.model_dump(mode="json") for lint in lints],
    )
    notify(
        on_event,
        f"extractor: resolved {len(resolved.module_paths())} module(s) over "
        f"{len(resolved.assignments)} path(s); {len(lints)} lint(s)",
    )

    # ── Stage 3B ──────────────────────────────────────────────────────────
    metadata = _stage3b_metadata(
        repo_path,
        repository,
        skeleton,
        resolved,
        base=base,
        config=config,
        telemetry=telemetry,
        decision=decision,
        assignment_tree=assignment_tree,
        coverage=coverage,
        lints=lints,
        on_event=on_event,
    )
    _persist_sessions(base, telemetry)

    # ── Stage 5 ───────────────────────────────────────────────────────────
    tree = _stage5_assignments_assemble(
        repo_path,
        repository,
        decision,
        skeleton,
        resolved,
        metadata,
        base=base,
        repo_fingerprint=repo_fingerprint,
        on_event=on_event,
    )

    _write_tree_decisions_v2(
        base,
        required=True,
        skeleton=skeleton,
        assignments=assignment_tree,
        resolved=resolved,
        metadata=metadata,
        coverage=coverage,
        lints=lints,
        issues=None,
        decision=decision,
        merge_threshold=config.merge_threshold,
        on_event=on_event,
    )
    return tree


__all__ = ["run_two_phase_extraction"]
